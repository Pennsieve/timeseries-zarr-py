"""Reader for the ZEP2 sharded arrays that shards.py writes.

The functions here read the layout that shards.py describes: the encoded inner
chunks, then the index, then the CRC-32C of the index. They take bytes that the
caller fetched and never touch a store. Each chunk is the little-endian bytes
of its rows, compressed with Zstd.
"""

import struct
import threading
import weakref
from collections import OrderedDict
from collections.abc import Buffer
from typing import Final

import google_crc32c
import numpy as np
import numpy.typing as npt
from numcodecs.zstd import Zstd

ShardIndex = npt.NDArray[np.uint64]
"""The (offset, length) pairs of a shard, shape (chunks_per_shard, 2), read-only.

A slot of (2^64 - 1, 2^64 - 1) holds no chunk.
"""

_MISSING: Final = 2**64 - 1
_SLOT_BYTES: Final = 16
_CHECKSUM_BYTES: Final = 4
_ZSTD_MAGIC: Final = bytes.fromhex("28b52ffd")
_ZSTD_MINIMUM_HEADER_BYTES: Final = 6
_ZSTD_SINGLE_SEGMENT: Final = 0x20
_ZSTD_DICTIONARY_BYTES: Final = (0, 1, 2, 4)
_ZSTD_CONTENT_SIZE_BYTES: Final = (1, 2, 4, 8)
_ZSTD = Zstd()

type _CacheKey = tuple[int, str, int]


def parse_index(
    data: Buffer, chunks_per_shard: int, array_path: str, shard: int
) -> ShardIndex:
    """Check the index of a shard and return its entries.

    Args:
        data: The index bytes and the checksum that follows them, the last
            16 * chunks_per_shard + 4 bytes of the shard object.
        chunks_per_shard: The chunk slots of a full shard.
        array_path: The path of the array, for error messages.
        shard: The shard number along axis 0, for error messages.

    Returns:
        The offset and length of each slot. A missing chunk reads
        (2^64 - 1, 2^64 - 1).

    Raises:
        ValueError: If data has the wrong length, if the checksum does not
            match, or if an entry ends past 2^64 - 1.
    """
    view = memoryview(data).cast("B")
    table_bytes = _SLOT_BYTES * chunks_per_shard
    expected = table_bytes + _CHECKSUM_BYTES
    where = f"shard {shard} of array {array_path}"
    if len(view) != expected:
        raise ValueError(
            f"the index of {where} has {len(view)} bytes, expected {expected} "
            f"for {chunks_per_shard} chunk slots"
        )
    stored = struct.unpack("<I", view[table_bytes:])[0]
    actual = google_crc32c.value(bytes(view[:table_bytes]))
    if stored != actual:
        raise ValueError(
            f"the index of {where} has CRC-32C {stored:#010x}, expected "
            f"{actual:#010x} from its entries"
        )
    flat = np.frombuffer(view[:table_bytes], dtype="<u8").astype(np.uint64)
    index = flat.reshape(chunks_per_shard, 2)
    offsets, lengths = index[:, 0], index[:, 1]
    present = ~((offsets == _MISSING) & (lengths == _MISSING))
    if (present & (offsets + lengths < offsets)).any():
        raise ValueError(
            f"the index of {where} has an entry that ends past 2^64 - 1, "
            "expected offset + length below 2^64"
        )
    index.setflags(write=False)
    return index


def chunk_span(
    index: ShardIndex, first_slot: int, stop_slot: int
) -> tuple[int, int] | None:
    """Return the byte range that holds the chunks of a run of slots.

    The range runs from the lowest offset to the highest chunk end among the
    present chunks, whatever order the chunks lie in.

    Args:
        index: The index of the shard.
        first_slot: The first slot of the run.
        stop_slot: One past the last slot of the run.

    Returns:
        The start and the end of the range in the shard object, the end
        exclusive. None if no slot of the run holds a chunk.

    Raises:
        ValueError: If the run is not within the slots of the index.
    """
    _check_slots(index, first_slot, stop_slot)
    entries = index[first_slot:stop_slot]
    present = ~((entries[:, 0] == _MISSING) & (entries[:, 1] == _MISSING))
    if not present.any():
        return None
    offsets = entries[present, 0]
    return int(offsets.min()), int((offsets + entries[present, 1]).max())


def decode_chunks(
    data: Buffer,
    data_offset: int,
    index: ShardIndex,
    first_slot: int,
    stop_slot: int,
    out: npt.NDArray[np.generic],
    fill_value: float,
    chunk_rows: int,
    array_path: str,
    shard: int,
) -> None:
    """Decode the chunks of a run of slots into out.

    Slot first_slot + k fills rows [k * chunk_rows, (k + 1) * chunk_rows) of
    out. A missing chunk fills its rows with fill_value. Each chunk comes from
    its own offset in data, so the chunks may lie in any order.

    Args:
        data: Bytes of the shard object that hold every present chunk of the
            run. The result of chunk_span covers them.
        data_offset: The offset of data[0] in the shard object.
        index: The index of the shard.
        first_slot: The first slot of the run.
        stop_slot: One past the last slot of the run.
        out: C-contiguous, writable, little-endian array of
            (stop_slot - first_slot) * chunk_rows rows.
        fill_value: The value of the rows of a missing chunk.
        chunk_rows: The rows of one inner chunk.
        array_path: The path of the array, for error messages.
        shard: The shard number along axis 0, for error messages.

    Raises:
        ValueError: If a chunk lies outside data, if a chunk does not decode,
            if a chunk decodes to other than chunk_rows rows, or if the run or
            out do not fit the arguments.
    """
    _check_slots(index, first_slot, stop_slot)
    rows = (stop_slot - first_slot) * chunk_rows
    if (
        out.shape[0] != rows
        or not out.flags.c_contiguous
        or not out.flags.writeable
        or out.dtype != out.dtype.newbyteorder("<")
    ):
        raise ValueError(
            f"out must be a writable C-contiguous little-endian array of "
            f"{rows} rows, got {out.shape[0]} rows of dtype {out.dtype}"
        )
    where = f"shard {shard} of array {array_path}"
    view = memoryview(data).cast("B")
    chunk_bytes = out[:chunk_rows].nbytes
    for slot in range(first_slot, stop_slot):
        begin = (slot - first_slot) * chunk_rows
        target = out[begin : begin + chunk_rows]
        offset, length = (int(value) for value in index[slot])
        if offset == _MISSING and length == _MISSING:
            target[...] = fill_value
            continue
        start = offset - data_offset
        if start < 0 or start + length > len(view):
            raise ValueError(
                f"chunk {slot} of {where} lies at bytes [{offset}, "
                f"{offset + length}) and the data holds "
                f"[{data_offset}, {data_offset + len(view)}), "
                "expected every chunk inside the data (the shard may be cut "
                "short)"
            )
        _decode_chunk(
            view[start : start + length], target, chunk_bytes, slot, where
        )


class ShardIndexCache:
    """A least-recently-used map of shard indexes, bounded in bytes.

    An entry costs the bytes of its index table plus the 4 bytes of its
    checksum, so the index of a shard with 512 chunk slots costs 8196 bytes.
    An entry is keyed by the identity of its store, the path of the array,
    and the shard number. A weak-reference callback drops the entries of a
    store when Python collects it. A store that Python cannot weakly reference
    holds no entries, because nothing could drop them. Every method is safe to
    call from several threads.

    Attributes:
        budget_bytes: The most bytes the entries may cost together.
    """

    def __init__(self, budget_bytes: int) -> None:
        """Create an empty cache.

        Args:
            budget_bytes: The most bytes the entries may cost together.
        """
        self.budget_bytes = budget_bytes
        self._lock = threading.RLock()
        self._entries: OrderedDict[_CacheKey, ShardIndex] = OrderedDict()
        self._bytes = 0
        self._finalizers: dict[int, weakref.finalize[[int], object]] = {}

    @property
    def size_bytes(self) -> int:
        """The bytes that the entries cost now."""
        with self._lock:
            return self._bytes

    def __len__(self) -> int:
        """Return the number of entries."""
        with self._lock:
            return len(self._entries)

    def get(
        self, store: object, array_path: str, shard: int
    ) -> ShardIndex | None:
        """Return the index of a shard and mark it as recently used.

        Args:
            store: The store that holds the array.
            array_path: The path of the array in the store.
            shard: The shard number along axis 0.

        Returns:
            The index, or None when the cache holds none.
        """
        key = (id(store), array_path, shard)
        with self._lock:
            index = self._entries.get(key)
            if index is not None:
                self._entries.move_to_end(key)
            return index

    def put(
        self, store: object, array_path: str, shard: int, index: ShardIndex
    ) -> None:
        """Store the index of a shard and drop the oldest entries over budget.

        An index that costs more than the budget is not stored. A store that
        cannot be weakly referenced stores nothing.

        Args:
            store: The store that holds the array.
            array_path: The path of the array in the store.
            shard: The shard number along axis 0.
            index: The index to keep.
        """
        cost = _entry_bytes(index)
        if cost > self.budget_bytes:
            return
        key = (id(store), array_path, shard)
        with self._lock:
            if key[0] not in self._finalizers:
                try:
                    self._finalizers[key[0]] = weakref.finalize(
                        store, self.evict_store_id, key[0]
                    )
                except TypeError:
                    return
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= _entry_bytes(previous)
            self._entries[key] = index
            self._bytes += cost
            while self._bytes > self.budget_bytes:
                _, oldest = self._entries.popitem(last=False)
                self._bytes -= _entry_bytes(oldest)

    def evict_store(self, store: object) -> None:
        """Drop every entry of a store and leave the entries of others.

        Args:
            store: The store whose entries to drop.
        """
        self.evict_store_id(id(store))

    def evict_store_id(self, store_id: int) -> None:
        """Drop every entry of the store that id() names.

        Args:
            store_id: The identity of the store whose entries to drop.
        """
        with self._lock:
            finalizer = self._finalizers.pop(store_id, None)
            if finalizer is not None:
                finalizer.detach()
            for key in [key for key in self._entries if key[0] == store_id]:
                self._bytes -= _entry_bytes(self._entries.pop(key))


def _entry_bytes(index: ShardIndex) -> int:
    """Return the bytes that an index costs in the cache."""
    return index.nbytes + _CHECKSUM_BYTES


def _check_slots(index: ShardIndex, first_slot: int, stop_slot: int) -> None:
    """Raise ValueError unless 0 <= first_slot <= stop_slot <= len(index)."""
    if not 0 <= first_slot <= stop_slot <= index.shape[0]:
        raise ValueError(
            f"the slots [{first_slot}, {stop_slot}) must lie within "
            f"[0, {index.shape[0]}]"
        )


def _decode_chunk(
    frame: memoryview,
    target: npt.NDArray[np.generic],
    chunk_bytes: int,
    slot: int,
    where: str,
) -> None:
    """Decode one Zstd frame into target, which holds chunk_bytes bytes."""
    declared = _frame_content_size(frame)
    try:
        if declared is None:
            decoded = bytes(_ZSTD.decode(frame))
            size = len(decoded)
            if size == chunk_bytes:
                target.reshape(-1).view(np.uint8)[:] = np.frombuffer(
                    decoded, dtype=np.uint8
                )
        else:
            size = declared
            if size == chunk_bytes:
                _ZSTD.decode(frame, out=target.reshape(-1).view(np.uint8))
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            f"chunk {slot} of {where} does not decode as Zstd, expected a "
            f"complete frame of {chunk_bytes} bytes: {error}"
        ) from error
    if size != chunk_bytes:
        raise ValueError(
            f"chunk {slot} of {where} decodes to {size} bytes, expected "
            f"{chunk_bytes} bytes"
        )


def _frame_content_size(frame: memoryview) -> int | None:
    """Return the decoded size that a Zstd frame header declares, or None."""
    if (
        len(frame) < _ZSTD_MINIMUM_HEADER_BYTES
        or bytes(frame[:4]) != _ZSTD_MAGIC
    ):
        return None
    descriptor = frame[4]
    size_flag = descriptor >> 6
    single_segment = bool(descriptor & _ZSTD_SINGLE_SEGMENT)
    if size_flag == 0 and not single_segment:
        return None
    start = (
        5
        + (0 if single_segment else 1)
        + _ZSTD_DICTIONARY_BYTES[descriptor & 3]
    )
    width = _ZSTD_CONTENT_SIZE_BYTES[size_flag]
    field = bytes(frame[start : start + width])
    if len(field) < width:
        return None
    size = int.from_bytes(field, "little")
    return size + 256 if size_flag == 1 else size
