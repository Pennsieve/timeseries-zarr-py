"""Streaming writer for the ZEP2 sharded arrays of a bundle.

A shard is the encoded inner chunks in chunk order, then an index with one
(offset, length) pair of little-endian uint64 per chunk slot, then the CRC-32C
of the index as a little-endian uint32. A slot that holds no chunk reads
(2^64 - 1, 2^64 - 1). Every chunk is the little-endian bytes of its rows,
compressed with Zstd and no frame checksum. These are the bytes that zarr
writes for an array with the sharding codec over the bytes and zstd codecs,
the crc32c index codec, and the index at the end.

The chunks of a shard arrive in order, so a shard file grows by one chunk at a
time and an array holds one partial chunk in memory.
"""

import struct
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import google_crc32c
import numpy as np
import numpy.typing as npt
from numcodecs.zstd import Zstd

from timeseries_zarr.constants import ENCODE_GROUP_BYTES

_MISSING = 2**64 - 1
_CHECKSUM_BYTES = 4


class ShardSink(Protocol):
    """Destination of the bytes of one shard."""

    def append(self, data: bytes) -> None:
        """Add data after the bytes already appended."""
        ...

    def close(self) -> None:
        """Finish the shard. No call follows."""
        ...


class FileShardSink:
    """Appends the bytes of one shard to a file.

    The file opens for each append and closes before the append returns, so a
    sink holds no file descriptor between calls. The first append creates the
    file and its parent directories, and replaces an existing file.
    """

    def __init__(self, path: Path) -> None:
        """Name the file. Nothing touches the disk until the first append.

        Args:
            path: The path of the shard file.
        """
        self._path = path
        self._started = False

    def append(self, data: bytes) -> None:
        """Add data to the end of the file."""
        if not self._started:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("ab" if self._started else "wb") as file:
            file.write(data)
        self._started = True

    def close(self) -> None:
        """Finish the shard. The sink holds no open file."""


class BufferShardSink:
    """Collects the bytes of one shard and hands them over on close."""

    def __init__(self, save: Callable[[bytes], None]) -> None:
        """Set the function that receives the whole shard.

        Args:
            save: Called once, from close, with the bytes of the shard.
        """
        self._save = save
        self._parts: list[bytes] = []

    def append(self, data: bytes) -> None:
        """Keep data after the bytes already kept."""
        self._parts.append(data)

    def close(self) -> None:
        """Call save with the joined bytes and drop them."""
        data = b"".join(self._parts)
        self._parts = []
        self._save(data)


@dataclass(frozen=True)
class ShardLayout:
    """The shape and encoding of one sharded array, as the shard writer needs it.

    Attributes:
        dtype: The element type. Chunks are stored little-endian.
        chunk_rows: The rows of one inner chunk.
        chunks_per_shard: The inner chunks of a full shard.
        trailing_shape: The shape of one row. Empty for a rank-1 array.
        length: The rows of the array. The last chunk pads to chunk_rows with
            fill_value, and the chunks of the last shard past the array end
            are missing.
        zstd_level: The Zstd level of every chunk.
        fill_value: The value that pads a short last chunk.
        open_sink: Returns the sink of the shard with the given index along
            axis 0.
    """

    dtype: np.dtype[np.generic]
    chunk_rows: int
    chunks_per_shard: int
    trailing_shape: tuple[int, ...]
    length: int
    zstd_level: int
    fill_value: float
    open_sink: Callable[[int], ShardSink]

    @property
    def chunk_count(self) -> int:
        """The chunks that hold rows, the last one possibly short."""
        return -(-self.length // self.chunk_rows)


class ChunkEncoder:
    """Encodes the rows of one inner chunk as the bytes codec then Zstd."""

    def __init__(self, layout: ShardLayout) -> None:
        """Build the encoder of an array.

        Args:
            layout: The layout of the array.
        """
        self._chunk_shape = (layout.chunk_rows, *layout.trailing_shape)
        self._dtype = layout.dtype.newbyteorder("<")
        self._fill_value = layout.fill_value
        self._zstd = Zstd(level=layout.zstd_level, checksum=False)

    def encode(self, rows: npt.NDArray[np.generic]) -> bytes:
        """Return the stored bytes of one chunk.

        Args:
            rows: The rows of the chunk, at most chunk_rows of them. A shorter
                block pads with the fill value to a whole chunk.

        Raises:
            ValueError: If rows has more than chunk_rows rows or a different
                trailing shape.
        """
        if rows.shape[1:] != self._chunk_shape[1:]:
            raise ValueError(
                f"a chunk row has shape {self._chunk_shape[1:]}, "
                f"got {rows.shape[1:]}"
            )
        if rows.shape[0] > self._chunk_shape[0]:
            raise ValueError(
                f"a chunk holds at most {self._chunk_shape[0]} rows, "
                f"got {rows.shape[0]}"
            )
        if rows.shape[0] < self._chunk_shape[0]:
            padded = np.full(self._chunk_shape, self._fill_value, self._dtype)
            padded[: rows.shape[0]] = rows
            rows = padded
        return bytes(
            self._zstd.encode(np.ascontiguousarray(rows, dtype=self._dtype))
        )


class ShardWriter:
    """Writes one shard, one encoded chunk at a time."""

    def __init__(self, sink: ShardSink, slots: int) -> None:
        """Start a shard.

        Args:
            sink: Where the bytes go.
            slots: The chunk slots of the shard.
        """
        self._sink = sink
        self._slots = slots
        self._lengths: list[int] = []

    def append(self, chunk: bytes) -> None:
        """Add the next chunk in chunk order.

        Raises:
            ValueError: If every slot already holds a chunk.
        """
        if len(self._lengths) == self._slots:
            raise ValueError(f"a shard holds at most {self._slots} chunks")
        self._sink.append(chunk)
        self._lengths.append(len(chunk))

    def close(self) -> None:
        """Write the index and its checksum, then close the sink.

        A slot after the last appended chunk is marked missing.
        """
        index = np.full((self._slots, 2), _MISSING, dtype="<u8")
        offset = 0
        for slot, length in enumerate(self._lengths):
            index[slot] = (offset, length)
            offset += length
        index_bytes = index.tobytes()
        checksum = google_crc32c.value(index_bytes)
        self._sink.append(index_bytes + struct.pack("<I", checksum))
        self._sink.close()


class ArrayWriter:
    """Writes the shards of one array from blocks of rows of any length."""

    def __init__(
        self, layout: ShardLayout, executor: Executor | None = None
    ) -> None:
        """Start an array.

        Args:
            layout: The layout of the array.
            executor: Encodes the chunks of one push in groups of at least
                ENCODE_GROUP_BYTES. The caller of push may run on a worker of
                the executor while another worker is idle. Chunks encode in
                the calling thread when None.
        """
        self._layout = layout
        self._encoder = ChunkEncoder(layout)
        self._executor = executor
        self._partial = np.empty(
            (layout.chunk_rows, *layout.trailing_shape), dtype=layout.dtype
        )
        self._partial_rows = 0
        self._rows = 0
        self._chunks = 0
        self._shard: ShardWriter | None = None
        chunk_bytes = max(1, self._partial.nbytes)
        self._group_chunks = max(1, -(-ENCODE_GROUP_BYTES // chunk_bytes))

    def push(self, rows: npt.NDArray[np.generic]) -> None:
        """Add the next rows of the array.

        Every complete chunk is encoded and appended to its shard before the
        call returns. At most one partial chunk stays in memory.

        Args:
            rows: Rows with the dtype and trailing shape of the layout.

        Raises:
            ValueError: If rows has another dtype or trailing shape, or the
                array would pass its length.
        """
        layout = self._layout
        if rows.dtype != layout.dtype:
            raise ValueError(
                f"rows must have dtype {layout.dtype}, got {rows.dtype}"
            )
        if rows.shape[1:] != layout.trailing_shape:
            raise ValueError(
                f"a row has shape {layout.trailing_shape}, got {rows.shape[1:]}"
            )
        if self._rows + rows.shape[0] > layout.length:
            raise ValueError(
                f"the array has {layout.length} rows, got "
                f"{self._rows + rows.shape[0]}"
            )
        self._rows += rows.shape[0]
        chunk_rows = layout.chunk_rows
        complete: list[npt.NDArray[np.generic]] = []
        start = 0
        if self._partial_rows:
            take = min(chunk_rows - self._partial_rows, rows.shape[0])
            self._partial[self._partial_rows : self._partial_rows + take] = (
                rows[:take]
            )
            self._partial_rows += take
            start = take
            if self._partial_rows < chunk_rows:
                return
            complete.append(self._partial.copy())
            self._partial_rows = 0
        while rows.shape[0] - start >= chunk_rows:
            complete.append(rows[start : start + chunk_rows])
            start += chunk_rows
        tail = rows.shape[0] - start
        self._partial[:tail] = rows[start:]
        self._partial_rows = tail
        self._append(self._encode(complete))

    def finish(self) -> None:
        """Pad and write the last chunk, then close the last shard.

        Raises:
            ValueError: If the rows pushed are fewer than the layout length.
        """
        if self._rows != self._layout.length:
            raise ValueError(
                f"the array has {self._layout.length} rows, got {self._rows}"
            )
        if self._partial_rows:
            last = self._encoder.encode(self._partial[: self._partial_rows])
            self._partial_rows = 0
            self._append([last])
        if self._shard is not None:
            self._shard.close()
            self._shard = None

    def _encode(self, chunks: Sequence[npt.NDArray[np.generic]]) -> list[bytes]:
        """Encode whole chunks in order, in groups on the executor when set."""
        if self._executor is None or len(chunks) <= self._group_chunks:
            return [self._encoder.encode(chunk) for chunk in chunks]
        size = self._group_chunks
        groups = [chunks[i : i + size] for i in range(0, len(chunks), size)]
        encoded: list[bytes] = []
        for group in self._executor.map(self._encode_group, groups):
            encoded.extend(group)
        return encoded

    def _encode_group(
        self, chunks: Sequence[npt.NDArray[np.generic]]
    ) -> list[bytes]:
        """Encode one group of whole chunks."""
        return [self._encoder.encode(chunk) for chunk in chunks]

    def _append(self, encoded: Sequence[bytes]) -> None:
        """Add encoded chunks to their shards, opening and closing shards."""
        layout = self._layout
        for chunk in encoded:
            if self._shard is None:
                self._shard = ShardWriter(
                    layout.open_sink(self._chunks // layout.chunks_per_shard),
                    layout.chunks_per_shard,
                )
            self._shard.append(chunk)
            self._chunks += 1
            if self._chunks % layout.chunks_per_shard == 0:
                self._shard.close()
                self._shard = None
