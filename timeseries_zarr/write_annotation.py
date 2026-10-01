"""Write one annotation event channel: marks, and the columns describing them.

A clinician's seizure onset and a detector's forty thousand candidates are the
same shape here. Every column but the timestamps is optional, and a view reads
which ones are present to decide what it can draw, so this writer asks the
source what it has rather than assuming a profile.

Two columns are stored in the Arrow style, a flat array plus an index of n+1
offsets into it: the payloads, and the channels a mark applies to. Both hold a
count per mark that varies from zero upward, which a fixed-width column cannot
carry without sizing every row for the widest one. Sharing the boundary between
neighbours is also what makes a gap or an overlap unrepresentable rather than
merely invalid, and what lets an empty span mean "the whole recording" with no
value reserved to say so.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import annotation_channel_attrs
from timeseries_zarr.constants import (
    FLOAT32_BYTES,
    INT64_BYTES,
    UINT16_BYTES,
)
from timeseries_zarr.protocols import AnnotationChannelSource
from timeseries_zarr.sizing import chunk_and_shard
from timeseries_zarr.types import ChunkShard, WriteOpts
from timeseries_zarr.zarr_io import (
    ZarrArray,
    ZarrGroup,
    create_array,
    create_group_with_attrs,
    write_region,
)

EVENTS_KEY = "events"
DURATIONS_KEY = "durations"
LABELS_KEY = "labels"
VALUES_KEY = "values"
BODIES_KEY = "bodies"
BODY_OFFSETS_KEY = "body_offsets"
CHANNEL_REFS_KEY = "channel_refs"
CHANNEL_REF_OFFSETS_KEY = "channel_ref_offsets"

BODY_TERMINATOR = b"\n"
"""Every payload ends with a newline.

It is what makes the decompressed bodies array readable on its own: NDJSON for
JSON payloads, plain lines otherwise, so an export is one array read and a
redaction is a text tool. The offsets remain the machine path.
"""


def _flat_array(
    group: ZarrGroup,
    name: str,
    values: npt.NDArray[np.generic],
    zstd_level: int,
    *,
    single_chunk: bool = False,
    compress: bool = True,
    sharded: bool = True,
) -> ZarrArray:
    """Create a rank-1 array and write it in one go.

    single_chunk stores the array as one chunk, which is what makes the chunk
    object itself the plain payload file a reader can open with no tooling.
    """
    length = int(values.shape[0])
    grid = (max(1, length),) if single_chunk else (max(1, min(length, 2**13)),)
    array = create_array(
        group=group,
        name=name,
        shape=(length,),
        dtype=values.dtype,
        chunk_shape=grid,
        shard_shape=grid,
        attrs={},
        zstd_level=zstd_level,
        compress=compress,
        sharded=sharded,
    )
    if length:
        write_region(array, 0, values)
    return array


def _sizing(
    shape: tuple[int, ...], dtype_size: int, opts: WriteOpts
) -> ChunkShard:
    return chunk_and_shard(
        level_shape=shape,
        dtype_size=dtype_size,
        inner_len=opts.inner_len,
        target_shard_bytes=opts.target_shard_bytes,
    )


def _write_column(
    group: ZarrGroup,
    name: str,
    values: npt.NDArray[np.generic],
    dtype_size: int,
    opts: WriteOpts,
) -> ZarrArray:
    """Create a per-mark column and stream it in, shard by shard."""
    sizing = _sizing((int(values.shape[0]),), dtype_size, opts)
    array = create_array(
        group=group,
        name=name,
        shape=(int(values.shape[0]),),
        dtype=values.dtype,
        chunk_shape=sizing.chunk_shape,
        shard_shape=sizing.shard_shape,
        attrs={},
        zstd_level=opts.zstd_level,
    )
    step = sizing.shard_shape[0]
    for begin in range(0, int(values.shape[0]), step):
        write_region(array, begin, values[begin : begin + step])
    return array


def pack_bodies(
    payloads: list[bytes],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint64]]:
    """Concatenate payloads and return them with their n+1 offsets.

    Each payload is newline-terminated on the way in, so the joined bytes read
    as NDJSON or as plain lines without the offsets being consulted.
    """
    terminated = [
        body if body.endswith(BODY_TERMINATOR) else body + BODY_TERMINATOR
        for body in payloads
    ]
    offsets = np.zeros(len(terminated) + 1, dtype=np.uint64)
    np.cumsum([len(body) for body in terminated], out=offsets[1:])
    joined = b"".join(terminated)
    return np.frombuffer(joined, dtype=np.uint8).copy(), offsets


def pack_channel_refs(
    refs: list[npt.NDArray[np.uint16]],
) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint64]]:
    """Concatenate per-mark channel indices and return them with their offsets.

    A mark naming no channels contributes nothing and leaves its two offsets
    equal, which is how the format says "this applies to the whole recording"
    without reserving an index to mean it.
    """
    offsets = np.zeros(len(refs) + 1, dtype=np.uint64)
    np.cumsum([int(item.shape[0]) for item in refs], out=offsets[1:])
    joined = (
        np.concatenate([item.astype(np.uint16) for item in refs])
        if refs
        else np.empty(0, dtype=np.uint16)
    )
    return joined, offsets


def _read_all[T: np.generic](
    reader: Callable[[int, int], npt.NDArray[T]], n: int, step: int
) -> npt.NDArray[T]:
    """Read a whole column through a windowed reader."""
    blocks = [
        reader(begin, min(begin + step, n)) for begin in range(0, n, step)
    ]
    return np.concatenate(blocks) if blocks else reader(0, 0)


def write_annotation_channel(
    parent: ZarrGroup,
    index: int,
    source: AnnotationChannelSource,
    *,
    onset_us: int,
    opts: WriteOpts,
) -> None:
    """Write one annotation channel as the subgroup named str(index).

    1. Create the channel group, declaring only the optional attributes whose
       columns are present.
    2. Write the timestamps, onset-relative like every other event channel.
    3. Write each optional column the source offers.

    Bodies and channel refs are gathered whole rather than streamed. The format
    stores bodies as a single chunk a reader opens in full, so a bundle whose
    payloads do not fit in memory is one no reader could open either.
    """
    n = source.num_events()
    max_duration_us = source.max_duration_us()
    body_media_type = source.body_media_type()

    group = create_group_with_attrs(
        parent,
        str(index),
        annotation_channel_attrs(
            source.id,
            source.start_us() - onset_us,
            source.name,
            source.unit,
            body_media_type,
            max_duration_us,
            source.label_names(),
        ),
    )

    step = max(1, opts.inner_len)

    events = _read_all(source.read_events, n, step).astype(np.int64) - onset_us
    _write_column(group, EVENTS_KEY, events, INT64_BYTES, opts)

    durations: npt.NDArray[np.int64] | None = None
    if max_duration_us is not None:
        durations = _read_all(source.read_durations, n, step).astype(np.int64)
        if (durations < 0).any():
            raise ValueError("a duration cannot be negative")
        _write_column(group, DURATIONS_KEY, durations, INT64_BYTES, opts)

    if source.num_labels():
        labels = _read_all(source.read_labels, n, step).astype(np.uint16)
        _write_column(group, LABELS_KEY, labels, UINT16_BYTES, opts)

    if source.has_values():
        values = _read_all(source.read_values, n, step).astype(np.float32)
        _write_column(group, VALUES_KEY, values, FLOAT32_BYTES, opts)

    if body_media_type is not None:
        payloads: list[bytes] = []
        for begin in range(0, n, step):
            payloads.extend(source.read_bodies(begin, min(begin + step, n)))
        bodies, body_offsets = pack_bodies(payloads)
        # Uncompressed and unsharded: the chunk object is then the payload file
        # itself, and body_offsets are literal byte ranges into it.
        _flat_array(
            group,
            BODIES_KEY,
            bodies,
            0,
            single_chunk=True,
            compress=False,
            sharded=False,
        )
        _write_column(group, BODY_OFFSETS_KEY, body_offsets, INT64_BYTES, opts)

    if durations is not None:
        # Computed from what was written, never from what the source claimed. A
        # bound that is too small makes a window query miss an interval still
        # open at its left edge, and nothing errors when that happens.
        group.attrs["max_duration_us"] = (
            int(durations.max()) if durations.size else 0
        )

    if source.has_channel_refs():
        refs: list[npt.NDArray[np.uint16]] = []
        for begin in range(0, n, step):
            refs.extend(source.read_channel_refs(begin, min(begin + step, n)))
        flat, ref_offsets = pack_channel_refs(refs)
        _write_column(group, CHANNEL_REFS_KEY, flat, UINT16_BYTES, opts)
        _write_column(
            group, CHANNEL_REF_OFFSETS_KEY, ref_offsets, INT64_BYTES, opts
        )


__all__ = [
    "BODIES_KEY",
    "BODY_OFFSETS_KEY",
    "CHANNEL_REFS_KEY",
    "CHANNEL_REF_OFFSETS_KEY",
    "DURATIONS_KEY",
    "EVENTS_KEY",
    "LABELS_KEY",
    "VALUES_KEY",
    "pack_bodies",
    "pack_channel_refs",
    "write_annotation_channel",
]
