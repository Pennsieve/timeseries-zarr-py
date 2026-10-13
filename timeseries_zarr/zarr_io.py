"""Zarr v3 calls for the writer."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import zarr
from zarr import Array, Group
from zarr.codecs.zstd import ZstdCodec
from zarr.core.common import JSON
from zarr.storage import LocalStore

# Handle types the other modules use to name what they pass and get.
type ZarrGroup = Group
type ZarrArray = Array[Any]


def open_group(path: Path) -> Group:
    """Create or open the Zarr v3 group rooted at a local filesystem path.

    Creates the directory and group metadata when path is empty or absent, and
    opens the existing group in place otherwise.
    """
    return zarr.open_group(store=LocalStore(path), mode="a", zarr_format=3)


def create_group_with_attrs(
    parent: Group, name: str, attrs: dict[str, object]
) -> Group:
    """Create a named child group under parent carrying the given attributes.

    The child sits directly beneath parent and shares its store. attrs lands
    in the child's zarr.json verbatim and unprefixed.
    """
    return parent.create_group(name, attributes=attrs)


def create_array(
    group: Group,
    name: str,
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    chunk_shape: tuple[int, ...],
    shard_shape: tuple[int, ...],
    attrs: dict[str, object],
    zstd_level: int,
    *,
    dimension_names: Sequence[str | None] | None = None,
    compress: bool = True,
) -> Array[Any]:
    """Create a v3 sharded array under group with attributes.

    The outer shard is shard_shape, split into inner chunks of chunk_shape by
    the ZEP2 sharding codec; chunk_shape must divide shard_shape along each
    axis. The inner chunks are Zstd-compressed at zstd_level, or stored as
    plain bytes when compress is false, which leaves each shard object as the
    chunk bytes followed by the shard index. attrs lands in the array's
    zarr.json verbatim. dimension_names names each axis, with None for an
    axis that has no name, and is omitted when None. Every shard written to
    the array reaches disk, including a shard whose values all equal the fill
    value, so a reader never meets a missing shard key.

    A float array declares NaN as its fill value and any other dtype 0. The
    fill is what Zarr serves for a chunk nobody wrote, and the format defines
    NaN as no data. Zarr's own default for a float array is 0.0, which would
    read an unwritten chunk as a flat zero signal. Raises ValueError if
    dimension_names does not have one entry per axis.
    """
    is_float = np.issubdtype(np.dtype(dtype), np.floating)
    return group.create_array(
        name=name,
        shape=shape,
        dtype=dtype,
        chunks=chunk_shape,
        shards=shard_shape,
        compressors=ZstdCodec(level=zstd_level) if compress else None,
        attributes=cast("dict[str, JSON]", attrs),
        fill_value=float("nan") if is_float else 0,
        dimension_names=dimension_names,
        config={"write_empty_chunks": True},
    )


def write_region(
    array: Array[Any], start: int, block: npt.NDArray[Any]
) -> None:
    """Write a contiguous block of rows into array at axis-0 offset start.

    The block's trailing dimensions must match the array's: rank-1 raw
    samples, or rank-2 (rows, 2) min/max envelopes. The region must lie within
    the array's bounds.
    """
    array[start : start + block.shape[0]] = block


def consolidate(root: Group) -> None:
    """Write the v3 inline consolidated_metadata block into the root group.

    Gathers the metadata of every descendant group and array into the root's
    own zarr.json, not a v2 .zmetadata sidecar.
    """
    zarr.consolidate_metadata(root.store)
