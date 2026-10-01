"""Zarr v3 calls for the writer."""

import json
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


def _fill_value(dtype: npt.DTypeLike) -> float | int:
    """Return the fill an unwritten chunk of this dtype reads back as."""
    return float("nan") if np.issubdtype(np.dtype(dtype), np.floating) else 0


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
    compress: bool = True,
    sharded: bool = True,
) -> Array[Any]:
    """Create a v3 sharded, Zstd-compressed array under group with attributes.

    The outer shard is shard_shape, split into inner chunks of chunk_shape by
    the ZEP2 sharding codec; chunk_shape must divide shard_shape along each
    axis. The inner chunks are Zstd-compressed at zstd_level. attrs lands in
    the array's zarr.json verbatim and unprefixed. Every shard written to the
    array reaches disk, including a shard whose values all equal the fill
    value, so a reader never meets a missing shard key.

    compress=False stores the array raw and sharded=False leaves the sharding
    codec out. Both are needed for the chunk object to be the bytes
    themselves: compression makes it a Zstd frame (zstd_level=0 included),
    and sharding appends an index footer. Together they are what lets the
    bodies chunk be opened as a text file with no tooling.

    A float array declares NaN as its fill value and an integer array 0. The
    fill is what Zarr serves for a chunk nobody wrote, and the format defines
    NaN as no data. Taking it from the dtype rather than from a caller is
    deliberate: zarr-python defaults to 0.0, so one float array created
    without it would read back as a zero-volt flatline where a gap belongs.
    """
    return group.create_array(
        name=name,
        shape=shape,
        dtype=dtype,
        chunks=chunk_shape,
        shards=shard_shape if sharded else None,
        compressors=ZstdCodec(level=zstd_level) if compress else None,
        attributes=cast("dict[str, JSON]", attrs),
        fill_value=_fill_value(dtype),
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


def write_meta_group(root_path: Path, attributes: dict[str, object]) -> None:
    """Write meta/zarr.json under root_path as a plain JSON group node.

    Written straight to the filesystem rather than through the Group API,
    and only after consolidate() has run, so the identity it holds never
    reaches the root object every cache and CDN touches. Consolidating it
    would inline a copy there and "rm -r meta/" would stop being enough to
    de-identify the bundle.
    """
    node = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": attributes,
    }
    meta_dir = root_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "zarr.json").write_text(json.dumps(node, indent=2) + "\n")


def consolidate(root: Group) -> None:
    """Write the v3 inline consolidated_metadata block into the root group.

    Gathers the metadata of every descendant group and array into the root's
    own zarr.json, not a v2 .zmetadata sidecar.
    """
    zarr.consolidate_metadata(root.store)
