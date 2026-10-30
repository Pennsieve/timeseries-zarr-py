"""Zarr v3 calls for the writer and the reader."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit
from urllib.request import url2pathname

import numpy as np
import numpy.typing as npt
import zarr
from zarr import Array, Group
from zarr.abc.store import Store
from zarr.buffer import default_buffer_prototype
from zarr.codecs import BytesCodec, Crc32cCodec, ShardingCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.array import AsyncArray
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding
from zarr.core.common import JSON
from zarr.core.dtype import parse_data_type
from zarr.core.group import GroupMetadata
from zarr.core.metadata import ArrayV3Metadata
from zarr.core.sync import sync
from zarr.storage import FsspecStore, LocalStore, StorePath, WrapperStore

from timeseries_zarr.extras import missing_extra_message
from timeseries_zarr.shards import (
    BufferShardSink,
    FileShardSink,
    ShardLayout,
    ShardSink,
)

# Handle types the other modules use to name what they pass and get.
type ZarrGroup = Group
type ZarrArray = Array[Any]
type BundleLocation = str | Path | Store | StorePath
type NodeMetadata = GroupMetadata | ArrayV3Metadata

_LOCAL_HOSTS = ("", "localhost")
_LOCAL_PROTOCOLS = frozenset({"file", "local"})
_FSSPEC = "fsspec"
_REMOTE_EXTRA = "remote"


def open_group(path: Path, attrs: dict[str, object] | None = None) -> Group:
    """Create or open the Zarr v3 group rooted at a local filesystem path.

    Creates the directory and group metadata when path is empty or absent, and
    opens the existing group in place otherwise.

    Args:
        path: The local directory of the group.
        attrs: The attributes of the group, stored verbatim when the group is
            created. Ignored when the group already exists, so reopening a
            bundle never rewrites its root attributes.
    """
    return zarr.open_group(
        store=LocalStore(path),
        mode="a",
        zarr_format=3,
        attributes=cast("dict[str, JSON] | None", attrs),
    )


def create_group_with_attrs(
    parent: Group, name: str, attrs: dict[str, object]
) -> Group:
    """Create a named child group under parent carrying the given attributes.

    The child sits directly beneath parent and shares its store.

    Args:
        parent: The group to create the child in.
        name: The name of the child group.
        attrs: The attributes of the child, stored verbatim and unprefixed in
            its zarr.json.
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
) -> Array[Any]:
    """Create a v3 sharded array under group with attributes.

    The ZEP2 sharding codec splits each outer shard into inner chunks. Every
    shard written to the array reaches disk, including a shard whose values
    all equal the fill value, so a reader never meets a missing shard key.

    A float array declares NaN as its fill value and any other dtype 0. The
    fill is what Zarr serves for a chunk nobody wrote, and the format defines
    NaN as no data. Zarr's own default for a float array is 0.0, which would
    read an unwritten chunk as a flat zero signal.

    Args:
        group: The group to create the array in.
        name: The name of the array.
        shape: The shape of the array.
        dtype: The element type of the array.
        chunk_shape: The shape of an inner chunk. It must divide shard_shape
            along each axis.
        shard_shape: The shape of an outer shard.
        attrs: The attributes of the array, stored verbatim in its zarr.json.
        zstd_level: The Zstd level of the inner chunks.
        dimension_names: One name for each axis, with None for an axis that
            has no name. Omitted when None.

    Raises:
        ValueError: If dimension_names does not have one entry per axis.
    """
    is_float = np.issubdtype(np.dtype(dtype), np.floating)
    return group.create_array(
        name=name,
        shape=shape,
        dtype=dtype,
        chunks=chunk_shape,
        shards=shard_shape,
        compressors=ZstdCodec(level=zstd_level),
        attributes=cast("dict[str, JSON]", attrs),
        fill_value=float("nan") if is_float else 0,
        dimension_names=dimension_names,
        config={"write_empty_chunks": True},
    )


def group_metadata(attrs: dict[str, object]) -> GroupMetadata:
    """Return the metadata of a v3 group carrying the given attributes.

    The metadata stores to the same zarr.json as create_group_with_attrs.

    Args:
        attrs: The attributes of the group, stored verbatim.
    """
    return GroupMetadata(
        attributes=cast("dict[str, JSON]", attrs), zarr_format=3
    )


def array_metadata(
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    chunk_shape: tuple[int, ...],
    shard_shape: tuple[int, ...],
    zstd_level: int,
    attrs: dict[str, object],
    *,
    dimension_names: Sequence[str | None] | None = None,
) -> ArrayV3Metadata:
    """Return the metadata of a sharded v3 array with Zstd-compressed chunks.

    The metadata stores to the same zarr.json as create_array. The fill value
    follows the rule of create_array: NaN for a float dtype and 0 for any other.

    Args:
        shape: The shape of the array.
        dtype: The element type of the array.
        chunk_shape: The shape of an inner chunk. It must divide shard_shape
            along each axis.
        shard_shape: The shape of an outer shard.
        zstd_level: The Zstd level of the inner chunks.
        attrs: The attributes of the array, stored verbatim.
        dimension_names: One name for each axis, with None for an axis that
            has no name. Omitted when None.

    Raises:
        ValueError: If dimension_names does not have one entry per axis, or if
            chunk_shape does not divide shard_shape.
    """
    numpy_dtype = np.dtype(dtype)
    return ArrayV3Metadata(
        shape=shape,
        data_type=parse_data_type(numpy_dtype, zarr_format=3),
        chunk_grid={
            "name": "regular",
            "configuration": {"chunk_shape": shard_shape},
        },
        chunk_key_encoding=DefaultChunkKeyEncoding(separator="/"),
        fill_value=(
            float("nan") if np.issubdtype(numpy_dtype, np.floating) else 0
        ),
        codecs=[
            ShardingCodec(
                chunk_shape=chunk_shape,
                codecs=[BytesCodec(), ZstdCodec(level=zstd_level)],
            )
        ],
        attributes=cast("dict[str, JSON]", attrs),
        dimension_names=dimension_names,
    )


def create_nodes(
    root: Group, nodes: Mapping[str, NodeMetadata]
) -> dict[str, ZarrGroup | ZarrArray]:
    """Create every node of nodes below root in one concurrent call.

    A node that names a missing parent group gets an empty one. A parent that
    exists is left as it is. An array handle writes every chunk, including one
    whose values all equal the fill value, as create_array does.

    Args:
        root: The group that the keys of nodes are relative to.
        nodes: The metadata of each node by path below root, such as 0/1/minmax.

    Returns:
        The handle of each node by the same path.

    Raises:
        zarr.errors.ContainsGroupError: If a group of nodes already exists.
        zarr.errors.ContainsArrayError: If an array of nodes already exists.
    """
    prefix = root.path

    def full_path(path: str) -> str:
        return f"{prefix}/{path}" if prefix else path

    created = dict(
        zarr.create_hierarchy(
            store=root.store,
            nodes={full_path(path): nodes[path] for path in nodes},
        )
    )
    handles: dict[str, ZarrGroup | ZarrArray] = {}
    for path, metadata in nodes.items():
        node = created[full_path(path)]
        if isinstance(metadata, ArrayV3Metadata):
            node = Array(
                AsyncArray(
                    metadata,
                    node.store_path,
                    config={"write_empty_chunks": True},
                )
            )
        handles[path] = node
    return handles


def shard_layout(array: Array[Any]) -> ShardLayout:
    """Return the layout that the shard writer needs to write array.

    The shard writer encodes one codec chain: the sharding codec over the bytes
    codec (little-endian) then the zstd codec with no frame checksum, with the
    bytes and crc32c codecs for the index and the index at the end of the
    shard. Axes after the first are not chunked. The sink of a shard writes the
    file under the root of a LocalStore, and calls set on any other store.

    Args:
        array: A v3 array in a writable store.

    Raises:
        ValueError: If the array is not a v3 array in a writable store, if its
            codec chain is not the chain above, or if an axis after the first
            is split into several chunks or shards.
    """
    metadata = array.metadata
    if not isinstance(metadata, ArrayV3Metadata):
        raise ValueError(
            f"{array.path} must be a Zarr v3 array, got format "
            f"{metadata.zarr_format}"
        )
    dtype = np.dtype(array.dtype)
    zstd_level = _zstd_level(array.path, metadata, dtype)
    chunks = array.chunks
    shards = array.shards
    if shards is None or not (chunks[1:] == shards[1:] == array.shape[1:]):
        raise ValueError(
            f"{array.path} must hold whole rows in every chunk and shard, "
            f"got shape {array.shape}, chunks {chunks}, shards {shards}"
        )
    store = array.store_path.store
    if store.read_only:
        raise ValueError(f"the store of {array.path} is read-only")
    fill_value = np.asarray(metadata.fill_value).item()
    trailing = (0,) * (len(array.shape) - 1)
    root = Path(store.root) if isinstance(store, LocalStore) else None

    def open_sink(index: int) -> ShardSink:
        key = metadata.encode_chunk_key((index, *trailing))
        if root is not None:
            return FileShardSink(root / array.store_path.path / key)
        target = array.store_path / key

        def save(data: bytes) -> None:
            sync(target.set(default_buffer_prototype().buffer.from_bytes(data)))

        return BufferShardSink(save)

    return ShardLayout(
        dtype=dtype,
        chunk_rows=chunks[0],
        chunks_per_shard=shards[0] // chunks[0],
        trailing_shape=array.shape[1:],
        length=array.shape[0],
        zstd_level=zstd_level,
        fill_value=fill_value,
        open_sink=open_sink,
    )


def _zstd_level(
    path: str, metadata: ArrayV3Metadata, dtype: np.dtype[Any]
) -> int:
    """Return the Zstd level of metadata when it holds the chain shard_layout encodes.

    Raises:
        ValueError: If the codec chain is anything else.
    """
    found = f"{path} has codecs {metadata.codecs}"
    match metadata.codecs:
        case (ShardingCodec() as sharding,):
            pass
        case _:
            raise ValueError(f"{found}; expected one sharding_indexed codec")
    # A one-byte dtype has no byte order, and zarr stores None for it.
    endians = ("little", None) if dtype.itemsize == 1 else ("little",)
    match sharding.codecs:
        case (
            BytesCodec(endian=endian),
            ZstdCodec(level=level, checksum=False),
        ) if endian in endians:
            pass
        case _:
            raise ValueError(
                f"{found}; the inner codecs must be bytes (little-endian) "
                "then zstd without a checksum"
            )
    match sharding.index_codecs:
        case (BytesCodec(endian="little"), Crc32cCodec()):
            pass
        case _:
            raise ValueError(
                f"{found}; the index codecs must be bytes (little-endian) "
                "then crc32c"
            )
    if sharding.index_location != "end":
        raise ValueError(
            f"{found}; the index location must be end, got "
            f"{sharding.index_location}"
        )
    return int(level)


def write_region(
    array: Array[Any], start: int, block: npt.NDArray[Any]
) -> None:
    """Write a contiguous block of rows into array at axis-0 offset start.

    The block's trailing dimensions must match the array's. The region must lie
    within the array's bounds.

    Args:
        array: The array to write to.
        start: The axis-0 offset of the first row.
        block: The rows to write.
    """
    array[start : start + block.shape[0]] = block


def consolidate(root: Group) -> None:
    """Write the v3 inline consolidated_metadata block into the root group.

    Gathers the metadata of every descendant group and array into the root's
    own zarr.json, not a v2 .zmetadata sidecar.

    Args:
        root: The root group of the bundle.
    """
    zarr.consolidate_metadata(root.store)


def open_root(
    location: BundleLocation, storage_options: dict[str, Any] | None = None
) -> Group:
    """Open the root group of a bundle for reading.

    location is a local path, a file URL, an fsspec URL, a zarr Store, or a
    StorePath. A string with the file scheme is a file URL and resolves to
    the local path it names. Any other string is a URL when it holds :// or
    an fsspec :: chain, and a local path otherwise. A relative path resolves
    against the working directory at the open. storage_options goes to
    fsspec with a remote URL. The open reads only the root zarr.json.

    Args:
        location: A local path, a file URL, an fsspec URL, a zarr Store, or a
            StorePath.
        storage_options: Options for fsspec, valid only with a remote URL.

    Raises:
        ValueError: If the root zarr.json carries no consolidated metadata,
            if a file URL names a host other than localhost or holds a query
            or fragment, or if the root is an array.
        FileNotFoundError: If location has no root zarr.json.
        TypeError: If storage_options comes with a memory URL or a location
            that is not a remote URL.
        ImportError: If a remote URL needs fsspec and fsspec is not installed.
            The name attribute is fsspec.
    """
    try:
        root = zarr.open_group(
            _local_location(_read_only_store_path(location)),
            mode="r",
            zarr_format=3,
            storage_options=storage_options,
        )
    except ModuleNotFoundError as error:
        # The module reported missing can be a submodule of fsspec.
        if (error.name or "").partition(".")[0] != _FSSPEC:
            raise
        raise ImportError(
            missing_extra_message(_FSSPEC, _REMOTE_EXTRA), name=_FSSPEC
        ) from error
    if root.metadata.consolidated_metadata is None:
        # zarr's own string for a store or a StorePath can be its repr.
        shown = (
            location
            if isinstance(location, str | Path)
            else store_location(root)
        )
        raise ValueError(
            f"{shown} has no consolidated metadata; a complete bundle "
            "carries it in its root zarr.json"
        )
    return root


def _read_only_store_path(location: BundleLocation) -> BundleLocation:
    """Return a StorePath over a read-only copy of its store.

    Returns location unchanged when it is not a StorePath, when its store is
    read-only, or when its store makes no read-only copy.
    """
    # zarr opens a StorePath as given and ignores mode="r".
    if not isinstance(location, StorePath) or location.store.read_only:
        return location
    try:
        store = location.store.with_read_only(True)
    except NotImplementedError:
        return location
    return StorePath(store, location.path)


def _local_location(location: BundleLocation) -> BundleLocation:
    """Return the absolute local path location names, or location unchanged.

    A Path and a file URL name a local path. Any other string is a URL when
    it holds :// or an fsspec :: chain, and a local path otherwise. A URL, a
    Store, and a StorePath return unchanged. A relative path resolves
    against the working directory of the call.

    Raises:
        ValueError: If a file URL names a host other than localhost or holds
            a query or fragment.
    """
    if isinstance(location, Path):
        return location.absolute()
    if not isinstance(location, str):
        return location
    parts = urlsplit(location)
    if parts.scheme == "file":
        if parts.netloc.lower() not in _LOCAL_HOSTS:
            raise ValueError(
                "file URL must name localhost or no host, got "
                f"{parts.netloc!r} in {location}"
            )
        if "#" in location or "?" in location:
            raise ValueError(
                "file URL must not hold a query or fragment; percent-encode "
                f"# as %23 and ? as %3F, got {location}"
            )
        return Path(url2pathname(parts.path)).absolute()
    if "://" in location or "::" in location:
        return location
    # zarr would read a colon in the first segment as a URL scheme, and a #
    # or ? in the path as a URL fragment or query.
    return Path(location).absolute()


def descendants(
    group: Group,
) -> tuple[dict[str, Group], dict[str, Array[Any]]]:
    """Return the groups and the arrays below group.

    Each dict maps the path relative to group, such as 0/1/minmax, to its
    node. Under consolidated metadata the walk makes no store reads.

    Returns:
        A dict of groups and a dict of arrays.
    """
    groups: dict[str, Group] = {}
    arrays: dict[str, Array[Any]] = {}
    for path, node in group.members(max_depth=None):
        if isinstance(node, Group):
            groups[path] = node
        else:
            arrays[path] = node
    return groups, arrays


def read_rows(array: Array[Any], start: int, stop: int) -> npt.NDArray[Any]:
    """Return rows start through stop - 1 of array.

    A stop past the last row returns the rows that exist, which may be none.

    Args:
        array: The array to read.
        start: The first row.
        stop: The row after the last one to read.

    Raises:
        ValueError: If start is negative or stop is less than start.
    """
    if start < 0 or stop < start:
        raise ValueError(
            f"row range must have 0 <= start <= stop, got [{start}, {stop})"
        )
    return np.asarray(array[start:stop])


def read_int(array: Array[Any], index: int) -> int:
    """Return the element at index of a rank-1 integer array.

    Args:
        array: The array to read.
        index: The element to read.

    Raises:
        ValueError: If index is negative.
        IndexError: If index is past the last element.
    """
    if index < 0:
        raise ValueError(f"index must be at least 0, got {index}")
    return int(np.asarray(array[index]))


def read_store_text(group: Group, name: str) -> str | None:
    """Return the text stored at name beside the zarr.json of group.

    Args:
        group: The group whose zarr.json sits beside the object.
        name: The name of the object.

    Returns:
        The decoded text, or None when the object is absent.

    Raises:
        UnicodeDecodeError: If the bytes of the object are not UTF-8.
    """
    # FsspecStore has no get_sync, and every store has the async get.
    key = group.store_path / name
    buffer = sync(key.get(prototype=default_buffer_prototype()))
    if buffer is None:
        return None
    return buffer.to_bytes().decode("utf-8")


def store_location(group: Group) -> str:
    """Return the URL of the store and path that hold group.

    A group in a local store, or in an FsspecStore over the local
    filesystem, gives a percent-encoded absolute file URL. A group in
    another FsspecStore gives the URL of its path in the filesystem of the
    store, with the protocol of that filesystem. A group in an FsspecStore
    over a caching filesystem gives an fsspec :: chain of the caching
    protocols. The chain ends in the URL of the group in the target
    filesystem. No part of that chain is percent-encoded. A group in an
    FsspecStore over an archive or a prefixed directory filesystem gives the
    URL of the group in the outer filesystem. That URL does not always
    reopen the group. A group in a wrapper store over a local store or an
    FsspecStore gives the URL of the store that the wrapper holds. A group
    in any other store, such as a zarr MemoryStore, gives zarr's own string
    for its store path. That string need not reopen the group, and a
    MemoryStore string never does.
    """
    store = group.store
    while isinstance(store, WrapperStore):
        store = store._store
    path = group.store_path.path
    if isinstance(store, LocalStore):
        # zarr's own string leaves # and ? unescaped in a file URL.
        return Path(store.root, path).absolute().as_uri()
    if isinstance(store, FsspecStore):
        return _fsspec_location(store, path)
    return str(group.store_path)


def _fsspec_location(store: FsspecStore, path: str) -> str:
    """Return the URL of path in store, as store_location describes it."""
    # zarr's own string for an FsspecStore is the repr of the store. The
    # join matches the one zarr makes for each key.
    prefix = store.path.rstrip("/")
    joined = (f"{prefix}/{path}" if prefix else path).rstrip("/")
    # zarr wraps a synchronous filesystem, and the wrapper holds it as
    # sync_fs.
    filesystem = getattr(store.fs, "sync_fs", store.fs)
    chain: list[str] = []
    # A caching filesystem keeps both its target filesystem and the protocol
    # of that target. A directory filesystem keeps no target_protocol.
    while hasattr(filesystem, "target_protocol") and hasattr(filesystem, "fs"):
        # fsspec before 2025.9.0 gives the protocol of the target for the
        # protocol of a caching filesystem instance.
        chain.append(_protocols(type(filesystem).protocol)[0])
        filesystem = getattr(filesystem.fs, "sync_fs", filesystem.fs)
    if chain:
        # fsspec does not percent-decode a URL inside a chain.
        return "::".join([*chain, str(filesystem.unstrip_protocol(joined))])
    if set(_protocols(filesystem.protocol)) & _LOCAL_PROTOCOLS:
        # fsspec leaves # and ? unescaped in a file URL.
        return Path(joined).absolute().as_uri()
    return str(filesystem.unstrip_protocol(joined))


def _protocols(protocol: str | Sequence[str]) -> tuple[str, ...]:
    """Return the protocol of an fsspec filesystem as a tuple.

    The main protocol comes first.
    """
    return (protocol,) if isinstance(protocol, str) else tuple(protocol)


def close_store(group: Group) -> None:
    """Close the store that holds group. A second call has no effect."""
    group.store.close()
