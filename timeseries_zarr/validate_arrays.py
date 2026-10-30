"""Check the zarr.json of one array and the presence of its chunks.

Each check returns findings instead of raising, so a validator reports every
violation of an array. The requirements are in sections 7.1 through 7.3 of
docs/spec.md.
"""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any, TypeGuard

import numpy as np
from zarr.abc.store import SuffixByteRequest
from zarr.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import StorePath

from timeseries_zarr.types import Finding

_ENDIAN = "little"
_EMPTY_ENTRY = 2**64 - 1
_INDEX_ENTRY_BYTES = 16
_INDEX_CHECKSUM_BYTES = 4
_CHUNK_LIMIT = 2**24
_FLOAT_FILL = "NaN"
_INTEGER_FILL = 0


@dataclass(frozen=True, slots=True)
class ArrayLayout:
    """The shape and the chunking that an array declares.

    Attributes:
        shape: The shape of the array.
        chunk_shape: The shape of a chunk object. For a sharded array, a
            shard.
        inner_shape: The shape of an inner chunk of a sharded array, or None
            when the array is not sharded.
    """

    shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    inner_shape: tuple[int, ...] | None


def is_integer(value: object) -> TypeGuard[int]:
    """Return whether value is a JSON integer."""
    return isinstance(value, int) and not isinstance(value, bool)


def _int_tuple(value: object, minimum: int) -> tuple[int, ...] | None:
    """Return value as a tuple of integers at least minimum, else None."""
    if not isinstance(value, list) or not all(
        is_integer(item) and item >= minimum for item in value
    ):
        return None
    return tuple(value)


def _codec_parts(codec: object) -> tuple[str | None, Mapping[str, Any]]:
    """Return the name and the configuration of one codec entry."""
    if isinstance(codec, str):
        return codec, {}
    if not isinstance(codec, dict):
        return None, {}
    name = codec.get("name")
    config = codec.get("configuration", {})
    return (
        name if isinstance(name, str) else None,
        config if isinstance(config, dict) else {},
    )


def _names(codecs: object) -> list[str | None] | None:
    """Return the names of a list of codecs, or None when codecs is no list."""
    if not isinstance(codecs, list):
        return None
    return [_codec_parts(codec)[0] for codec in codecs]


def _bytes_problem(config: Mapping[str, Any], itemsize: int) -> str | None:
    """Return what is wrong with a bytes codec of a dtype, or None."""
    endian = config.get("endian")
    if endian == _ENDIAN or (endian is None and itemsize == 1):
        return None
    if endian is None:
        return f'the bytes codec must declare "endian": "{_ENDIAN}"'
    return f'the bytes codec has endian {endian!r}, expected "{_ENDIAN}"'


def _chain_problem(codecs: object, itemsize: int) -> str | None:
    """Return what is wrong with a chain of bytes and zstd, or None.

    A valid chain is bytes, optionally followed by zstd. The endian
    configuration of bytes may be absent for a one-byte dtype.
    """
    if not isinstance(codecs, list) or not codecs:
        return "the codec chain must be a non-empty list"
    names = _names(codecs)
    if names not in (["bytes"], ["bytes", "zstd"]):
        return (
            f"the codec chain must be bytes, then optionally zstd, got {names}"
        )
    return _bytes_problem(_codec_parts(codecs[0])[1], itemsize)


def _index_problem(config: Mapping[str, Any]) -> str | None:
    """Return what is wrong with the index of a sharding codec, or None."""
    index_codecs = config.get("index_codecs")
    if not isinstance(index_codecs, list) or _names(index_codecs) != [
        "bytes",
        "crc32c",
    ]:
        return (
            "the shard index codecs must be bytes then crc32c, got "
            f"{_names(index_codecs)}"
        )
    endian = _codec_parts(index_codecs[0])[1].get("endian")
    if endian != _ENDIAN:
        return f'the shard index bytes codec must declare "endian": "{_ENDIAN}"'
    location = config.get("index_location", "end")
    if location != "end":
        return f'the shard index location must be "end", got {location!r}'
    return None


def _inner_shape(
    config: Mapping[str, Any], outer: tuple[int, ...]
) -> tuple[int, ...] | None:
    """Return the inner chunk shape of a sharding codec, or None if invalid."""
    inner = _int_tuple(config.get("chunk_shape"), 1)
    if (
        inner is None
        or len(inner) != len(outer)
        or any(shard % chunk for shard, chunk in zip(outer, inner, strict=True))
    ):
        return None
    return inner


def _check_codecs(
    path: str,
    node: Mapping[str, Any],
    itemsize: int,
    outer: tuple[int, ...] | None,
) -> tuple[tuple[int, ...] | None, list[Finding]]:
    """Check the codec chain of an array against section 7.2.

    Returns:
        The inner chunk shape when the array is sharded and valid, and the
        findings.
    """
    codecs = node.get("codecs")
    problem: str | None
    inner: tuple[int, ...] | None = None
    if isinstance(codecs, list) and _names(codecs) == ["sharding_indexed"]:
        config = _codec_parts(codecs[0])[1]
        problem = _chain_problem(config.get("codecs"), itemsize) or (
            _index_problem(config)
        )
        if outer is not None:
            inner = _inner_shape(config, outer)
            if inner is None and problem is None:
                problem = (
                    "the inner chunk shape must have one positive integer "
                    f"per axis that divides the shard shape {list(outer)}"
                )
    else:
        problem = _chain_problem(codecs, itemsize)
    if problem is None:
        return inner, []
    return None, [Finding(path, "7.2", problem)]


def _check_grid(
    path: str, node: Mapping[str, Any], rank: int
) -> tuple[tuple[int, ...] | None, list[Finding]]:
    """Check the chunk grid and the chunk key encoding (section 7.1)."""
    findings: list[Finding] = []
    grid = node.get("chunk_grid")
    shape = None
    if isinstance(grid, dict) and grid.get("name") == "regular":
        configuration = grid.get("configuration")
        if isinstance(configuration, dict):
            shape = _int_tuple(configuration.get("chunk_shape"), 1)
    if shape is None or len(shape) != rank:
        findings.append(
            Finding(
                path,
                "7.1",
                "chunk_grid must be regular with one positive chunk length "
                f"for each of the {rank} axes",
            )
        )
        shape = None
    encoding = node.get("chunk_key_encoding")
    separator = None
    if isinstance(encoding, dict) and encoding.get("name") == "default":
        configuration = encoding.get("configuration", {})
        if isinstance(configuration, dict):
            separator = configuration.get("separator", "/")
    if separator != "/":
        findings.append(
            Finding(
                path,
                "7.1",
                'chunk_key_encoding must be "default" with the separator "/"',
            )
        )
    return shape, findings


def _check_fill(
    path: str, node: Mapping[str, Any], dtype: str
) -> list[Finding]:
    """Check the fill value of an array against section 7.3."""
    fill = node.get("fill_value")
    if dtype.startswith("float"):
        valid = isinstance(fill, str) and fill == _FLOAT_FILL
        expected = f'"{_FLOAT_FILL}"'
    else:
        valid = is_integer(fill) and fill == _INTEGER_FILL
        expected = str(_INTEGER_FILL)
    if valid:
        return []
    return [
        Finding(
            path,
            "7.3",
            f"fill_value is {fill!r} but a {dtype} array declares {expected}",
        )
    ]


def _shape_text(shape: Sequence[int | None]) -> str:
    """Return a shape with the free axes written as an asterisk."""
    parts = ["*" if size is None else str(size) for size in shape]
    return f"({', '.join(parts)})"


def _check_shape(
    path: str,
    node: Mapping[str, Any],
    expected: Sequence[int | None],
    section: str,
) -> tuple[tuple[int, ...] | None, list[Finding]]:
    """Check that the shape matches expected, where a None axis is free."""
    shape = _int_tuple(node.get("shape"), 0)
    if shape is not None and len(shape) == len(expected):
        bad = [
            axis
            for axis, size in enumerate(expected)
            if size is not None and shape[axis] != size
        ]
        if not bad:
            return shape, []
    return None, [
        Finding(
            path,
            section,
            f"shape is {node.get('shape')!r} but must be {_shape_text(expected)}",
        )
    ]


def check_array(
    path: str,
    node: Mapping[str, Any],
    dtype: str,
    itemsize: int,
    expected_shape: Sequence[int | None],
    section: str,
) -> tuple[ArrayLayout | None, list[Finding]]:
    """Check the zarr.json of an array against sections 7.1 through 7.3.

    Args:
        path: The path of the array, for the findings.
        node: The parsed zarr.json of the array.
        dtype: The Zarr name of the dtype the array must hold.
        itemsize: The width of dtype in bytes.
        expected_shape: The shape the array must have, with None for an axis
            of any length.
        section: The section that defines the array and its shape.

    Returns:
        The layout of the array, or None when the shape or the chunking
        is invalid, and the findings.
    """
    if node.get("node_type") != "array":
        return None, [Finding(path, section, "the node must be an array")]
    findings: list[Finding] = []
    if node.get("data_type") != dtype:
        findings.append(
            Finding(
                path,
                section,
                f"data_type is {node.get('data_type')!r} but must be {dtype}",
            )
        )
    shape, shape_findings = _check_shape(path, node, expected_shape, section)
    chunk_shape, grid_findings = _check_grid(path, node, len(expected_shape))
    inner, codec_findings = _check_codecs(path, node, itemsize, chunk_shape)
    findings += shape_findings + _check_fill(path, node, dtype)
    findings += grid_findings + codec_findings
    if shape is None or chunk_shape is None or codec_findings:
        return None, findings
    return ArrayLayout(shape, chunk_shape, inner), findings


def _chunk_key(index: Sequence[int]) -> str:
    """Return the key of the chunk at index, below the array."""
    return "c/" + "/".join(str(part) for part in index)


def _read_shard_index(
    store_path: StorePath, key: str, inner_counts: tuple[int, ...]
) -> tuple[np.ndarray[Any, np.dtype[np.uint64]] | None, str | None]:
    """Return the entries of the shard index at key, or the reason it is absent.

    Returns:
        An array of shape inner_counts + (2,) holding the offset and the
        length of each inner chunk, or None, and a reason when the index
        cannot be read.
    """
    entries = prod(inner_counts)
    size = entries * _INDEX_ENTRY_BYTES + _INDEX_CHECKSUM_BYTES
    buffer = sync(
        (store_path / key).get(
            default_buffer_prototype(), SuffixByteRequest(size)
        )
    )
    if buffer is None:
        return None, None
    data = buffer.to_bytes()
    if len(data) < size:
        return None, f"the shard object {key} is too short to hold its index"
    index = np.frombuffer(
        data, dtype="<u8", count=entries * 2, offset=len(data) - size
    )
    return index.reshape((*inner_counts, 2)), None


def _missing_inner_chunks(
    store_path: StorePath,
    array_path: str,
    layout: ArrayLayout,
    shard: Sequence[int],
) -> list[str]:
    """Return the descriptions of the missing inner chunks of one shard.

    Only a chunk that intersects the extent of the array counts.
    """
    inner = layout.inner_shape
    assert inner is not None
    counts = tuple(
        outer // size
        for outer, size in zip(layout.chunk_shape, inner, strict=True)
    )
    key = f"{array_path}/{_chunk_key(shard)}"
    index, problem = _read_shard_index(store_path, key, counts)
    axes = [
        np.arange(count) + start * count
        for count, start in zip(counts, shard, strict=True)
    ]
    coordinates = np.meshgrid(*axes, indexing="ij")
    inside = np.logical_and.reduce(
        [
            grid * size < extent
            for grid, size, extent in zip(
                coordinates, inner, layout.shape, strict=True
            )
        ]
    )
    if problem is not None:
        return [problem]
    if index is None:
        return [
            f"chunk {_chunk_key(position)} (shard object absent)"
            for position in zip(
                *(grid[inside].tolist() for grid in coordinates), strict=True
            )
        ]
    empty = (index == _EMPTY_ENTRY).all(axis=-1) & inside
    return [
        f"chunk {_chunk_key(position)}"
        for position in zip(
            *(grid[empty].tolist() for grid in coordinates), strict=True
        )
    ]


def find_missing_chunks(
    store_path: StorePath, array_path: str, layout: ArrayLayout
) -> list[str]:
    """Return a description of every chunk of an array that is not stored.

    A chunk of a sharded array is an inner chunk, and it is missing when its
    index entry is empty (section 7.3). Only a chunk that intersects the
    extent of the array counts. An array with a zero-length axis has no
    chunks.

    Args:
        store_path: The root of the bundle.
        array_path: The path of the array below the root.
        layout: The layout the array declares.

    Returns:
        The descriptions, empty when every chunk is present.

    Raises:
        ValueError: If the array holds more than 2^24 chunks.
    """
    if 0 in layout.shape:
        return []
    grid = [
        range(-(-size // chunk))
        for size, chunk in zip(layout.shape, layout.chunk_shape, strict=True)
    ]
    inner_chunks = (
        1
        if layout.inner_shape is None
        else prod(
            outer // size
            for outer, size in zip(
                layout.chunk_shape, layout.inner_shape, strict=True
            )
        )
    )
    if prod(len(axis) for axis in grid) * inner_chunks > _CHUNK_LIMIT:
        raise ValueError(
            f"the array holds more than {_CHUNK_LIMIT} chunks, too many to check"
        )
    missing: list[str] = []
    for index in itertools.product(*grid):
        if layout.inner_shape is not None:
            missing += _missing_inner_chunks(
                store_path, array_path, layout, index
            )
        elif not sync(
            (store_path / f"{array_path}/{_chunk_key(index)}").exists()
        ):
            missing.append(f"chunk {_chunk_key(index)}")
    return missing
