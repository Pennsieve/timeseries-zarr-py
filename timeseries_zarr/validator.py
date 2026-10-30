"""Check a bundle against the validation rules of docs/spec.md.

validate_bundle implements section 8.3 (structural validation) and, with
full set, section 8.4 (full validation). It reports every violation it
finds. The module reads zarr.json files and recording.json as plain JSON.
"""

import datetime as dt
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn, TypeGuard

import numpy as np
import numpy.typing as npt
import zarr
from zarr.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import StorePath

from timeseries_zarr.attrs import label_count
from timeseries_zarr.constants import (
    CHANNEL_KINDS,
    CHANNELS_KEY,
    COUNT_KEY,
    DURATIONS_KEY,
    ENVELOPE_PAIR_SIZE,
    EVENT_COUNT_LIMIT,
    EVENTS_KEY,
    FORMAT_VERSION,
    LABELS_KEY,
    LEVEL_PERIOD_TOLERANCE,
    MAJOR_VERSION,
    MEAN_KEY,
    MICROSECONDS_PER_SECOND,
    MIN_DECIMATION_FACTOR,
    MINMAX_KEY,
    RAW_KEY,
    RUNS_COLUMNS,
    RUNS_KEY,
    SAMPLES_PER_BIN_LIMIT,
    TEXT_KEY,
    TIME_LIMIT_US,
    TSZ_KEY,
    VALUES_KEY,
    VAR_KEY,
    WAVEFORM_MEAN_KEY,
    WAVEFORMS_KEY,
    ZARR_METADATA_FILE,
)
from timeseries_zarr.recording import RECORDING_FILE
from timeseries_zarr.runs import check_runs_table, first_run_end_out_of_range
from timeseries_zarr.types import Finding
from timeseries_zarr.validate_arrays import (
    ArrayLayout,
    check_array,
    find_missing_chunks,
    is_integer,
)
from timeseries_zarr.validate_full import (
    LevelArrays,
    check_continuous_levels,
    check_count_level,
    check_raw_outside_runs,
    check_waveform_mean,
)
from timeseries_zarr.windows import nearest_integer
from timeseries_zarr.zarr_io import BundleLocation, ZarrArray, open_store_path

_INTEGER_NAME = re.compile(r"0|[1-9][0-9]*")
_LEVEL_NAME = re.compile(r"[1-9][0-9]*")
_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
_RFC3339 = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})[Tt]([0-9]{2}):([0-9]{2}):([0-9]{2})"
    r"(?:\.[0-9]+)?(?:[Zz]|[+-]([0-9]{2}):([0-9]{2}))"
)
_IMPLEMENTED_MINOR = int(FORMAT_VERSION.partition(".")[2])
_ZARR_FORMAT = 3
_LAST_HOUR = 23
_LAST_MINUTE = 59
_LAST_SECOND = 60
_BITS_PER_BYTE = 8
_ASCII_LIMIT = 128
_BYTE_MASK = 0xFF
_OBJECT_KEYS = ("subject", "session", "source")

_FLOAT32 = ("float32", 4)
_INT64 = ("int64", 8)
_UINT8 = ("uint8", 1)
_UINT16 = ("uint16", 2)
_UINT32 = ("uint32", 4)

type _Node = dict[str, Any]
type _Counts = npt.NDArray[np.integer[Any]]


def _reject_constant(name: str) -> NoReturn:
    """Raise ValueError for NaN or an infinity, which JSON does not define."""
    raise ValueError(f"{name} is not a JSON value")


def _is_number(value: object) -> TypeGuard[float]:
    """Return whether value is a finite JSON number."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_time(value: object) -> TypeGuard[int]:
    """Return whether value is a JSON integer in (-2^53, 2^53)."""
    return is_integer(value) and -TIME_LIMIT_US < value < TIME_LIMIT_US


def _in_time_range(values: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
    """Return where values lie in (-2^53, 2^53)."""
    return (values > -TIME_LIMIT_US) & (values < TIME_LIMIT_US)


def _is_datetime(value: object) -> bool:
    """Return whether value is an RFC 3339 datetime with a UTC offset."""
    if not isinstance(value, str):
        return False
    match = _RFC3339.fullmatch(value)
    if match is None:
        return False
    year, month, day, hour, minute, second = (
        int(part) for part in match.groups()[:6]
    )
    offset_hour, offset_minute = (int(part or 0) for part in match.groups()[6:])
    try:
        dt.date(year, month, day)
    except ValueError:
        return False
    return (
        hour <= _LAST_HOUR
        and minute <= _LAST_MINUTE
        and second <= _LAST_SECOND
        and offset_hour <= _LAST_HOUR
        and offset_minute <= _LAST_MINUTE
    )


class _Validation:
    """One validation of one bundle."""

    def __init__(self, root: StorePath, *, full: bool) -> None:
        self.root = root
        self.full = full
        self.findings: list[Finding] = []
        self.nodes: dict[str, _Node] = {}
        self.broken: set[str] = set()
        self.children: dict[str, dict[str, _Node]] = {}
        self.layouts: dict[str, ArrayLayout] = {}
        self.data: dict[str, Any] = {}
        self.handles: dict[str, ZarrArray | None] = {}

    def add(self, path: str, section: str, message: str) -> None:
        """Record a finding."""
        self.findings.append(Finding(path, section, message))

    def extend(self, findings: Sequence[Finding]) -> None:
        """Record findings."""
        self.findings.extend(findings)

    def run(self) -> list[Finding]:
        """Validate the bundle and return the findings."""
        root = self.load_node("", required=True)
        if root is None:
            return self.findings
        self.nodes[""] = root
        if not self.check_root(root):
            return self.findings
        self.walk()
        self.check_consolidated(root)
        self.check_channels()
        self.check_recording()
        return self.findings

    def read_json(self, key: str) -> tuple[object, str | None]:
        """Return the JSON at key, or None, and the reason it is not JSON."""
        buffer = sync((self.root / key).get(default_buffer_prototype()))
        if buffer is None:
            return None, None
        try:
            text = buffer.to_bytes().decode("utf-8")
            return json.loads(text, parse_constant=_reject_constant), None
        except ValueError as error:
            return None, str(error)

    def load_node(self, path: str, *, required: bool = False) -> _Node | None:
        """Return the parsed zarr.json of a node, or None.

        A node without a zarr.json is not a node. A zarr.json that is not a
        valid Zarr v3 node is a finding.
        """
        key = f"{path}/{ZARR_METADATA_FILE}" if path else ZARR_METADATA_FILE
        document, problem = self.read_json(key)
        if problem is None and document is None:
            if required:
                self.add(path, "3.1", f"the bundle has no {key}")
            return None
        self.broken.add(path)
        if problem is not None:
            self.add(path, "7.1", f"{key} is not valid JSON: {problem}")
            return None
        if not isinstance(document, dict):
            self.add(path, "7.1", f"{key} must hold a JSON object")
            return None
        if document.get("zarr_format") != _ZARR_FORMAT:
            self.add(
                path,
                "7.1",
                f"zarr_format is {document.get('zarr_format')!r} but must be 3",
            )
            return None
        if document.get("node_type") not in ("group", "array"):
            self.add(path, "7.1", 'node_type must be "group" or "array"')
            return None
        self.broken.discard(path)
        return document

    def list_names(self, path: str) -> list[str]:
        """Return the names of the entries of a group."""
        target = self.root / path if path else self.root

        async def names() -> list[str]:
            return [name async for name in target.store.list_dir(target.path)]

        return sorted(sync(names()))

    def walk(self) -> None:
        """Load every node below the root, descending through groups.

        A node whose zarr.json is invalid still has its children loaded, so
        one bad node does not hide the nodes below it from the check of the
        consolidated metadata.
        """
        pending = [""]
        while pending:
            parent = pending.pop()
            self.children[parent] = {}
            node = self.nodes.get(parent)
            if node is not None and node["node_type"] != "group":
                continue
            for name in self.list_names(parent):
                if name == ZARR_METADATA_FILE:
                    continue
                path = f"{parent}/{name}" if parent else name
                child = self.load_node(path)
                if child is not None:
                    self.nodes[path] = child
                    self.children[parent][name] = child
                if child is not None or path in self.broken:
                    pending.append(path)

    def tsz(self, path: str, *, root: bool = False) -> _Node | None:
        """Return the tsz object of a node, or None with a finding."""
        attributes = self.nodes[path].get("attributes", {})
        fields = (
            attributes.get(TSZ_KEY) if isinstance(attributes, dict) else None
        )
        if not isinstance(fields, dict):
            self.add(path, "6.1", "attributes must hold a tsz object")
            return None
        if not root and "version" in fields:
            self.add(path, "6.2", "only the root group carries version")
        return fields

    def check_root(self, root: _Node) -> bool:
        """Check the root group and return whether to continue."""
        if root["node_type"] != "group":
            self.add("", "3.1", "the root must be a group")
            return False
        fields = self.tsz("", root=True)
        if fields is None:
            return False
        version = fields.get("version")
        match = (
            _VERSION.fullmatch(version) if isinstance(version, str) else None
        )
        if match is None:
            self.add(
                "",
                "6.2",
                "tsz.version must be a string of the form major.minor, "
                f"got {version!r}",
            )
            return False
        major, minor = (int(part) for part in match.groups())
        if major != MAJOR_VERSION:
            self.add(
                "",
                "9",
                f"version {version} has major version {major}; this "
                f"validator implements major version {MAJOR_VERSION}",
            )
            return False
        if minor != _IMPLEMENTED_MINOR:
            self.add(
                "",
                "9",
                f"minor version {minor} is unsupported; this validator "
                f"implements {FORMAT_VERSION}",
            )
        return True

    def check_consolidated(self, root: _Node) -> None:
        """Check the consolidated metadata of the root (section 6.2)."""
        block = root.get("consolidated_metadata")
        if not isinstance(block, dict):
            self.add("", "6.2", "zarr.json has no consolidated_metadata")
            return
        if block.get("kind") != "inline":
            self.add("", "6.2", 'consolidated_metadata kind must be "inline"')
        if block.get("must_understand") is not False:
            self.add(
                "", "6.2", "consolidated_metadata must_understand must be false"
            )
        listed = block.get("metadata")
        if not isinstance(listed, dict):
            self.add("", "6.2", "consolidated_metadata must hold metadata")
            return
        present = (set(self.nodes) | self.broken) - {""}
        for path in sorted(present - set(listed)):
            self.add(
                path, "6.2", "the node is not in the consolidated metadata"
            )
        for path in sorted(set(listed) - present):
            self.add(
                path, "6.2", "the consolidated metadata lists a missing node"
            )
        for path in sorted(present & set(listed) & set(self.nodes)):
            if _canonical(listed[path]) != _canonical(self.nodes[path]):
                self.add(
                    path,
                    "6.2",
                    "the consolidated metadata differs from the zarr.json "
                    "of the node",
                )

    def check_channels(self) -> None:
        """Check the channel groups of the root (section 3.2)."""
        top = self.children[""]
        integer = [name for name in top if _INTEGER_NAME.fullmatch(name)]
        groups = sorted(
            (name for name in integer if top[name]["node_type"] == "group"),
            key=int,
        )
        for name in sorted(set(integer) - set(groups), key=int):
            self.add(
                name,
                "3.2",
                "an array has an integer name; only a channel group may",
            )
        if groups != [str(index) for index in range(len(groups))]:
            self.add(
                "",
                "3.2",
                f"the channel groups are named {groups}; they must be "
                "named 0 through C - 1",
            )
        for name in groups:
            self.check_channel(name, len(groups))

    def string_attribute(
        self,
        path: str,
        fields: Mapping[str, Any],
        key: str,
        *,
        required: bool,
    ) -> None:
        """Check that a tsz key holds a string, and that it exists if required."""
        if key not in fields:
            if required:
                self.add(path, "6.3", f"tsz.{key} is required")
        elif not isinstance(fields[key], str):
            self.add(path, "6.3", f"tsz.{key} must be a string")

    def check_common_attributes(
        self, path: str, fields: Mapping[str, Any]
    ) -> None:
        """Check name, id, and provenance of a channel group."""
        self.string_attribute(path, fields, "name", required=True)
        self.string_attribute(path, fields, "id", required=False)
        if "provenance" in fields and not isinstance(
            fields["provenance"], dict
        ):
            self.add(path, "6.3", "tsz.provenance must be an object")

    def check_channel(self, path: str, channel_count: int) -> None:
        """Check one channel group."""
        fields = self.tsz(path)
        if fields is None:
            return
        kind = fields.get("kind")
        if kind not in CHANNEL_KINDS:
            self.add(
                path,
                "6.3",
                f'tsz.kind must be "continuous" or "event", got {kind!r}',
            )
            return
        self.check_common_attributes(path, fields)
        if kind == "continuous":
            self.check_continuous(path, fields)
        else:
            self.check_event(path, fields, channel_count)

    # Arrays.

    def member(
        self,
        parent: str,
        name: str,
        dtype: tuple[str, int],
        shape: Sequence[int | None],
        section: str,
        *,
        required: bool = False,
        present: bool = False,
        tsz_allowed: bool = False,
    ) -> ArrayLayout | None:
        """Check the array name of a group and return its layout.

        Args:
            parent: The path of the group.
            name: The name of the array.
            dtype: The Zarr dtype name and its width in bytes.
            shape: The shape the array must have, with None for a free axis.
            section: The section that defines the array.
            required: Whether the array must exist.
            present: Whether every chunk of the array must be stored.
            tsz_allowed: Whether the array may carry a tsz object.
        """
        path = f"{parent}/{name}"
        node = self.children.get(parent, {}).get(name)
        if node is None:
            if required:
                self.add(parent, section, f"the array {name} is required")
            return None
        if node["node_type"] == "array" and not tsz_allowed:
            attributes = node.get("attributes")
            if isinstance(attributes, dict) and TSZ_KEY in attributes:
                self.add(path, "6.1", "this array must not carry a tsz object")
        layout, findings = check_array(
            path, node, dtype[0], dtype[1], shape, section
        )
        self.extend(findings)
        if layout is None or findings:
            return None
        if present:
            try:
                missing = find_missing_chunks(self.root, path, layout)
            except ValueError as error:
                self.add(path, "7.3", str(error))
                return None
            if missing:
                self.add(
                    path,
                    "7.3",
                    f"{len(missing)} chunks are missing, such as "
                    f"{missing[0]}; every chunk of this array must be stored",
                )
                return None
        self.layouts[path] = layout
        return layout

    def handle(self, path: str) -> ZarrArray | None:
        """Return an open zarr array of a valid array, or None with a finding."""
        if path not in self.layouts:
            return None
        if path not in self.handles:
            try:
                self.handles[path] = zarr.open_array(self.root / path, mode="r")
            except Exception as error:  # noqa: BLE001
                self.add(
                    path, "7.1", f"zarr-python cannot open the array: {error}"
                )
                self.handles[path] = None
        return self.handles[path]

    def read(self, path: str) -> npt.NDArray[Any] | None:
        """Return the content of a valid array, or None with a finding."""
        if path not in self.data:
            array = self.handle(path)
            try:
                self.data[path] = (
                    None if array is None else np.asarray(array[...])
                )
            except Exception as error:  # noqa: BLE001
                self.add(path, "7.1", f"the array cannot be read: {error}")
                self.data[path] = None
        return self.data[path]  # type: ignore[no-any-return]

    # Continuous channels.

    def check_continuous(self, path: str, fields: Mapping[str, Any]) -> None:
        """Check a continuous channel group (section 4)."""
        self.string_attribute(path, fields, "unit", required=True)
        rate = self.rate_attribute(path, fields)
        offset = self.offset_attribute(path, fields, "offset_us")
        raw = self.member(
            path, RAW_KEY, _FLOAT32, (None,), "4.1", required=True
        )
        samples = None if raw is None else raw.shape[0]
        if rate is not None and offset is not None and samples:
            last = offset + (samples - 1) * MICROSECONDS_PER_SECOND / rate
            if not -TIME_LIMIT_US < last < TIME_LIMIT_US:
                self.add(
                    path,
                    "2.3",
                    f"the grid time of sample {samples - 1} is {last} us, "
                    "outside (-2^53, 2^53)",
                )
        table = self.check_runs(path, rate, offset, samples)
        levels = self.check_levels(path, rate, samples)
        if self.full and raw is not None and samples is not None:
            self.full_continuous(path, table, levels)

    def rate_attribute(
        self, path: str, fields: Mapping[str, Any]
    ) -> float | None:
        """Return rate_hz when it is valid, else None with a finding."""
        rate = fields.get("rate_hz")
        if _is_number(rate) and rate > 0:
            return float(rate)
        self.add(
            path,
            "6.3",
            f"tsz.rate_hz must be a finite number above 0, got {rate!r}",
        )
        return None

    def offset_attribute(
        self,
        path: str,
        fields: Mapping[str, Any],
        key: str,
        *,
        default: int | None = 0,
    ) -> int | None:
        """Return a time attribute, the default when absent, else None."""
        if key not in fields:
            return default
        value = fields[key]
        if _is_time(value):
            return int(value)
        self.add(
            path,
            "2.3",
            f"tsz.{key} must be a JSON integer in (-2^53, 2^53), got {value!r}",
        )
        return None

    def check_runs(
        self,
        path: str,
        rate: float | None,
        offset: int | None,
        samples: int | None,
    ) -> npt.NDArray[np.int64] | None:
        """Check the runs table of a channel and return it when valid."""
        layout = self.member(
            path, RUNS_KEY, _INT64, (None, RUNS_COLUMNS), "4.1", present=True
        )
        if layout is None:
            return None
        runs_path = f"{path}/{RUNS_KEY}"
        if samples == 0:
            self.add(runs_path, "4.5", "a channel with no samples has no runs")
            return None
        table = self.read(runs_path)
        if table is None or rate is None or offset is None or samples is None:
            return None
        return (
            table
            if self.runs_valid(runs_path, table, rate, offset, samples)
            else None
        )

    def runs_valid(
        self,
        path: str,
        table: npt.NDArray[np.int64],
        rate: float,
        offset: int,
        samples: int,
    ) -> bool:
        """Check a runs table against section 4.2 and return whether it holds."""
        if table.shape[0] < 1:
            self.add(path, "4.2", "the table must hold at least 1 run")
            return False
        if not _in_time_range(table[:, 2]).all():
            self.add(
                path, "2.3", "a time in column 2 lies outside (-2^53, 2^53)"
            )
            return False
        try:
            check_runs_table(path, table, offset, rate, samples)
        except ValueError as error:
            self.add(path, "4.2", str(error).removeprefix(f"{path}: "))
            return False
        failure = first_run_end_out_of_range(table, rate)
        if failure is not None:
            self.add(
                path,
                "2.3",
                f"the last sample of run {failure[0]} is at {failure[1]} us, "
                "outside (-2^53, 2^53)",
            )
            return False
        return True

    def check_levels(
        self, path: str, rate: float | None, samples: int | None
    ) -> list[LevelArrays]:
        """Check the level groups of a continuous channel (section 4.3).

        Returns:
            The arrays of each level whose members are all valid.
        """
        names = sorted(
            (n for n in self.children[path] if _LEVEL_NAME.fullmatch(n)),
            key=int,
        )
        if names != [str(index) for index in range(1, len(names) + 1)]:
            self.add(
                path,
                "4.3",
                f"the levels are named {names}; they must be named 1 "
                "through K with no gaps",
            )
        if samples == 0 and names:
            self.add(path, "4.5", "a channel with no samples has no levels")
        arrays: list[LevelArrays] = []
        previous: int | None = None
        for name in names:
            level_path = f"{path}/{name}"
            if self.children[path][name]["node_type"] != "group":
                self.add(
                    level_path,
                    "3.3",
                    f"the child {name} has a level name, so it must be a group",
                )
                previous = None
                continue
            width = self.check_level_attributes(level_path, rate, previous)
            previous = width
            if width is None or samples is None:
                self.check_level_members(level_path, None)
                continue
            members = self.check_level_members(level_path, -(-samples // width))
            if members is not None and self.full:
                level = self.level_arrays(level_path, width, members)
                if level is not None:
                    arrays.append(level)
        return arrays

    def check_level_attributes(
        self, path: str, rate: float | None, previous: int | None
    ) -> int | None:
        """Check period_us of a level and return its width in samples."""
        fields = self.tsz(path)
        if fields is None:
            return None
        if "offset_us" in fields:
            self.add(path, "6.4", "a continuous level must not carry offset_us")
        period = fields.get("period_us")
        if not _is_number(period) or period <= 0:
            self.add(
                path,
                "6.4",
                f"tsz.period_us must be a finite number above 0, got {period!r}",
            )
            return None
        if rate is None:
            return None
        scaled = period * rate / MICROSECONDS_PER_SECOND
        if not math.isfinite(scaled):
            self.add(path, "4.3", "the bin width in samples is not finite")
            return None
        width = nearest_integer(scaled)
        exact = width * MICROSECONDS_PER_SECOND / rate
        if not MIN_DECIMATION_FACTOR <= width < SAMPLES_PER_BIN_LIMIT:
            self.add(
                path,
                "4.3",
                f"the bin width is {width} samples; it must be at least 2 "
                "and below 2^32",
            )
            return None
        if abs(period - exact) > LEVEL_PERIOD_TOLERANCE * abs(exact):
            self.add(
                path,
                "4.3",
                f"period_us {period} is not within 1e-12 relative of "
                f"{exact}, the width of {width} samples",
            )
        if previous is not None and (
            width % previous or width < MIN_DECIMATION_FACTOR * previous
        ):
            self.add(
                path,
                "4.3",
                f"the bin width {width} must be an integer multiple of the "
                f"previous level width {previous}, at least 2 times it",
            )
        return width

    def check_level_members(
        self, path: str, bins: int | None
    ) -> dict[str, ArrayLayout] | None:
        """Check the members of a continuous level.

        Returns:
            The layouts of the members, or None when one is invalid.
        """
        members = {
            MINMAX_KEY: self.member(
                path,
                MINMAX_KEY,
                _FLOAT32,
                (bins, ENVELOPE_PAIR_SIZE),
                "4.1",
                required=True,
            ),
            COUNT_KEY: self.member(
                path, COUNT_KEY, _UINT32, (bins,), "4.1", required=True
            ),
            MEAN_KEY: self.member(path, MEAN_KEY, _FLOAT32, (bins,), "4.1"),
            VAR_KEY: self.member(path, VAR_KEY, _FLOAT32, (bins,), "4.1"),
        }
        rows = {layout.shape[0] for layout in members.values() if layout}
        if len(rows) > 1:
            self.add(
                path,
                "8.3",
                "the members of the level have different numbers of rows",
            )
        valid = {name: layout for name, layout in members.items() if layout}
        broken = any(
            name in self.children[path] and name not in valid
            for name in members
        )
        required = {MINMAX_KEY, COUNT_KEY}
        if (
            broken
            or len(rows) > 1
            or bins is None
            or not required <= set(valid)
        ):
            return None
        return valid

    def level_arrays(
        self, path: str, width: int, members: Mapping[str, ArrayLayout]
    ) -> LevelArrays | None:
        """Return the open arrays of a valid level, or None if one will not open."""
        handles = {
            name: self.handle(f"{path}/{name}")
            for name in (MINMAX_KEY, COUNT_KEY, MEAN_KEY, VAR_KEY)
            if name in members
        }
        count, minmax = handles[COUNT_KEY], handles[MINMAX_KEY]
        if count is None or minmax is None:
            return None
        return LevelArrays(
            path=path,
            width=width,
            count=count,
            minmax=minmax,
            mean=handles.get(MEAN_KEY),
            var=handles.get(VAR_KEY),
        )

    def full_continuous(
        self,
        path: str,
        table: npt.NDArray[np.int64] | None,
        levels: Sequence[LevelArrays],
    ) -> None:
        """Check the raw samples and the levels of a channel (section 8.4)."""
        raw = self.handle(f"{path}/{RAW_KEY}")
        if raw is None:
            return
        try:
            if table is not None:
                self.extend(
                    check_raw_outside_runs(f"{path}/{RAW_KEY}", raw, table)
                )
            self.extend(check_continuous_levels(raw, levels))
        except Exception as error:  # noqa: BLE001
            self.add(path, "8.4", f"the arrays cannot be read: {error}")

    # Event channels.

    def check_event(
        self, path: str, fields: Mapping[str, Any], channel_count: int
    ) -> None:
        """Check an event channel group (section 5)."""
        self.string_attribute(path, fields, "unit", required=False)
        names, names_valid = self.label_names(path, fields)
        events_layout = self.member(
            path,
            EVENTS_KEY,
            _INT64,
            (None,),
            "5.1",
            required=True,
            present=True,
        )
        count = None if events_layout is None else events_layout.shape[0]
        if count is not None and count >= EVENT_COUNT_LIMIT:
            self.add(
                path,
                "5.1",
                f"the channel holds {count} events; an event channel holds "
                "fewer than 2^32",
            )

        def column(
            name: str,
            dtype: tuple[str, int],
            trailing: tuple[int | None, ...] = (),
            *,
            tsz_allowed: bool = False,
        ) -> ArrayLayout | None:
            return self.member(
                path,
                name,
                dtype,
                (count, *trailing),
                "5.1",
                present=True,
                tsz_allowed=tsz_allowed,
            )

        column(DURATIONS_KEY, _INT64)
        labels_layout = column(LABELS_KEY, _UINT16)
        column(VALUES_KEY, _FLOAT32)
        text = column(TEXT_KEY, _UINT8, (None,))
        channel_bytes = -(-channel_count // _BITS_PER_BYTE)
        channels = column(CHANNELS_KEY, _UINT8, (channel_bytes,))
        waveforms = column(WAVEFORMS_KEY, _FLOAT32, (None,), tsz_allowed=True)
        events = self.check_event_times(f"{path}/{EVENTS_KEY}", events_layout)
        self.check_durations(
            path, fields, DURATIONS_KEY in self.children[path], events
        )
        has_labels = LABELS_KEY in self.children[path]
        labels = self.read(f"{path}/{LABELS_KEY}") if labels_layout else None
        total = self.label_total(path, names, names_valid, has_labels, labels)
        self.check_waveforms(path, waveforms, total, has_labels, labels)
        if text is not None:
            self.check_text(f"{path}/{TEXT_KEY}", text)
        if channels is not None:
            self.check_channels_column(
                f"{path}/{CHANNELS_KEY}", channel_count, channel_bytes
            )
        self.check_count_levels(path, events, labels, total, count)

    def label_names(
        self, path: str, fields: Mapping[str, Any]
    ) -> tuple[list[str] | None, bool]:
        """Return label_names, and whether the attribute is valid."""
        if "label_names" not in fields:
            return None, True
        names = fields["label_names"]
        if (
            isinstance(names, list)
            and names
            and all(isinstance(name, str) for name in names)
        ):
            return names, True
        self.add(
            path, "5.3", "tsz.label_names must be a list of at least 1 string"
        )
        return None, False

    def check_event_times(
        self, path: str, layout: ArrayLayout | None
    ) -> npt.NDArray[np.int64] | None:
        """Check that the event times ascend and lie in range."""
        events = self.read(path) if layout is not None else None
        if events is None or events.size == 0:
            return events
        if (events[1:] < events[:-1]).any():
            self.add(path, "5.1", "the event times must be non-decreasing")
        if not _in_time_range(events).all():
            self.add(path, "2.3", "an event time lies outside (-2^53, 2^53)")
        return events

    def check_durations(
        self,
        path: str,
        fields: Mapping[str, Any],
        has_durations: bool,
        events: npt.NDArray[np.int64] | None,
    ) -> None:
        """Check max_duration_us and the values of durations (section 5.2)."""
        bound = self.max_duration(path, fields, has_durations=has_durations)
        durations_path = f"{path}/{DURATIONS_KEY}"
        durations = self.read(durations_path)
        if durations is None or durations.size == 0:
            return
        if durations.min() < 0:
            self.add(durations_path, "5.2", "every duration must be at least 0")
        in_range = _in_time_range(durations)
        if not in_range.all():
            self.add(
                durations_path, "2.3", "a duration lies outside (-2^53, 2^53)"
            )
        if events is not None and events.shape == durations.shape:
            both = in_range & _in_time_range(events)
            ends = events[both] + durations[both]
            if not _in_time_range(ends).all():
                self.add(
                    durations_path,
                    "2.3",
                    "the end of an interval lies outside (-2^53, 2^53)",
                )
        if bound is not None and durations.max() > bound:
            self.add(
                durations_path,
                "5.2",
                f"a duration of {int(durations.max())} us exceeds "
                f"max_duration_us {bound}",
            )

    def max_duration(
        self, path: str, fields: Mapping[str, Any], *, has_durations: bool
    ) -> int | None:
        """Return max_duration_us when it is valid, else None."""
        if "max_duration_us" not in fields:
            if has_durations:
                self.add(
                    path,
                    "5.2",
                    "tsz.max_duration_us is required when durations exists",
                )
            return None
        if not has_durations:
            self.add(
                path,
                "5.2",
                "tsz.max_duration_us must not exist without durations",
            )
        value = fields["max_duration_us"]
        if _is_time(value) and value >= 0:
            return int(value)
        self.add(
            path,
            "6.3",
            "tsz.max_duration_us must be a JSON integer in [0, 2^53), got "
            f"{value!r}",
        )
        return None

    def label_total(
        self,
        path: str,
        names: list[str] | None,
        names_valid: bool,
        has_labels: bool,
        labels: npt.NDArray[np.uint16] | None,
    ) -> int | None:
        """Return the number of labels L (section 5.3), or None if unknown."""
        if not names_valid:
            return None
        if names is None and has_labels and labels is None:
            return None
        largest = (
            int(labels.max()) if labels is not None and labels.size else None
        )
        total = label_count(None if names is None else len(names), largest)
        if largest is not None and largest >= total:
            self.add(
                f"{path}/{LABELS_KEY}",
                "5.3",
                f"a label of {largest} is not below the label count {total}",
            )
        return total

    def check_waveforms(
        self,
        path: str,
        waveforms: ArrayLayout | None,
        total: int | None,
        has_labels: bool,
        labels: npt.NDArray[np.uint16] | None,
    ) -> None:
        """Check waveforms and waveform_mean (sections 5.6 and 6.5)."""
        mean_layout = self.member(
            path,
            WAVEFORM_MEAN_KEY,
            _FLOAT32,
            (total, None),
            "5.1",
            present=True,
            tsz_allowed=True,
        )
        mean_path = f"{path}/{WAVEFORM_MEAN_KEY}"
        if mean_layout is not None and not has_labels:
            self.add(mean_path, "5.6", "waveform_mean needs the labels array")
        waveform_attributes, mean_attributes = (
            None
            if layout is None
            else self.waveform_attributes(f"{path}/{name}", layout)
            for name, layout in (
                (WAVEFORMS_KEY, waveforms),
                (WAVEFORM_MEAN_KEY, mean_layout),
            )
        )
        if waveform_attributes is None or mean_attributes is None:
            return
        if waveform_attributes != mean_attributes:
            self.add(
                mean_path,
                "5.6",
                "waveform_mean and waveforms must agree on P, period_us, "
                "and event_sample",
            )
        elif self.full and labels is not None and total is not None:
            self.full_waveform_mean(path, labels, total)

    def waveform_attributes(
        self, path: str, layout: ArrayLayout
    ) -> tuple[int, object, object] | None:
        """Check the tsz object of a waveform array.

        Returns:
            The number of samples P, period_us, and event_sample, or None.
        """
        width = layout.shape[1]
        if width < 1:
            self.add(path, "5.6", "P must be at least 1")
        fields = self.tsz(path)
        if fields is None:
            return None
        period = fields.get("period_us")
        event_sample = fields.get("event_sample", 0)
        valid = True
        if not _is_number(period) or period <= 0:
            self.add(
                path,
                "6.5",
                f"tsz.period_us must be a finite number above 0, got {period!r}",
            )
            valid = False
        if not (is_integer(event_sample) and 0 <= event_sample < width):
            self.add(
                path,
                "6.5",
                f"tsz.event_sample must be a JSON integer in [0, {width}), "
                f"got {event_sample!r}",
            )
            valid = False
        return (width, period, event_sample) if valid else None

    def full_waveform_mean(
        self, path: str, labels: npt.NDArray[np.uint16], total: int
    ) -> None:
        """Check waveform_mean against waveforms (section 8.4)."""
        mean_path = f"{path}/{WAVEFORM_MEAN_KEY}"
        waveforms = self.handle(f"{path}/{WAVEFORMS_KEY}")
        mean = self.handle(mean_path)
        if waveforms is None or mean is None:
            return
        try:
            self.extend(
                check_waveform_mean(mean_path, waveforms, mean, labels, total)
            )
        except Exception as error:  # noqa: BLE001
            self.add(mean_path, "8.4", f"the arrays cannot be read: {error}")

    def check_text(self, path: str, layout: ArrayLayout) -> None:
        """Check that every row of text is UTF-8 and then zero bytes."""
        if layout.shape[1] < 1:
            self.add(path, "5.5", "W must be at least 1")
            return
        rows = self.read(path)
        if rows is None or rows.shape[0] == 0:
            return
        filled = rows != 0
        gap = (~filled[:, :-1] & filled[:, 1:]).any(axis=1)
        wide = np.flatnonzero((rows >= _ASCII_LIMIT).any(axis=1) & ~gap)
        lengths = filled.sum(axis=1)
        invalid = gap.copy()
        for row in wide:
            try:
                bytes(rows[row, : lengths[row]]).decode("utf-8")
            except UnicodeDecodeError:
                invalid[row] = True
        if invalid.any():
            self.add(
                path,
                "5.5",
                f"row {int(np.flatnonzero(invalid)[0])} is not UTF-8 text "
                "followed by zero bytes; "
                f"{int(invalid.sum())} rows are invalid",
            )

    def check_channels_column(
        self, path: str, channel_count: int, width: int
    ) -> None:
        """Check that no bit of channels at a position of C or more is set."""
        rows = self.read(path)
        if rows is None or rows.shape[0] == 0:
            return
        used = channel_count - _BITS_PER_BYTE * (width - 1)
        spare = _BYTE_MASK & ~((1 << used) - 1)
        if (rows[:, -1] & spare).any():
            self.add(
                path,
                "5.4",
                f"a bit at a position of {channel_count} or more is set",
            )

    def check_count_levels(
        self,
        path: str,
        events: npt.NDArray[np.int64] | None,
        labels: npt.NDArray[np.uint16] | None,
        total: int | None,
        count: int | None,
    ) -> None:
        """Check the count levels of an event channel (section 5.7)."""
        names = sorted(
            (n for n in self.children[path] if _LEVEL_NAME.fullmatch(n)),
            key=int,
        )
        if names != [str(index) for index in range(1, len(names) + 1)]:
            self.add(
                path,
                "5.7",
                f"the count levels are named {names}; they must be named "
                "1 through J with no gaps",
            )
        if count == 0 and names:
            self.add(
                path, "5.7", "a channel with no events has no count levels"
            )
            return
        previous: int | None = None
        for name in names:
            level_path = f"{path}/{name}"
            if self.children[path][name]["node_type"] != "group":
                self.add(
                    level_path,
                    "3.3",
                    f"the child {name} has a level name, so it must be a group",
                )
                previous = None
                continue
            period, offset = self.count_level_attributes(
                level_path, events, previous
            )
            previous = period
            self.check_count_array(
                level_path, events, labels, total, period, offset
            )

    def count_level_attributes(
        self,
        path: str,
        events: npt.NDArray[np.int64] | None,
        previous: int | None,
    ) -> tuple[int | None, int | None]:
        """Check period_us and offset_us of a count level.

        Returns:
            The period and the offset, each None when invalid.
        """
        fields = self.tsz(path)
        if fields is None:
            return None, None
        period = fields.get("period_us")
        if not _is_time(period) or period <= 0:
            self.add(
                path,
                "5.7",
                "tsz.period_us must be a positive JSON integer below 2^53, "
                f"got {period!r}",
            )
            period = None
        offset = self.offset_attribute(path, fields, "offset_us", default=None)
        first = int(events[0]) if events is not None and events.size else None
        if offset is None and "offset_us" not in fields:
            if first is not None and first < 0:
                self.add(
                    path, "6.4", "offset_us is required when events[0] < 0"
                )
            offset = 0
        if period is None:
            return None, offset
        if previous is not None and (
            period % previous or period < MIN_DECIMATION_FACTOR * previous
        ):
            self.add(
                path,
                "5.7",
                f"period_us {period} must be an integer multiple of {previous}, "
                "the period of the level below, at least 2 times it",
            )
        if offset is not None:
            if offset % period:
                self.add(
                    path,
                    "6.4",
                    f"offset_us {offset} must be a multiple of period_us {period}",
                )
            if first is not None and offset > first:
                self.add(
                    path,
                    "6.4",
                    f"offset_us {offset} must be at most events[0], {first}",
                )
        return int(period), offset

    def check_count_array(
        self,
        path: str,
        events: npt.NDArray[np.int64] | None,
        labels: npt.NDArray[np.uint16] | None,
        total: int | None,
        period: int | None,
        offset: int | None,
    ) -> None:
        """Check the count array of a count level, and recompute it if full."""
        bins = None
        if (
            events is not None
            and events.size
            and period is not None
            and offset is not None
        ):
            bins = (int(events[-1]) - offset) // period + 1
            end = offset + bins * period
            if not -TIME_LIMIT_US < end < TIME_LIMIT_US:
                self.add(
                    path,
                    "2.3",
                    f"the end of the last bin, {end} us, lies outside "
                    "(-2^53, 2^53)",
                )
        layout = self.member(
            path, COUNT_KEY, _UINT32, (bins, total), "5.1", required=True
        )
        if (
            self.full
            and layout is not None
            and events is not None
            and events.size
            and bins is not None
            and period is not None
            and offset is not None
            and total is not None
        ):
            count_path = f"{path}/{COUNT_KEY}"
            array = self.handle(count_path)
            if array is None:
                return
            try:
                self.extend(
                    check_count_level(
                        count_path, events, labels, total, offset, period, array
                    )
                )
            except Exception as error:  # noqa: BLE001
                self.add(
                    count_path, "8.4", f"the array cannot be read: {error}"
                )

    # Recording metadata.

    def check_recording(self) -> None:
        """Check recording.json, when present (section 6.6)."""
        document, problem = self.read_json(RECORDING_FILE)
        if problem is not None:
            self.add(
                RECORDING_FILE, "6.6", f"the file is not valid JSON: {problem}"
            )
            return
        if document is None:
            return
        if not isinstance(document, dict):
            self.add(RECORDING_FILE, "6.6", "the file must hold a JSON object")
            return
        for key in _OBJECT_KEYS:
            if key in document and not isinstance(document[key], dict):
                self.add(RECORDING_FILE, "6.6", f"{key} must be an object")
        session = document.get("session")
        times: dict[str, object] = {}
        if "reference_time" in document:
            times["reference_time"] = document["reference_time"]
        if isinstance(session, dict) and "start_time" in session:
            times["session.start_time"] = session["start_time"]
        for key, value in times.items():
            if not _is_datetime(value):
                self.add(
                    RECORDING_FILE,
                    "6.6",
                    f"{key} must be an RFC 3339 datetime with a UTC offset, "
                    f"got {value!r}",
                )


def _canonical(node: object) -> str:
    """Return the JSON of a node, without its nested consolidated metadata."""
    if isinstance(node, dict):
        node = {
            key: value
            for key, value in node.items()
            if key != "consolidated_metadata"
        }
    return json.dumps(node, sort_keys=True)


def validate_bundle(
    location: BundleLocation, *, full: bool = False
) -> list[Finding]:
    """Check a bundle against docs/spec.md and return every violation.

    Structural validation (section 8.3) reads every zarr.json, recording.json,
    the runs arrays, and the per-event columns other than waveforms. Full
    validation (section 8.4) also reads raw, every level member, and the
    waveform arrays, and recomputes the statistics they hold.

    Args:
        location: A local path, a file URL, an fsspec URL, a zarr Store, or a
            StorePath.
        full: Whether to run full validation.

    Returns:
        The findings in the order the validator found them. An empty list
        means the bundle conforms. A local path that does not exist gives
        one finding.

    Raises:
        ValueError: If the store cannot list its keys.
    """
    try:
        root = open_store_path(location)
    except FileNotFoundError as error:
        return [Finding("", "3.1", f"the bundle does not exist: {error}")]
    if not root.store.supports_listing:
        raise ValueError(
            "structural validation needs a store that lists its keys"
        )
    return _Validation(root, full=full).run()
