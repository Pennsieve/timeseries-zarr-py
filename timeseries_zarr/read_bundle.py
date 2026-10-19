"""Open a bundle and parse its channels and its recording.json."""

import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, Self, SupportsIndex

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import parse_channel_kind, parse_version, tsz_fields
from timeseries_zarr.constants import CHANNEL_KINDS
from timeseries_zarr.read_continuous import (
    ContinuousChannel,
    parse_continuous_channel,
)
from timeseries_zarr.read_event import EventChannel, parse_event_channel
from timeseries_zarr.read_recording import Recording, parse_recording_text
from timeseries_zarr.recording import RECORDING_FILE
from timeseries_zarr.stack import BinMember, Stack, stack_channels
from timeseries_zarr.windows import (
    canonical_number,
    integer_argument,
    window_bounds,
)
from timeseries_zarr.zarr_io import (
    BundleLocation,
    ZarrArray,
    ZarrGroup,
    close_store,
    descendants,
    open_root,
    read_store_text,
    store_location,
)

type Channel = ContinuousChannel | EventChannel

_PACKAGE_DIRECTORY = f"{Path(__file__).parent}{os.sep}"

_ABSENT = "-"
_COLUMN_SEPARATOR = "  "
_DATETIME_LIMIT_EXPONENT = 62
_DATETIME_LIMIT_US = 2**_DATETIME_LIMIT_EXPONENT
_DATETIME_BOUND = (
    "time_us must hold NaN or times below "
    f"2^{_DATETIME_LIMIT_EXPONENT} microseconds in magnitude"
)
_DESCRIBE_HEADER: tuple[str, ...] = (
    "key",
    "kind",
    "name",
    "id",
    "unit",
    "rate_hz",
    "length",
    "levels",
)


def _counted(count: int, noun: str) -> str:
    """Return count followed by noun, plural unless count is 1."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _level_span(count: int) -> str:
    """Return the level numbers 1 through count as a range."""
    if count == 0:
        return _ABSENT
    if count == 1:
        return "1"
    return f"1-{count}"


def _describe_row(channel: Channel) -> tuple[str, ...]:
    """Return the cells of the describe table for one channel."""
    if isinstance(channel, ContinuousChannel):
        return (
            str(channel.key),
            channel.kind,
            channel.name,
            _ABSENT if channel.id is None else channel.id,
            channel.unit,
            str(channel.rate_hz),
            _counted(channel.num_samples, "sample"),
            _level_span(len(channel.levels)),
        )
    return (
        str(channel.key),
        channel.kind,
        channel.name,
        _ABSENT if channel.id is None else channel.id,
        _ABSENT if channel.unit is None else channel.unit,
        _ABSENT,
        _counted(channel.num_events, "event"),
        _level_span(len(channel.count_levels)),
    )


def _table(rows: Sequence[tuple[str, ...]]) -> str:
    """Return rows as lines of cells padded to the widest cell of a column.

    The last column has no padding.
    """
    widths = [
        max(len(row[column]) for row in rows)
        for column in range(len(rows[0]) - 1)
    ]
    lines = []
    for row in rows:
        padded = [
            cell.ljust(width)
            for cell, width in zip(row[:-1], widths, strict=True)
        ]
        lines.append(_COLUMN_SEPARATOR.join([*padded, row[-1]]))
    return "\n".join(lines)


def _microsecond_offsets(
    time_us: npt.ArrayLike,
) -> npt.NDArray[np.timedelta64]:
    """Return time_us as whole microseconds, with NaT for a NaN.

    An integer keeps its value, and a fraction rounds half to even. Raises
    ValueError if a time is infinite or reaches _DATETIME_LIMIT_US in
    magnitude.
    """
    try:
        values = np.asarray(time_us)
        is_integer = np.issubdtype(values.dtype, np.integer)
        if not is_integer:
            values = np.rint(np.asarray(time_us, dtype=np.float64))
    except OverflowError as error:
        raise ValueError(
            f"{_DATETIME_BOUND}, got a number beyond the float64 range"
        ) from error
    flat = np.ravel(values)
    # The bound keeps the reference time plus an offset inside datetime64[us].
    inside = (flat > -_DATETIME_LIMIT_US) & (flat < _DATETIME_LIMIT_US)
    if not is_integer:
        inside |= np.isnan(flat)
    if not inside.all():
        outside = flat[~inside][0]
        shown = int(outside) if is_integer else float(outside)
        raise ValueError(f"{_DATETIME_BOUND}, got {shown}")
    return values.astype("timedelta64[us]")


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Bundle:
    """A bundle opened for reading.

    version is the tsz.version of the root group. recording is None when
    the bundle has no recording.json. channels maps each channel key to its
    channel in key order. It leaves out a channel group whose kind is
    unknown.
    """

    root: ZarrGroup
    version: str
    recording: Recording | None
    channels: Mapping[int, Channel]

    @property
    def continuous(self) -> dict[int, ContinuousChannel]:
        """Continuous channels by key, in key order."""
        return {
            key: channel
            for key, channel in self.channels.items()
            if isinstance(channel, ContinuousChannel)
        }

    @property
    def events(self) -> dict[int, EventChannel]:
        """Event channels by key, in key order."""
        return {
            key: channel
            for key, channel in self.channels.items()
            if isinstance(channel, EventChannel)
        }

    @property
    def location(self) -> str:
        """URL of the store and path that hold the root group.

        A store over an archive or a prefixed directory filesystem gives the
        URL in its outer filesystem, which need not reopen. A wrapper store
        over a local store or an FsspecStore gives the URL of the store it
        holds. Any other store, such as a MemoryStore, gives zarr's own
        string, which need not reopen.
        """
        return store_location(self.root)

    def channel(self, name_or_id: str) -> Channel:
        """Return the channel whose name, or else whose id, is name_or_id.

        A name match takes precedence over an id match. Raises ValueError if
        several channels share the name, or share the id when no name
        matches. Raises KeyError if no channel has that name or id.
        """
        named = [
            key
            for key, channel in self.channels.items()
            if channel.name == name_or_id
        ]
        identified = [
            key
            for key, channel in self.channels.items()
            if channel.id == name_or_id
        ]
        for noun, keys in (("name", named), ("id", identified)):
            if len(keys) == 1:
                return self.channels[keys[0]]
            if keys:
                listed = ", ".join(str(key) for key in keys)
                raise ValueError(
                    f"channels {listed} share the {noun} {name_or_id!r}; "
                    "select one of them by key in channels"
                )
        raise KeyError(
            f"no channel has the name or id {name_or_id!r}; describe() lists "
            "the name and id of each channel"
        )

    def describe(self) -> str:
        """Return a plain-text table with one line per channel after a header.

        An absent value shows as a hyphen. levels gives the range of level
        numbers, or of count level numbers on an event channel.
        """
        rows = [_DESCRIBE_HEADER]
        rows.extend(
            _describe_row(channel) for channel in self.channels.values()
        )
        return _table(rows)

    def datetimes(self, time_us: npt.ArrayLike) -> npt.NDArray[np.datetime64]:
        """Return the UTC wall-clock time of each time in time_us.

        time_us counts microseconds from the reference time, and a fraction
        rounds to the nearest microsecond. A time halfway between two
        microseconds rounds to the even one. The result has dtype
        datetime64[us], the shape of time_us, and NaT for a NaN.

        Raises ValueError if recording.json holds no reference_time, or one
        whose UTC time falls outside years 1 through 9999. Also raises
        ValueError if a time is infinite or reaches 2^62 microseconds in
        magnitude.
        """
        reference = (
            None if self.recording is None else self.recording.reference_time
        )
        if reference is None:
            raise ValueError(
                f"{RECORDING_FILE} holds no reference_time; "
                "wall-clock times are unavailable"
            )
        try:
            utc = reference.astimezone(UTC)
        except OverflowError as error:
            raise ValueError(
                f"{RECORDING_FILE}: reference_time must fall within years 1 "
                f"through 9999 in UTC, got {reference.isoformat()!r}"
            ) from error
        # numpy warns on a datetime that carries a UTC offset.
        reference64 = np.datetime64(utc.replace(tzinfo=None), "us")
        offsets = _microsecond_offsets(time_us)
        return np.asarray(reference64 + offsets, dtype="datetime64[us]")

    def stack(
        self,
        keys: Sequence[int],
        start_us: SupportsIndex | None = None,
        end_us: SupportsIndex | None = None,
        *,
        level: SupportsIndex = 0,
        member: BinMember = "mean",
    ) -> Stack:
        """Return the window of each continuous channel in keys as a Stack.

        The rows follow the order of keys. The window is [start_us, end_us),
        and a None bound leaves that side open. Level 0 reads the raw
        samples, and the member of the Stack is then None. A level of at
        least 1 reads member from the bins of that level. The rows span every
        entry that overlaps the window on any channel. A row holds NaN past
        the end of its channel, or 0 when member is count.

        Raises KeyError if no channel has a key. Raises ValueError if keys is
        empty, repeats a key, or names an event channel. Also raises
        ValueError if the channels differ in rate_hz or offset_us, or a
        channel lacks the level or member. Also raises ValueError if member
        is not a BinMember at a level of at least 1, or end_us is before
        start_us. Raises TypeError if a bound or level is not an integer.
        """
        channels = [self._continuous_channel(key) for key in keys]
        number = integer_argument(level, "level")
        start, end = window_bounds(start_us, end_us)
        return stack_channels(channels, start, end, level=number, member=member)

    def _continuous_channel(self, key: int) -> ContinuousChannel:
        """Return the continuous channel with that key.

        Raises KeyError if no channel has the key and ValueError if it names
        an event channel.
        """
        channel = self.channels.get(key)
        if isinstance(channel, ContinuousChannel):
            return channel
        listed = ", ".join(str(known) for known in self.continuous) or "none"
        if channel is None:
            raise KeyError(
                f"the bundle has no channel {key!r}; its continuous channel "
                f"keys are {listed}"
            )
        raise ValueError(
            f"channel {key} is an event channel; the continuous channel keys "
            f"are {listed}"
        )

    def close(self) -> None:
        """Close the store that holds the bundle. A second call does nothing."""
        close_store(self.root)

    def __enter__(self) -> Self:
        """Return the bundle."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the store that holds the bundle."""
        self.close()

    def __repr__(self) -> str:
        """Return the location, the version, and the channel counts."""
        return (
            f"Bundle({self.location!r}, version={self.version!r}, "
            f"continuous={len(self.continuous)}, events={len(self.events)})"
        )


def _by_top_name[T](nodes: Mapping[str, T]) -> dict[str, dict[str, T]]:
    """Group nodes below the root by the top-level group that holds them.

    The inner dict keys each node by its path below that group, such as
    1/minmax.
    """
    grouped: dict[str, dict[str, T]] = {}
    for path, node in nodes.items():
        top, separator, below = path.partition("/")
        if separator:
            grouped.setdefault(top, {})[below] = node
    return grouped


def _parse_channel(
    key: int,
    attributes: Mapping[str, object],
    groups: Mapping[str, ZarrGroup],
    arrays: Mapping[str, ZarrArray],
) -> Channel | None:
    """Return the channel with that key, or None if its kind is unknown.

    An unknown kind issues a UserWarning. Raises ValueError if the channel
    group does not parse.
    """
    where = f"channel {key}"
    kind = parse_channel_kind(attributes, where)
    if kind == "continuous":
        return parse_continuous_channel(key, attributes, groups, arrays)
    if kind == "event":
        return parse_event_channel(key, attributes, groups, arrays)
    unknown = tsz_fields(attributes, where)["kind"]
    known = " or ".join(repr(name) for name in CHANNEL_KINDS)
    # The warning names the first caller outside this package.
    warnings.warn(
        f"skipped channel {key}: tsz.kind must be {known}, got {unknown!r}",
        UserWarning,
        skip_file_prefixes=(_PACKAGE_DIRECTORY,),
    )
    return None


def parse_bundle(root: ZarrGroup) -> Bundle:
    """Return the bundle that an open root group holds.

    A channel group is a top-level group whose name is a decimal number
    without a leading zero, and that number is its key. The parser ignores
    every other top-level node. It skips a channel group of unknown kind
    with a UserWarning. It reads recording.json through the store of root.

    Raises ValueError if the root tsz.version is not a major.minor string
    with major version MAJOR_VERSION. Also raises ValueError if a channel
    group or recording.json does not parse.
    """
    version = parse_version(dict(root.attrs))
    groups, arrays = descendants(root)
    groups_below = _by_top_name(groups)
    arrays_below = _by_top_name(arrays)
    # A nested path holds a slash, so canonical_number reads no number in it.
    numbered = sorted(
        (number, name)
        for name in groups
        if (number := canonical_number(name)) is not None
    )
    channels: dict[int, Channel] = {}
    for key, name in numbered:
        channel = _parse_channel(
            key,
            dict(groups[name].attrs),
            groups_below.get(name, {}),
            arrays_below.get(name, {}),
        )
        if channel is not None:
            channels[key] = channel
    try:
        text = read_store_text(root, RECORDING_FILE)
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{RECORDING_FILE} is not valid UTF-8 at byte {error.start}"
        ) from error
    return Bundle(
        root=root,
        version=version,
        recording=None if text is None else parse_recording_text(text),
        channels=MappingProxyType(channels),
    )


def open_bundle(
    location: BundleLocation, *, storage_options: dict[str, Any] | None = None
) -> Bundle:
    """Open the bundle at location for reading.

    location is a local path, a file URL, an fsspec URL, a zarr Store, or a
    StorePath. A relative path resolves against the working directory at the
    open. storage_options goes to fsspec with a remote URL. The open reads
    the root zarr.json and recording.json and no other object. It skips a
    channel group of unknown kind with a UserWarning.

    Raises ValueError if the root zarr.json carries no consolidated metadata
    or the root tsz.version is not a major.minor string with major version 1.
    Also raises ValueError if a channel group or recording.json does not
    parse. Also raises ValueError if a file URL names a host other than
    localhost or holds a query or fragment. Raises FileNotFoundError if
    location has no root zarr.json and ValueError if its root is an array.
    Raises TypeError if storage_options comes with a memory URL or a
    location that is not a remote URL. Raises ImportError if a remote URL
    needs fsspec and fsspec is not installed.
    """
    return parse_bundle(open_root(location, storage_options))
