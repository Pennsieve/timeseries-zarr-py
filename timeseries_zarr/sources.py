"""Writer sources that read the channels of an open bundle.

A source gives the writer wall-clock times in microseconds since the Unix
epoch. It adds epoch_us, the wall-clock time of the reference time, to every
time the bundle stores.
"""

import copy
from collections.abc import Iterable, Mapping
from typing import get_args

import numpy as np
import numpy.typing as npt

from timeseries_zarr.columns import decode_channels_rows, decode_event_text
from timeseries_zarr.constants import (
    CHANNELS_KEY,
    DURATIONS_KEY,
    LABELS_KEY,
    MAX_LABELS,
    TEXT_KEY,
    WAVEFORM_MEAN_KEY,
)
from timeseries_zarr.read_bundle import Bundle
from timeseries_zarr.read_continuous import ContinuousChannel
from timeseries_zarr.read_event import EventChannel, Snippets
from timeseries_zarr.recording import epoch_us
from timeseries_zarr.types import EventColumn, RecordingMeta
from timeseries_zarr.windows import (
    check_label_range,
    check_not_string,
    integer_argument,
)
from timeseries_zarr.zarr_io import ZarrArray, read_rows

_EVENT_COLUMNS: tuple[EventColumn, ...] = get_args(EventColumn.__value__)
"""Every optional column of an event source, in EventColumn order."""


def _channel_id(channel: ContinuousChannel | EventChannel) -> str:
    """Return the id of channel, or its key as a decimal string without one."""
    return str(channel.key) if channel.id is None else channel.id


def _provenance(
    channel: ContinuousChannel | EventChannel,
) -> dict[str, object] | None:
    """Return a deep copy of the channel provenance, or None without one."""
    provenance = channel.provenance
    return None if provenance is None else copy.deepcopy(dict(provenance))


def _event_columns(names: Iterable[str], argument: str) -> set[EventColumn]:
    """Return the event columns that names holds.

    argument names the argument in an error message. Raises TypeError if
    names is a string and ValueError if it holds a name outside EventColumn.
    """
    check_not_string(names, argument)
    given = list(names)
    for name in given:
        if name not in _EVENT_COLUMNS:
            raise ValueError(
                f"{argument} must name only the columns {_EVENT_COLUMNS!r}, "
                f"got {name!r}"
            )
    return {column for column in _EVENT_COLUMNS if column in given}


def _carried_columns(
    channel: EventChannel, wanted: set[EventColumn]
) -> frozenset[EventColumn]:
    """Return the columns of wanted that channel holds.

    The result leaves out a text column of width 0, and waveform_mean when it
    leaves out labels.
    """
    held: dict[EventColumn, bool] = {
        "durations": channel.durations is not None,
        "labels": channel.labels is not None,
        "text": channel.text is not None and channel.text.shape[1] > 0,
        "channels": channel.channels is not None,
        "waveform_mean": channel.waveform_mean is not None,
    }
    carried = {column for column in wanted if held[column]}
    if "labels" not in carried:
        carried.discard("waveform_mean")
    return frozenset(carried)


class BundleContinuousSource:
    """A continuous channel of an open bundle as a writer source.

    epoch_us is the wall-clock time of the reference time in microseconds
    since the Unix epoch. Sample 0 starts at epoch_us plus the offset_us of
    the channel. Raises TypeError if epoch_us is not an integer.
    """

    __slots__ = ("_channel", "_epoch_us")

    def __init__(self, channel: ContinuousChannel, *, epoch_us: int) -> None:
        """Wrap channel with the wall-clock time of its reference time."""
        self._channel = channel
        self._epoch_us = integer_argument(epoch_us, "epoch_us")

    @property
    def id(self) -> str:
        """Id of the channel, or its key as a decimal string without one."""
        return _channel_id(self._channel)

    @property
    def name(self) -> str:
        """Name of the channel."""
        return self._channel.name

    @property
    def unit(self) -> str:
        """Unit of the samples."""
        return self._channel.unit

    def provenance(self) -> dict[str, object] | None:
        """Return a copy of the channel provenance, or None without one."""
        return _provenance(self._channel)

    def rate_hz(self) -> float:
        """Return the sample rate in hertz."""
        return self._channel.rate_hz

    def start_us(self) -> int:
        """Return the wall-clock microseconds of sample 0."""
        return self._epoch_us + self._channel.offset_us

    def num_samples(self) -> int:
        """Return the number of raw samples."""
        return self._channel.num_samples

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return raw samples start through stop - 1 as float32."""
        samples = read_rows(self._channel.raw, start, stop)
        return np.asarray(samples, dtype=np.float32)


class BundleEventSource:
    """An event channel of an open bundle as a writer source.

    ids maps the key of each channel the writer receives to its id.
    epoch_us is the wall-clock time of the reference time in microseconds
    since the Unix epoch. columns names the optional columns to carry, and
    None names every one. The source carries a named column when the channel
    holds it, except a text column of width 0, and waveform_mean when it
    carries no labels. The source never carries values, waveforms, or the
    event_sample of waveform_mean.

    Raises TypeError if epoch_us is not an integer or columns is a string.
    Raises ValueError if columns names a column outside EventColumn.
    """

    __slots__ = (
        "_arrays",
        "_channel",
        "_columns",
        "_epoch_us",
        "_ids",
        "_max_duration",
        "_num_labels",
    )

    def __init__(
        self,
        channel: EventChannel,
        ids: Mapping[int, str],
        *,
        epoch_us: int,
        columns: Iterable[str] | None = None,
    ) -> None:
        """Wrap channel with the ids of the bundle and its reference time."""
        wanted = (
            set(_EVENT_COLUMNS)
            if columns is None
            else _event_columns(columns, "columns")
        )
        self._channel = channel
        self._ids = dict(ids)
        self._epoch_us = integer_argument(epoch_us, "epoch_us")
        self._columns = _carried_columns(channel, wanted)
        self._arrays: dict[str, ZarrArray | None] = {
            DURATIONS_KEY: channel.durations,
            LABELS_KEY: channel.labels,
            TEXT_KEY: channel.text,
            CHANNELS_KEY: channel.channels,
        }
        # A fallback reads a whole column, and a caller can ask again.
        self._max_duration: int | None = None
        self._num_labels: int | None = None

    @property
    def id(self) -> str:
        """Id of the channel, or its key as a decimal string without one."""
        return _channel_id(self._channel)

    @property
    def name(self) -> str:
        """Name of the channel."""
        return self._channel.name

    @property
    def unit(self) -> str | None:
        """Unit of the channel, or None without one."""
        return self._channel.unit

    def provenance(self) -> dict[str, object] | None:
        """Return a copy of the channel provenance, or None without one."""
        return _provenance(self._channel)

    def num_events(self) -> int:
        """Return the number of events."""
        return self._channel.num_events

    def columns(self) -> frozenset[EventColumn]:
        """Return the optional columns the source carries."""
        return self._columns

    def label_names(self) -> list[str] | None:
        """Return the label names of the channel, or None without them."""
        names = self._channel.label_names
        return None if names is None else list(names)

    def num_labels(self) -> int:
        """Return the number of label categories, or 0 without labels.

        The number is the length of label_names, else the label columns of a
        count level, else the rows of waveform_mean. Without any of them it
        is one more than the largest label, or 0 without events.
        """
        if "labels" not in self._columns:
            return 0
        channel = self._channel
        if channel.label_names is not None:
            return len(channel.label_names)
        level = next(iter(channel.count_levels.values()), None)
        if level is not None:
            return level.num_labels
        if channel.waveform_mean is not None:
            return channel.waveform_mean.array.shape[0]
        if self._num_labels is None:
            labels = read_rows(self._array(LABELS_KEY), 0, channel.num_events)
            self._num_labels = int(labels.max()) + 1 if labels.size else 0
        return self._num_labels

    def max_duration_us(self) -> int | None:
        """Return the bound on every duration, or None without durations.

        The bound is the max_duration_us of the channel, else the largest
        duration, or 0 without events.
        """
        if "durations" not in self._columns:
            return None
        channel = self._channel
        if channel.max_duration_us is not None:
            return channel.max_duration_us
        if self._max_duration is None:
            durations = read_rows(
                self._array(DURATIONS_KEY), 0, channel.num_events
            )
            self._max_duration = int(durations.max(initial=0))
        return self._max_duration

    def text_width(self) -> int:
        """Return the byte width of the text column, or 0 without text."""
        if "text" not in self._columns:
            return 0
        return self._array(TEXT_KEY).shape[1]

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the wall-clock times of events start through stop - 1."""
        times = read_rows(self._channel.events, start, stop)
        return np.asarray(times, dtype=np.int64) + self._epoch_us

    def read_durations(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the durations of events start through stop - 1.

        Raises ValueError if the channel has no durations.
        """
        rows = read_rows(self._array(DURATIONS_KEY), start, stop)
        return np.asarray(rows, dtype=np.int64)

    def read_labels(self, start: int, stop: int) -> npt.NDArray[np.uint16]:
        """Return the labels of events start through stop - 1.

        Raises ValueError if the channel has no labels, or if a label is
        below 0 or at least MAX_LABELS.
        """
        rows = read_rows(self._array(LABELS_KEY), start, stop)
        check_label_range(f"channel {self._channel.key}", rows, MAX_LABELS)
        return np.asarray(rows, dtype=np.uint16)

    def read_text(self, start: int, stop: int) -> list[str]:
        """Return the decoded texts of events start through stop - 1.

        Raises ValueError if the channel has no text or the text of an event
        is not valid UTF-8.
        """
        rows = read_rows(self._array(TEXT_KEY), start, stop)
        return decode_event_text(
            rows,
            range(start, start + rows.shape[0]),
            f"channel {self._channel.key}",
        )

    def read_channel_ids(self, start: int, stop: int) -> list[list[str]]:
        """Return the channel ids each event start through stop - 1 names.

        An empty list means the event applies to the whole recording. Raises
        ValueError if the channel has no channels column or an event names a
        channel key that ids lacks.
        """
        rows = read_rows(self._array(CHANNELS_KEY), start, stop)
        id_lists = []
        for index, keys in enumerate(decode_channels_rows(rows), start=start):
            for key in keys:
                if key not in self._ids:
                    raise ValueError(
                        f"channel {self._channel.key}: event {index} names "
                        f"channel key {key}, and the source maps the channel "
                        f"keys {sorted(self._ids)}"
                    )
            id_lists.append([self._ids[key] for key in keys])
        return id_lists

    def waveform_mean(self) -> npt.NDArray[np.float32]:
        """Return the mean snippet of each label, shaped (labels, samples).

        Raises ValueError if the channel has no waveform_mean.
        """
        values = self._snippets().read().values
        return np.asarray(values, dtype=np.float32)

    def waveform_period_us(self) -> float:
        """Return the microseconds between adjacent snippet samples.

        Raises ValueError if the channel has no waveform_mean.
        """
        return self._snippets().period_us

    def _array(self, name: str) -> ZarrArray:
        """Return the array of the per-event column called name.

        Raises ValueError if the channel has no such column.
        """
        array = self._arrays[name]
        if array is None:
            raise ValueError(_missing_column(self._channel.key, name))
        return array

    def _snippets(self) -> Snippets:
        """Return the waveform_mean of the channel.

        Raises ValueError if the channel has none.
        """
        snippets = self._channel.waveform_mean
        if snippets is None:
            raise ValueError(
                _missing_column(self._channel.key, WAVEFORM_MEAN_KEY)
            )
        return snippets


def _missing_column(key: int, name: str) -> str:
    """Return the error message for a column that channel key lacks."""
    return (
        f"channel {key} has no {name} column; read only a column that "
        "columns() names"
    )


def bundle_epoch_us(bundle: Bundle) -> int:
    """Return the reference time in microseconds since the Unix epoch.

    The reference time is the reference_time of recording.json, and the
    arithmetic is exact in integers. A bundle without that time gives 0.
    """
    recording = bundle.recording
    reference = None if recording is None else recording.reference_time
    return 0 if reference is None else epoch_us(reference)


def recording_meta(bundle: Bundle) -> RecordingMeta | None:
    """Return the recording input that rewrites the recording.json of bundle.

    A bundle without recording.json gives None. The result holds the
    reference time and the session start of recording.json, and subject,
    session, and source as deep copies in plain dicts. The writer uses the
    reference time as it is, so a rewrite keeps every offset and a channel
    that begins before it gets a negative offset. session leaves out
    start_time, which the writer puts back from session_start_time.
    wall_clock is whether recording.json holds a reference_time, so a
    rewrite of a bundle without one stays without one.
    """
    recording = bundle.recording
    if recording is None:
        return None
    session = copy.deepcopy(dict(recording.session))
    session.pop("start_time", None)
    return RecordingMeta(
        session_start_time=recording.session_start_time,
        reference_time=recording.reference_time,
        subject=copy.deepcopy(dict(recording.subject)),
        session=session,
        source=copy.deepcopy(dict(recording.source)),
        wall_clock=recording.reference_time is not None,
    )


def channel_ids(
    bundle: Bundle, keys: Iterable[int] | None = None
) -> dict[int, str]:
    """Return the id of each channel of bundle that keys names, in key order.

    None names every channel, and the result leaves out a key the bundle
    lacks. A channel without an id gets its key as a decimal string. Raises
    ValueError if two of those channels share an id.
    """
    chosen = None if keys is None else set(keys)
    ids: dict[int, str] = {}
    owners: dict[str, int] = {}
    for key, channel in bundle.channels.items():
        if chosen is not None and key not in chosen:
            continue
        channel_id = _channel_id(channel)
        if channel_id in owners:
            raise ValueError(
                f"channels {owners[channel_id]} and {key} share the id "
                f"{channel_id!r}; keys must name at most one of them"
            )
        owners[channel_id] = key
        ids[key] = channel_id
    return ids


def bundle_sources(
    bundle: Bundle,
    keys: Iterable[int] | None = None,
    *,
    epoch_us: int | None = None,
    drop_columns: Iterable[str] = frozenset(),
) -> tuple[list[BundleContinuousSource], list[BundleEventSource]]:
    """Return writer sources for channels of bundle, each list in key order.

    keys names the channels to keep, and None keeps every channel. epoch_us
    is the wall-clock time of the reference time. None takes the
    reference_time of recording.json in microseconds since the Unix epoch, or
    0 without one.
    drop_columns names the event columns that
    no event source carries. Each event source maps the key of every kept
    channel to its id. A key whose channel has no id maps to the key as a
    decimal string. An event that names a channel outside keys then makes
    read_channel_ids raise ValueError during write_bundle, unless
    drop_columns holds channels. The output_path of write_bundle must differ
    from the location of bundle.

    Raises KeyError if keys names a channel the bundle lacks. Raises
    TypeError if epoch_us is not an integer or drop_columns is a string.
    Raises ValueError if drop_columns names a column outside EventColumn or
    two kept channels share an id.
    """
    wanted = set(_EVENT_COLUMNS) - _event_columns(drop_columns, "drop_columns")
    chosen = set(bundle.channels)
    if keys is not None:
        chosen = set()
        for key in keys:
            if key not in bundle.channels:
                listed = (
                    ", ".join(str(known) for known in bundle.channels) or "none"
                )
                raise KeyError(
                    f"the bundle has no channel {key!r}; its channel keys "
                    f"are {listed}"
                )
            chosen.add(key)
    epoch = (
        bundle_epoch_us(bundle)
        if epoch_us is None
        else integer_argument(epoch_us, "epoch_us")
    )
    ids = channel_ids(bundle, chosen)
    continuous: list[BundleContinuousSource] = []
    events: list[BundleEventSource] = []
    for key, channel in bundle.channels.items():
        if key not in chosen:
            continue
        if isinstance(channel, ContinuousChannel):
            continuous.append(BundleContinuousSource(channel, epoch_us=epoch))
        else:
            events.append(
                BundleEventSource(channel, ids, epoch_us=epoch, columns=wanted)
            )
    return continuous, events
