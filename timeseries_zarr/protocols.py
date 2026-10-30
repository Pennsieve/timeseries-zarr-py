"""Typed source protocols the writer consumes for channel data."""

import inspect
from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from timeseries_zarr.runs import Run
from timeseries_zarr.types import EventColumn


def missing_members(source: object, protocol: type) -> list[str]:
    """Return the sorted names of the protocol members that source lacks.

    A name is missing under the same test isinstance applies to a runtime
    checkable protocol. A method set to None counts as missing, so a source
    that fails isinstance has at least one.

    Args:
        source: Object to inspect.
        protocol: Runtime checkable protocol class that lists the members.

    Returns:
        The missing member names in sorted order, empty when none is missing.
    """
    # Python 3.12 has no typing.get_protocol_members.
    members: frozenset[str] = getattr(protocol, "__protocol_attrs__")  # noqa: B009
    data_members: frozenset[str] = getattr(
        protocol, "__non_callable_proto_members__", frozenset()
    )
    missing = []
    for name in members:
        try:
            value = inspect.getattr_static(source, name)
        except AttributeError:
            missing.append(name)
            continue
        if value is None and name not in data_members:
            missing.append(name)
    return sorted(missing)


@runtime_checkable
class ProvenanceSource(Protocol):
    """An optional addition to a channel source that records its producer."""

    def provenance(self) -> dict[str, object] | None:
        """Return what produced the channel, or None when unrecorded.

        The object names the producing software, for example its name and
        version. Its content is the producer's choice. It must be JSON
        serializable with finite numbers, and it must not identify a person
        or hold a date.
        """
        ...


def source_provenance(source: object) -> dict[str, object] | None:
    """Return the provenance of source, or None without a provenance method."""
    if isinstance(source, ProvenanceSource):
        return source.provenance()
    return None


@runtime_checkable
class ContinuousChannelMetadata(Protocol):
    """The identity, unit, rate, start, and length of a continuous channel."""

    @property
    def id(self) -> str:
        """Opaque upstream channel identifier."""
        ...

    @property
    def name(self) -> str:
        """Human-readable display label for the channel."""
        ...

    @property
    def unit(self) -> str:
        """Unit of the channel's samples, for example "uV"."""
        ...

    def rate_hz(self) -> float:
        """Return the channel's sample rate in hertz."""
        ...

    def start_us(self) -> int:
        """Return the microseconds since the Unix epoch of sample index 0."""
        ...

    def num_samples(self) -> int:
        """Return the number of samples the channel delivers.

        A pause between runs adds no sample to the count.
        """
        ...


@runtime_checkable
class RunsSource(Protocol):
    """An optional addition to a continuous channel that states its runs."""

    def runs(self) -> Sequence[Run]:
        """Return the runs of the channel in time order.

        A run is a span of samples one sample period apart. The first run
        starts at start_us, and the run lengths total num_samples. A pause
        between runs is not a sample, so the source does not deliver one.
        """
        ...


def source_runs(source: ContinuousChannelMetadata) -> Sequence[Run]:
    """Return the runs of source, or one run without a runs method.

    The one run starts at start_us and holds num_samples samples.
    """
    if isinstance(source, RunsSource):
        return list(source.runs())
    return [Run(source.start_us(), source.num_samples())]


@runtime_checkable
class ContinuousChannelSource(ContinuousChannelMetadata, Protocol):
    """A continuous channel the writer reads raw samples from."""

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the half-open [start, stop) sample window as float32."""
        ...


type Window = Sequence[npt.ArrayLike] | npt.NDArray[np.generic]
"""One window of a MultiChannelSource: a sequence of blocks or a (rows, channels) array."""


@runtime_checkable
class MultiChannelSource(Protocol):
    """Continuous channels the writer reads together, one window at a time.

    A source over a format that stores every channel in one stream implements
    this protocol. channels returns the channels in channel order.
    iter_windows yields one window at a time, and the writer calls it once per
    write. The writer reads the next window while it writes the previous one,
    so the source must not overwrite the arrays of a window it has yielded.

    A window is a rank-2 array of (rows, channels), or a sequence with one
    rank-1 block per channel, in channel order. The columns of the array are
    the blocks. The writer copies an array into channel-major order once, so
    each channel reads contiguous rows. The writer casts each block to
    float32, so a strided column view or an int16 block works. The blocks of
    one window may differ in length, and a block may be empty, but the columns
    of an array all have its row count. A channel whose samples end before the
    others gives an empty block in each later window, which only the sequence
    form can express. For each channel, the blocks concatenate to its
    recorded samples, and their lengths total num_samples. A pause between
    runs is not pushed.

    Each channel may define provenance as ProvenanceSource declares it. Each
    channel may define runs as RunsSource declares it. A channel without runs
    is one run.
    """

    def channels(self) -> Sequence[ContinuousChannelMetadata]:
        """Return the channels in channel order."""
        ...

    def iter_windows(self) -> Iterator[Window]:
        """Yield each window as one block per channel.

        A window is a sequence of rank-1 blocks or an array of (rows,
        channels).
        """
        ...


@runtime_checkable
class WaveformEventSampleSource(Protocol):
    """An optional addition to an event source with a waveform_mean column."""

    def waveform_event_sample(self) -> int:
        """Return the index of the snippet sample at the event time.

        The index is at least 0 and below the samples of a snippet. A source
        without this method places the event at sample 0.
        """
        ...


def source_waveform_event_sample(source: object) -> int:
    """Return the waveform event sample of source, or 0 without the method."""
    if isinstance(source, WaveformEventSampleSource):
        return source.waveform_event_sample()
    return 0


@runtime_checkable
class EventChannelSource(Protocol):
    """An event channel the writer reads event times and columns from.

    Every channel has event times. Each other column is optional, and columns
    names the ones this source provides: the writer calls a column's reader
    only for a column named there. Times are microseconds since the Unix
    epoch, and the writer places them relative to the reference time.
    """

    @property
    def id(self) -> str:
        """Opaque upstream identifier, stable across exports."""
        ...

    @property
    def name(self) -> str:
        """Human-readable display label for the channel."""
        ...

    @property
    def unit(self) -> str | None:
        """Unit of the channel's measurements, or None when it has none."""
        ...

    def num_events(self) -> int:
        """Return the total number of events."""
        ...

    def columns(self) -> frozenset[EventColumn]:
        """Return the optional columns this source provides."""
        ...

    def label_names(self) -> list[str] | None:
        """Return the name of each label category in label order, or None."""
        ...

    def num_labels(self) -> int:
        """Return the number of label categories, or 0 without labels.

        With a labels column this is the number of label names when the source
        has label_names, else one more than the largest label, or 1 when the
        source has no events.
        """
        ...

    def max_duration_us(self) -> int | None:
        """Return an upper bound on every duration, or None without durations.

        The bound may be loose.
        """
        ...

    def text_width(self) -> int:
        """Return the byte width of the text column, or 0 without text."""
        ...

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of event times.

        Each time is in microseconds since the Unix epoch. The times are
        non-decreasing across the whole channel.
        """
        ...

    def read_durations(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of durations in microseconds."""
        ...

    def read_labels(self, start: int, stop: int) -> npt.NDArray[np.uint16]:
        """Return the [start, stop) window of label categories."""
        ...

    def read_text(self, start: int, stop: int) -> list[str]:
        """Return the [start, stop) window of event texts."""
        ...

    def read_channel_ids(self, start: int, stop: int) -> list[list[str]]:
        """Return the channel ids that each event in [start, stop) applies to.

        An empty list means the event applies to no channel.
        """
        ...

    def waveform_mean(self) -> npt.NDArray[np.float32]:
        """Return the mean snippet of each label, shaped (labels, samples).

        The writer stores NaN for a label that has no events.
        """
        ...

    def waveform_period_us(self) -> float:
        """Return the microseconds between adjacent snippet samples."""
        ...
