"""Typed source protocols the writer consumes for channel data."""

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from timeseries_zarr.types import EventColumn


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
    """Return the provenance of source, or None when it has no provenance method."""
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
        """Return the total number of raw samples in the channel."""
        ...


@runtime_checkable
class ContinuousChannelSource(ContinuousChannelMetadata, Protocol):
    """A continuous channel the writer reads raw samples from."""

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the half-open [start, stop) sample window as float32."""
        ...


@runtime_checkable
class MultiChannelSource(Protocol):
    """Continuous channels the writer reads together, one window at a time.

    A source over a format that stores every channel in one stream implements
    this protocol. The source decodes the stream once per write.

    channels returns the channels in channel order, the same on every call. A
    source must return a stored list. The writer calls channels more than once
    per write.

    iter_windows yields one window at a time, and the writer calls it once per
    write. A window is a sequence with one rank-1 float32 block per channel, in
    channel order. The blocks of one window may differ in length, and a block
    may be empty. A channel whose samples end before the others gives an empty
    block in each later window. For each channel, the blocks concatenate to
    its samples, and their lengths total num_samples. The columns of a (rows,
    channels) array are the blocks of a window.

    Each channel may define provenance as ProvenanceSource declares it.
    """

    def channels(self) -> Sequence[ContinuousChannelMetadata]:
        """Return the channels in channel order."""
        ...

    def iter_windows(self) -> Iterator[Sequence[npt.NDArray[np.float32]]]:
        """Yield each window as a sequence of one block per channel."""
        ...


@runtime_checkable
class EventChannelSource(Protocol):
    """An event channel the writer reads event times and optional columns from.

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
        """Return the number of label categories, or 0 without a labels column."""
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
        """Return the ids of the channels each event in [start, stop) applies to.

        An empty list means the event applies to the whole recording.
        """
        ...

    def waveform_mean(self) -> npt.NDArray[np.float32]:
        """Return the mean snippet of each label category, shaped (labels, samples)."""
        ...

    def waveform_period_us(self) -> float:
        """Return the microseconds between adjacent snippet samples."""
        ...
