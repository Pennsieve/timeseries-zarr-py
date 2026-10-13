"""Typed source protocols the writer consumes for channel data."""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from timeseries_zarr.types import EventColumn


@runtime_checkable
class ContinuousChannelSource(Protocol):
    """A continuous channel the writer reads raw samples from."""

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
        """Physical unit of the stored samples, for example "uV"."""
        ...

    def rate_hz(self) -> float:
        """Return the channel's sample rate in hertz."""
        ...

    def start_us(self) -> int:
        """Return the wall-clock microseconds of sample index 0."""
        ...

    def num_samples(self) -> int:
        """Return the total number of raw samples in the channel."""
        ...

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the half-open [start, stop) sample window as float32."""
        ...


@runtime_checkable
class EventChannelSource(Protocol):
    """An event channel the writer reads event times and optional columns from.

    Every channel has event times. Each other column is optional, and columns
    names the ones this source provides: the writer calls a column's reader
    only for a column named there. Times are wall-clock microseconds, and the
    writer places them relative to the bundle origin.
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

    def start_us(self) -> int:
        """Return the earliest wall-clock microseconds the channel holds.

        No event precedes it. It is the first event, or the recording start
        when that is earlier or the channel has no events.
        """
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
        """Return the [start, stop) window of wall-clock microsecond times.

        The times are non-decreasing across the whole channel.
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
