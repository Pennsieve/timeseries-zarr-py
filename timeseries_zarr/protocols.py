"""Typed source protocols the writer consumes for channel data."""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


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

    def offset_uv(self) -> float:
        """Return the channel's DC offset, in the unit its samples are stored in.

        The writer subtracts it before folding, so the statistics hold the
        signal without its bias and a reader adds it back in float64. A source
        that has no offset to declare returns 0.0.
        """
        ...

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the half-open [start, stop) sample window as float32."""
        ...


@runtime_checkable
class UnitChannelSource(Protocol):
    """A unit (spike) channel the writer reads events, classifications, and waveforms from.

    rate_hz is the waveform sample rate, not an event rate.
    """

    @property
    def id(self) -> str:
        """Opaque upstream unit identifier."""
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
        """Return the waveform sample rate in hertz."""
        ...

    def start_us(self) -> int:
        """Return the wall-clock microseconds of the recording start."""
        ...

    def num_events(self) -> int:
        """Return the total number of spike events."""
        ...

    def num_labels(self) -> int:
        """Return how many distinct labels the channel can use.

        The width of a count row, so it is the label space rather than the
        labels actually seen. 0 means the channel carries no labels.
        """
        ...

    def points_per_event(self) -> int:
        """Return the number of waveform samples stored per event."""
        ...

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of absolute-microsecond timestamps."""
        ...

    def read_labels(self, start: int, stop: int) -> npt.NDArray[np.uint16]:
        """Return the half-open [start, stop) window of per-event labels.

        Cluster ids for sorted spikes. u2, so a high-density sort is not
        capped at 256.
        """
        ...

    def read_waveforms(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the float32 waveforms for events [start, stop).

        One row per event, points_per_event samples wide.
        """
        ...


@runtime_checkable
class AnnotationChannelSource(Protocol):
    """An annotation channel: marks placed on the timeline by a person or a detector.

    The same event shape as a spike channel, with a different set of columns.
    Which columns a source offers is what a view reads to decide how to draw it,
    so each one is optional and announced rather than assumed.
    """

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
        """Physical unit of the per-event value, when there is one."""
        ...

    def start_us(self) -> int:
        """Return the wall-clock microseconds of the first mark."""
        ...

    def num_events(self) -> int:
        """Return the total number of marks."""
        ...

    def num_labels(self) -> int:
        """Return the width of the label space; 0 when the channel has no labels."""
        ...

    def label_names(self) -> list[str] | None:
        """Return names for the label values, index-aligned, or None for bare ids."""
        ...

    def max_duration_us(self) -> int | None:
        """Return an upper bound on any duration, or None when marks are points.

        Not None is what declares the durations column present. The bound may be
        loose: a reader widens its backward search by it to catch an interval
        still open at a window's left edge, and pays only a slightly wider read.
        """
        ...

    def body_media_type(self) -> str | None:
        """Return the media type of the payloads, or None when there are none.

        Not None is what declares the bodies column present.
        """
        ...

    def has_values(self) -> bool:
        """Whether each mark carries a numeric measurement."""
        ...

    def has_channel_refs(self) -> bool:
        """Whether marks name the continuous channels they belong to."""
        ...

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of absolute-microsecond timestamps."""
        ...

    def read_durations(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of durations in microseconds."""
        ...

    def read_labels(self, start: int, stop: int) -> npt.NDArray[np.uint16]:
        """Return the [start, stop) window of per-mark labels."""
        ...

    def read_values(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the [start, stop) window of per-mark measurements."""
        ...

    def read_bodies(self, start: int, stop: int) -> list[bytes]:
        """Return one payload per mark in [start, stop), already encoded.

        Payloads are returned whole rather than as a byte stream because the
        format stores them as one chunk a reader opens in full.
        """
        ...

    def read_channel_refs(
        self, start: int, stop: int
    ) -> list[npt.NDArray[np.uint16]]:
        """Return the channel indices each mark in [start, stop) applies to.

        One array per mark, which may be empty: a mark that belongs to the
        recording rather than to named electrodes names none.
        """
        ...
