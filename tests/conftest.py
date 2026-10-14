"""Shared pytest fixtures."""

import numpy as np
import pytest

from timeseries_zarr.columns import text_width


class ArrayContinuousSource:
    """In-memory ContinuousChannelSource backed by a float32 array."""

    def __init__(
        self,
        samples,
        *,
        id="ch-0",
        rate_hz=32000.0,
        start_us=0,
        name="ch-0",
        unit="uV",
    ):
        self._samples = np.asarray(samples, dtype=np.float32)
        self.id = id
        self.name = name
        self.unit = unit
        self._rate_hz = rate_hz
        self._start_us = start_us

    def rate_hz(self):
        return self._rate_hz

    def start_us(self):
        return self._start_us

    def num_samples(self):
        return int(self._samples.shape[0])

    def read_samples(self, start, stop):
        return self._samples[start:stop]


@pytest.fixture
def continuous_source():
    """Return the ArrayContinuousSource factory."""
    return ArrayContinuousSource


class ArrayEventSource:
    """In-memory EventChannelSource backed by arrays.

    events are int64 absolute microseconds. An optional column is offered only
    when its data is given. The label count, the text width, and the duration
    bound come from the data unless passed.
    """

    def __init__(
        self,
        events,
        *,
        durations=None,
        labels=None,
        label_names=None,
        text=None,
        channel_ids=None,
        waveform_mean=None,
        waveform_period_us=100.0,
        max_duration_us=None,
        start_us=None,
        id="evt-0",
        name="evt-0",
        unit=None,
    ):
        self._events = np.asarray(events, dtype=np.int64)
        self._durations = (
            None if durations is None else np.asarray(durations, dtype=np.int64)
        )
        self._labels = (
            None if labels is None else np.asarray(labels, dtype=np.uint16)
        )
        self._label_names = label_names
        self._text = None if text is None else list(text)
        self._channel_ids = (
            None if channel_ids is None else [list(ids) for ids in channel_ids]
        )
        self._waveform_mean = (
            None
            if waveform_mean is None
            else np.asarray(waveform_mean, dtype=np.float32)
        )
        self._waveform_period_us = waveform_period_us
        self._max_duration_us = max_duration_us
        self._start_us = start_us
        self.id = id
        self.name = name
        self.unit = unit

    def start_us(self):
        if self._start_us is not None:
            return self._start_us
        return int(self._events[0]) if self._events.shape[0] else 0

    def num_events(self):
        return int(self._events.shape[0])

    def columns(self):
        present = {
            "durations": self._durations,
            "labels": self._labels,
            "text": self._text,
            "channels": self._channel_ids,
            "waveform_mean": self._waveform_mean,
        }
        return frozenset(
            key for key, value in present.items() if value is not None
        )

    def label_names(self):
        return self._label_names

    def num_labels(self):
        if self._labels is None:
            return 0
        if self._label_names is not None:
            return len(self._label_names)
        return int(self._labels.max()) + 1 if self._labels.shape[0] else 0

    def max_duration_us(self):
        if self._durations is None:
            return None
        if self._max_duration_us is not None:
            return self._max_duration_us
        return int(self._durations.max(initial=0))

    def text_width(self):
        return 0 if self._text is None else text_width(self._text)

    def read_events(self, start, stop):
        return self._events[start:stop]

    def read_durations(self, start, stop):
        return self._durations[start:stop]

    def read_labels(self, start, stop):
        return self._labels[start:stop]

    def read_text(self, start, stop):
        return self._text[start:stop]

    def read_channel_ids(self, start, stop):
        return self._channel_ids[start:stop]

    def waveform_mean(self):
        if self._waveform_mean is None:
            raise ValueError("this source has no waveform_mean column")
        return self._waveform_mean

    def waveform_period_us(self):
        return self._waveform_period_us


@pytest.fixture
def event_source():
    """Return the ArrayEventSource factory."""
    return ArrayEventSource
