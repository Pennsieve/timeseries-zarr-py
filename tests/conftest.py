"""Shared pytest fixtures."""

import numpy as np
import pytest
from pynwb.testing.mock.ecephys import mock_ElectricalSeries


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
        offset_uv=0.0,
    ):
        self._samples = np.asarray(samples, dtype=np.float32)
        self._offset_uv = offset_uv
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

    def offset_uv(self):
        return self._offset_uv

    def read_samples(self, start, stop):
        return self._samples[start:stop]


@pytest.fixture
def continuous_source():
    """Return the ArrayContinuousSource factory."""
    return ArrayContinuousSource


class ArrayUnitSource:
    """In-memory UnitChannelSource backed by arrays.

    events are int64 absolute microseconds. labels and waveforms default
    to zeros when not given.
    """

    def __init__(
        self,
        events,
        *,
        labels=None,
        waveforms=None,
        points_per_event=4,
        num_labels=1,
        id="unit-0",
        rate_hz=32000.0,
        start_us=0,
        name="unit-0",
        unit="uV",
    ):
        self._events = np.asarray(events, dtype=np.int64)
        n = int(self._events.shape[0])
        self._points_per_event = points_per_event
        self._num_labels = num_labels
        self._labels = (
            np.zeros(n, dtype=np.uint16)
            if labels is None
            else np.asarray(labels, dtype=np.uint16)
        )
        self._waveforms = (
            np.zeros((n, points_per_event), dtype=np.float32)
            if waveforms is None
            else np.asarray(waveforms, dtype=np.float32)
        )
        self.id = id
        self.name = name
        self.unit = unit
        self._rate_hz = rate_hz
        self._start_us = start_us

    def rate_hz(self):
        return self._rate_hz

    def start_us(self):
        return self._start_us

    def num_events(self):
        return int(self._events.shape[0])

    def num_labels(self):
        return self._num_labels

    def points_per_event(self):
        return self._points_per_event

    def read_events(self, start, stop):
        return self._events[start:stop]

    def read_labels(self, start, stop):
        return self._labels[start:stop]

    def read_waveforms(self, start, stop):
        return self._waveforms[start:stop]


@pytest.fixture
def unit_source():
    """Return the ArrayUnitSource factory."""
    return ArrayUnitSource


@pytest.fixture
def electrical_series():
    """Return the mock_ElectricalSeries factory.

    The mock builds its own electrode table and accepts data, rate, and
    starting_time overrides.
    """
    return mock_ElectricalSeries
