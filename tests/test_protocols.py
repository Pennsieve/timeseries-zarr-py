import numpy as np

from timeseries_zarr.protocols import ContinuousChannelSource, UnitChannelSource


class FakeContinuous:
    id = "ch-0"
    name = "Channel 0"
    unit = "uV"

    def rate_hz(self):
        return 32000.0

    def start_us(self):
        return 0

    def num_samples(self):
        return 4

    def offset_uv(self):
        return 0.0

    def read_samples(self, start, stop):
        return np.zeros(stop - start, dtype=np.float32)


def test_conforming_instance_is_recognized():
    assert isinstance(FakeContinuous(), ContinuousChannelSource)


def test_missing_name_attr_is_not_recognized():
    class MissingName:
        id = "ch-0"
        unit = "uV"

        def rate_hz(self):
            return 32000.0

        def start_us(self):
            return 0

        def num_samples(self):
            return 4

        def read_samples(self, start, stop):
            return np.zeros(stop - start, dtype=np.float32)

    assert not isinstance(MissingName(), ContinuousChannelSource)


def test_missing_unit_attr_is_not_recognized():
    class MissingUnit:
        id = "ch-0"
        name = "Channel 0"

        def rate_hz(self):
            return 32000.0

        def start_us(self):
            return 0

        def num_samples(self):
            return 4

        def read_samples(self, start, stop):
            return np.zeros(stop - start, dtype=np.float32)

    assert not isinstance(MissingUnit(), ContinuousChannelSource)


def test_missing_method_is_not_recognized():
    class MissingReadSamples:
        id = "ch-0"

        def rate_hz(self):
            return 32000.0

        def start_us(self):
            return 0

        def num_samples(self):
            return 4

    assert not isinstance(MissingReadSamples(), ContinuousChannelSource)


def test_missing_id_attr_is_not_recognized():
    class MissingId:
        def rate_hz(self):
            return 32000.0

        def start_us(self):
            return 0

        def num_samples(self):
            return 4

        def read_samples(self, start, stop):
            return np.zeros(stop - start, dtype=np.float32)

    assert not isinstance(MissingId(), ContinuousChannelSource)


class FakeUnit:
    id = "u-0"
    name = "Unit 0"
    unit = "uV"

    def rate_hz(self):
        return 30000.0

    def start_us(self):
        return 0

    def num_events(self):
        return 3

    def num_labels(self):
        return 1

    def points_per_event(self):
        return 32

    def read_events(self, start, stop):
        return np.zeros(stop - start, dtype=np.int64)

    def read_labels(self, start, stop):
        return np.zeros(stop - start, dtype=np.uint16)

    def read_waveforms(self, start, stop):
        return np.zeros((stop - start, 32), dtype=np.float32)


def test_conforming_unit_instance_is_recognized():
    assert isinstance(FakeUnit(), UnitChannelSource)


def test_unit_missing_name_or_unit_is_not_recognized():
    class MissingNameUnit:
        id = "u-0"

        def rate_hz(self):
            return 30000.0

        def start_us(self):
            return 0

        def num_events(self):
            return 3

        def points_per_event(self):
            return 32

        def read_events(self, start, stop):
            return np.zeros(stop - start, dtype=np.int64)

        def read_labels(self, start, stop):
            return np.zeros(stop - start, dtype=np.uint16)

        def read_waveforms(self, start, stop):
            return np.zeros((stop - start, 32), dtype=np.float32)

    assert not isinstance(MissingNameUnit(), UnitChannelSource)


def test_unit_missing_method_is_not_recognized():
    class MissingReadWaveforms:
        id = "u-0"

        def rate_hz(self):
            return 30000.0

        def start_us(self):
            return 0

        def num_events(self):
            return 3

        def points_per_event(self):
            return 32

        def read_events(self, start, stop):
            return np.zeros(stop - start, dtype=np.int64)

        def read_labels(self, start, stop):
            return np.zeros(stop - start, dtype=np.uint16)

    assert not isinstance(MissingReadWaveforms(), UnitChannelSource)
