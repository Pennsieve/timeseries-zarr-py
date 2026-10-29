from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from tests.conftest import ArrayContinuousSource, ArrayMultiChannelSource
from timeseries_zarr.protocols import (
    ContinuousChannelMetadata,
    ContinuousChannelSource,
    EventChannelSource,
    MultiChannelSource,
    ProvenanceSource,
    RunsSource,
    missing_members,
    source_provenance,
    source_runs,
)
from timeseries_zarr.runs import Run


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


def _event_members():
    """Return every member of a conforming event source, by name."""
    return {
        "id": "evt-0",
        "name": "Events",
        "unit": None,
        "num_events": lambda self: 3,
        "columns": lambda self: frozenset(),
        "label_names": lambda self: None,
        "num_labels": lambda self: 0,
        "max_duration_us": lambda self: None,
        "text_width": lambda self: 0,
        "read_events": lambda self, start, stop: np.zeros(
            stop - start, dtype=np.int64
        ),
        "read_durations": lambda self, start, stop: np.zeros(
            stop - start, dtype=np.int64
        ),
        "read_labels": lambda self, start, stop: np.zeros(
            stop - start, dtype=np.uint16
        ),
        "read_text": lambda self, start, stop: [""] * (stop - start),
        "read_channel_ids": lambda self, start, stop: [[]] * (stop - start),
        "waveform_mean": lambda self: np.zeros((1, 4), dtype=np.float32),
        "waveform_period_us": lambda self: 100.0,
    }


def test_conforming_event_instance_is_recognized():
    assert isinstance(type("Fake", (), _event_members())(), EventChannelSource)


@pytest.mark.parametrize("missing", sorted(_event_members()))
def test_event_source_missing_any_member_is_not_recognized(missing):
    members = {k: v for k, v in _event_members().items() if k != missing}
    assert not isinstance(type("Partial", (), members)(), EventChannelSource)


def test_event_source_with_an_extra_start_us_member_is_recognized():
    members = {**_event_members(), "start_us": lambda self: 0}
    assert isinstance(type("Fake", (), members)(), EventChannelSource)


def test_continuous_source_is_not_an_event_source():
    assert not isinstance(FakeContinuous(), EventChannelSource)


def test_event_source_is_not_a_continuous_source():
    fake = type("Fake", (), _event_members())()
    assert not isinstance(fake, ContinuousChannelSource)


def test_array_event_source_satisfies_the_protocol(event_source):
    assert isinstance(event_source([1, 2, 3]), EventChannelSource)


def test_array_event_source_offers_only_the_columns_it_is_given(event_source):
    assert event_source([1, 2]).columns() == frozenset()
    full = event_source(
        [1, 2],
        durations=[0, 5],
        labels=[0, 1],
        text=["a", "bc"],
        channel_ids=[[], ["x"]],
        waveform_mean=np.zeros((2, 3)),
    )
    assert full.columns() == {
        "durations",
        "labels",
        "text",
        "channels",
        "waveform_mean",
    }


def test_array_event_source_derives_label_count_text_width_and_duration_bound(
    event_source,
):
    source = event_source(
        [1, 2, 3],
        durations=[0, 7, 3],
        labels=[0, 2, 1],
        text=["a", "caf\N{LATIN SMALL LETTER E WITH ACUTE}", ""],
    )
    assert source.num_labels() == 3
    assert source.text_width() == 5
    assert source.max_duration_us() == 7


def test_array_event_source_without_optional_columns_reports_none_or_zero(
    event_source,
):
    source = event_source([1, 2])
    assert source.num_labels() == 0
    assert source.text_width() == 0
    assert source.max_duration_us() is None
    assert source.label_names() is None


def test_array_event_source_label_names_set_the_label_count(event_source):
    source = event_source([1], labels=[0], label_names=["a", "b", "c", "d"])
    assert source.num_labels() == 4


def test_array_event_source_start_is_the_first_event_or_the_given_value(
    event_source,
):
    assert event_source([40, 50]).start_us() == 40
    assert event_source([]).start_us() == 0
    assert event_source([40], start_us=7).start_us() == 7


def test_array_event_source_reads_windows_of_each_column(event_source):
    source = event_source(
        [10, 20, 30],
        durations=[1, 2, 3],
        labels=[0, 1, 0],
        text=["a", "b", "c"],
        channel_ids=[[], ["x"], ["y", "z"]],
    )
    assert source.read_events(1, 3).tolist() == [20, 30]
    assert source.read_durations(0, 2).tolist() == [1, 2]
    assert source.read_labels(1, 3).tolist() == [1, 0]
    assert source.read_text(2, 3) == ["c"]
    assert source.read_channel_ids(1, 3) == [["x"], ["y", "z"]]


def test_array_event_source_without_waveform_mean_raises_on_read(event_source):
    with pytest.raises(ValueError, match="waveform_mean"):
        event_source([1]).waveform_mean()


def _multi_members() -> dict[str, Callable[[Any], object]]:
    """Return every member of a conforming multi-channel source, by name."""
    return {
        "channels": lambda self: [FakeContinuous()],
        "iter_windows": lambda self: iter([[np.zeros(4, dtype=np.float32)]]),
    }


def test_conforming_multi_channel_instance_is_recognized() -> None:
    assert isinstance(type("Fake", (), _multi_members())(), MultiChannelSource)


@pytest.mark.parametrize("missing", sorted(_multi_members()))
def test_multi_channel_source_missing_any_member_is_not_recognized(
    missing: str,
) -> None:
    members = {k: v for k, v in _multi_members().items() if k != missing}
    assert not isinstance(type("Partial", (), members)(), MultiChannelSource)


def test_a_channel_without_read_samples_is_metadata_and_not_a_source() -> None:
    class MetadataOnly:
        id = "ch-0"
        name = "Channel 0"
        unit = "uV"

        def rate_hz(self) -> float:
            return 32000.0

        def start_us(self) -> int:
            return 0

        def num_samples(self) -> int:
            return 4

    assert isinstance(MetadataOnly(), ContinuousChannelMetadata)
    assert not isinstance(MetadataOnly(), ContinuousChannelSource)


def test_a_continuous_source_is_also_channel_metadata() -> None:
    assert isinstance(FakeContinuous(), ContinuousChannelMetadata)


def test_a_multi_channel_source_is_neither_a_continuous_nor_an_event_source() -> (
    None
):
    fake = type("Fake", (), _multi_members())()
    assert not isinstance(fake, ContinuousChannelSource)
    assert not isinstance(fake, EventChannelSource)


def test_array_multi_channel_source_satisfies_the_protocol(
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    source = multi_channel_source([continuous_source(np.zeros(4))])
    assert isinstance(source, MultiChannelSource)
    assert all(
        isinstance(channel, ContinuousChannelMetadata)
        for channel in source.channels()
    )


def test_array_multi_channel_source_windows_concatenate_to_each_channel(
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    samples = [
        np.arange(20, dtype=np.float32),
        np.arange(7, dtype=np.float32) + 100,
    ]
    source = multi_channel_source(
        [continuous_source(each) for each in samples], window_rows=[6, 3]
    )
    windows = list(source.iter_windows())
    assert [len(window) for window in windows] == [2] * len(windows)
    for index, expected in enumerate(samples):
        joined = np.concatenate([window[index] for window in windows])
        assert joined.dtype == np.float32
        assert np.array_equal(joined, expected)


def test_array_multi_channel_source_gives_an_empty_block_past_a_channels_end(
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    source = multi_channel_source(
        [
            continuous_source(np.arange(10)),
            continuous_source(np.arange(3)),
        ],
        window_rows=4,
    )
    windows = list(source.iter_windows())
    assert [window[0].shape[0] for window in windows] == [4, 4, 2]
    assert [window[1].shape[0] for window in windows] == [3, 0, 0]


def test_a_source_without_provenance_still_satisfies_both_protocols():
    assert isinstance(FakeContinuous(), ContinuousChannelSource)
    fake = type("Fake", (), _event_members())()
    assert isinstance(fake, EventChannelSource)
    assert not isinstance(fake, ProvenanceSource)


def test_source_provenance_is_none_without_a_provenance_method():
    assert source_provenance(FakeContinuous()) is None


def test_source_provenance_returns_what_the_source_gives():
    class Marked(FakeContinuous):
        def provenance(self):
            return {"software": "x 1"}

    assert isinstance(Marked(), ProvenanceSource)
    assert source_provenance(Marked()) == {"software": "x 1"}


def test_missing_members_is_empty_for_a_conforming_source() -> None:
    assert missing_members(FakeContinuous(), ContinuousChannelSource) == []


@pytest.mark.parametrize("missing", sorted(_event_members()))
def test_missing_members_names_the_one_member_an_event_source_lacks(
    missing: str,
) -> None:
    members = {k: v for k, v in _event_members().items() if k != missing}
    source = type("Partial", (), members)()
    assert missing_members(source, EventChannelSource) == [missing]


def test_missing_members_sorts_the_names() -> None:
    assert missing_members(object(), MultiChannelSource) == [
        "channels",
        "iter_windows",
    ]


def test_missing_members_includes_the_members_a_protocol_inherits() -> None:
    names = missing_members(object(), ContinuousChannelSource)
    assert {"id", "read_samples"} <= set(names)


def test_a_source_without_runs_is_one_run_from_its_start_and_length():
    assert source_runs(FakeContinuous()) == [Run(0, 4)]


def test_a_source_without_runs_is_not_a_runs_source():
    assert not isinstance(FakeContinuous(), RunsSource)


def test_source_runs_returns_what_the_source_states():
    class Gapped(FakeContinuous):
        def runs(self):
            return (Run(0, 2), Run(5_000, 2))

    assert isinstance(Gapped(), RunsSource)
    assert source_runs(Gapped()) == [Run(0, 2), Run(5_000, 2)]


def test_a_runs_source_still_satisfies_the_channel_protocols():
    class Gapped(FakeContinuous):
        def runs(self):
            return (Run(0, 4),)

    assert isinstance(Gapped(), ContinuousChannelSource)
    assert isinstance(Gapped(), ContinuousChannelMetadata)


def test_source_runs_reads_the_runs_of_a_multi_channel_channel():
    class Gapped:
        id = "ch-0"
        name = "Channel 0"
        unit = "uV"

        def rate_hz(self):
            return 1000.0

        def start_us(self):
            return 10

        def num_samples(self):
            return 3

        def runs(self):
            return [Run(10, 1), Run(2_010, 2)]

    (channel,) = [Gapped()]
    assert isinstance(channel, ContinuousChannelMetadata)
    assert source_runs(channel) == [Run(10, 1), Run(2_010, 2)]
