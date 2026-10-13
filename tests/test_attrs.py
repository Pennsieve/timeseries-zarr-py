import pytest

from timeseries_zarr.attrs import (
    channel_group_attrs,
    continuous_channel_attrs,
    event_channel_attrs,
    level_array_attrs,
    level_group_attrs,
    root_attrs,
    waveform_array_attrs,
)
from timeseries_zarr.constants import FORMAT_VERSION


def test_root_attrs_carries_the_format_version_under_tsz():
    assert root_attrs() == {"tsz": {"version": FORMAT_VERSION}}


def test_continuous_channel_attrs_nests_every_key_under_tsz():
    attrs = continuous_channel_attrs("LA4", "N:channel:7f3a", 2000.0, 0, "uV")
    assert attrs == {
        "tsz": {
            "kind": "continuous",
            "name": "LA4",
            "id": "N:channel:7f3a",
            "rate_hz": 2000.0,
            "offset_us": 0,
            "unit": "uV",
        }
    }


def test_continuous_channel_attrs_keeps_a_positive_offset():
    attrs = continuous_channel_attrs("c", "i", 512.0, 500_000, "uV")
    assert attrs["tsz"]["offset_us"] == 500_000


@pytest.mark.parametrize("rate_hz", [0.0, -1.0])
def test_continuous_channel_attrs_rejects_a_non_positive_rate(rate_hz):
    with pytest.raises(ValueError, match="rate_hz"):
        continuous_channel_attrs("c", "i", rate_hz, 0, "uV")


def test_continuous_channel_attrs_rejects_a_negative_offset():
    with pytest.raises(ValueError, match="offset_us"):
        continuous_channel_attrs("c", "i", 512.0, -1, "uV")


def test_event_channel_attrs_minimal_holds_kind_and_name_only():
    attrs = event_channel_attrs("marks", None, None, None, None, None)
    assert attrs == {"tsz": {"kind": "event", "name": "marks"}}


def test_event_channel_attrs_writes_every_given_key():
    attrs = event_channel_attrs(
        "seizure marks",
        "N:ann:1",
        "a.u.",
        ["onset", "offset"],
        600_000_000,
        {"software": "review-tool 3.2"},
    )
    assert attrs == {
        "tsz": {
            "kind": "event",
            "name": "seizure marks",
            "id": "N:ann:1",
            "unit": "a.u.",
            "label_names": ["onset", "offset"],
            "max_duration_us": 600_000_000,
            "provenance": {"software": "review-tool 3.2"},
        }
    }


def test_event_channel_attrs_keeps_a_zero_duration_bound():
    attrs = event_channel_attrs("m", None, None, None, 0, None)
    assert attrs["tsz"]["max_duration_us"] == 0


def test_event_channel_attrs_rejects_empty_label_names():
    with pytest.raises(ValueError, match="label_names"):
        event_channel_attrs("m", None, None, [], None, None)


def test_event_channel_attrs_rejects_a_negative_duration_bound():
    with pytest.raises(ValueError, match="max_duration_us"):
        event_channel_attrs("m", None, None, None, -1, None)


@pytest.mark.parametrize("period_us", [31.25, 125.0, 512000.0])
def test_level_group_attrs_nests_period_under_tsz(period_us):
    assert level_group_attrs(period_us) == {"tsz": {"period_us": period_us}}


@pytest.mark.parametrize("period_us", [0.0, -31.25])
def test_level_group_attrs_rejects_a_non_positive_period(period_us):
    with pytest.raises(ValueError, match="period_us"):
        level_group_attrs(period_us)


@pytest.mark.parametrize("period_us", [31.25, 125.0])
def test_waveform_array_attrs_without_event_sample(period_us):
    assert waveform_array_attrs(period_us) == {"tsz": {"period_us": period_us}}


def test_waveform_array_attrs_with_event_sample():
    assert waveform_array_attrs(122.0703125, 16) == {
        "tsz": {"period_us": 122.0703125, "event_sample": 16}
    }


def test_waveform_array_attrs_keeps_event_sample_zero():
    assert waveform_array_attrs(31.25, 0)["tsz"]["event_sample"] == 0


def test_waveform_array_attrs_rejects_a_non_positive_period():
    with pytest.raises(ValueError, match="period_us"):
        waveform_array_attrs(0.0)


def test_waveform_array_attrs_rejects_a_negative_event_sample():
    with pytest.raises(ValueError, match="event_sample"):
        waveform_array_attrs(31.25, -1)


@pytest.mark.parametrize("kind", ["continuous", "unit"])
def test_channel_group_attrs(kind):
    attrs = channel_group_attrs(
        "N:channel:abc", 32000.0, 1000, kind, "channel one", "uV"
    )
    assert attrs == {
        "id": "N:channel:abc",
        "rate_hz": 32000.0,
        "start_us": 1000,
        "kind": kind,
        "name": "channel one",
        "unit": "uV",
    }


@pytest.mark.parametrize("period_us", [31.25, 125.0, 512000.0])
def test_level_array_attrs(period_us):
    assert level_array_attrs(period_us) == {"period_us": period_us}
