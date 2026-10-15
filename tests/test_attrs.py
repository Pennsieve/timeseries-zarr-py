import datetime as dt

import numpy as np
import pytest

from timeseries_zarr.attrs import (
    continuous_channel_attrs,
    event_channel_attrs,
    level_group_attrs,
    root_attrs,
    waveform_array_attrs,
)
from timeseries_zarr.constants import FORMAT_VERSION


def test_root_attrs_carries_the_format_version_under_tsz():
    assert root_attrs() == {"tsz": {"version": FORMAT_VERSION}}


def test_continuous_channel_attrs_nests_every_key_under_tsz():
    attrs = continuous_channel_attrs(
        "LA4", "N:channel:7f3a", 2000.0, 0, "uV", None
    )
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
    attrs = continuous_channel_attrs("c", "i", 512.0, 500_000, "uV", None)
    assert attrs["tsz"]["offset_us"] == 500_000


@pytest.mark.parametrize("rate_hz", [0.0, -1.0])
def test_continuous_channel_attrs_rejects_a_non_positive_rate(rate_hz):
    with pytest.raises(ValueError, match="rate_hz"):
        continuous_channel_attrs("c", "i", rate_hz, 0, "uV", None)


def test_continuous_channel_attrs_rejects_a_negative_offset():
    with pytest.raises(ValueError, match="offset_us"):
        continuous_channel_attrs("c", "i", 512.0, -1, "uV", None)


def test_continuous_channel_attrs_writes_the_given_provenance_last():
    attrs = continuous_channel_attrs(
        "LA4", "i", 2000.0, 0, "uV", {"software": "acquire 4.1"}
    )
    assert attrs == {
        "tsz": {
            "kind": "continuous",
            "name": "LA4",
            "id": "i",
            "rate_hz": 2000.0,
            "offset_us": 0,
            "unit": "uV",
            "provenance": {"software": "acquire 4.1"},
        }
    }
    assert list(attrs["tsz"])[-1] == "provenance"


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


def _continuous(provenance):
    return continuous_channel_attrs("c", "i", 512.0, 0, "uV", provenance)


def _event(provenance):
    return event_channel_attrs("m", None, None, None, None, provenance)


_BUILDERS = pytest.mark.parametrize(
    "build", [_continuous, _event], ids=["continuous", "event"]
)


@_BUILDERS
def test_channel_attrs_omit_provenance_when_it_is_none(build):
    assert "provenance" not in build(None)["tsz"]


@_BUILDERS
def test_channel_attrs_keep_a_nested_provenance_object(build):
    provenance = {
        "software": "detector 2.4",
        "parameters": {"threshold": 3.5, "bands_hz": [80, 250], "causal": True},
        "notes": None,
        "label": "caf\N{LATIN SMALL LETTER E WITH ACUTE}",
    }
    assert build(provenance)["tsz"]["provenance"] == provenance


@_BUILDERS
def test_channel_attrs_keep_an_empty_provenance_object(build):
    assert build({})["tsz"]["provenance"] == {}


@_BUILDERS
@pytest.mark.parametrize("provenance", [[1, 2], "detector 2.4", 3])
def test_channel_attrs_reject_a_provenance_that_is_not_an_object(
    build, provenance
):
    with pytest.raises(ValueError, match="provenance must be a dict"):
        build(provenance)


@_BUILDERS
@pytest.mark.parametrize(
    "provenance",
    [
        {"threshold": np.float32(3.5)},
        {"count": np.int64(3)},
        {"when": dt.datetime(2024, 1, 1)},
        {"bands": {80, 250}},
    ],
    ids=["numpy-float", "numpy-int", "datetime", "set"],
)
def test_channel_attrs_reject_a_provenance_value_json_cannot_store(
    build, provenance
):
    with pytest.raises(ValueError, match="provenance must hold only JSON"):
        build(provenance)


@_BUILDERS
@pytest.mark.parametrize(
    "provenance",
    [
        {"threshold": float("nan")},
        {"threshold": float("inf")},
        {"threshold": float("-inf")},
        {"bands": [80.0, {"high": float("nan")}]},
    ],
    ids=["nan", "inf", "negative-inf", "nested-nan"],
)
def test_channel_attrs_reject_a_non_finite_provenance_number(build, provenance):
    with pytest.raises(ValueError, match="finite"):
        build(provenance)


@_BUILDERS
def test_channel_attrs_reject_a_self_referencing_provenance(build):
    provenance = {}
    provenance["self"] = provenance
    with pytest.raises(ValueError, match="provenance must hold only JSON"):
        build(provenance)
