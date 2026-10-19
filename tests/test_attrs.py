import dataclasses
import datetime as dt
import json
import re

import numpy as np
import pytest
import zarr

from timeseries_zarr.attrs import (
    ContinuousAttrs,
    CountLevelAttrs,
    EventAttrs,
    WaveformAttrs,
    continuous_channel_attrs,
    event_channel_attrs,
    level_group_attrs,
    parse_channel_kind,
    parse_continuous_attrs,
    parse_count_level_attrs,
    parse_event_attrs,
    parse_level_attrs,
    parse_version,
    parse_waveform_attrs,
    root_attrs,
    tsz_fields,
    waveform_array_attrs,
)
from timeseries_zarr.constants import FORMAT_VERSION
from timeseries_zarr.zarr_io import (
    consolidate,
    create_array,
    create_group_with_attrs,
    open_group,
)


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


def test_continuous_channel_attrs_keeps_a_negative_offset():
    attrs = continuous_channel_attrs("c", "i", 512.0, -32_000, "uV", None)
    assert attrs["tsz"]["offset_us"] == -32_000


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


@pytest.mark.parametrize("offset_us", [0, 16_000, -32_000])
def test_level_group_attrs_writes_a_count_level_offset_on_the_grid(offset_us):
    assert level_group_attrs(16_000.0, offset_us) == {
        "tsz": {"period_us": 16_000.0, "offset_us": offset_us}
    }


def test_level_group_attrs_rejects_an_offset_off_the_grid():
    with pytest.raises(ValueError, match="multiple of period_us"):
        level_group_attrs(16_000.0, -31_999)


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


def _stored(attributes):
    """Return attributes as a reader gets them back from zarr.json."""
    return json.loads(json.dumps(attributes))


def _invalid(where, key, expectation, value):
    return re.escape(f"{where}: tsz.{key} must be {expectation}, got {value!r}")


def _missing(where, key, expectation):
    return re.escape(f"{where}: tsz.{key} is missing; it must be {expectation}")


_CONTINUOUS = {
    "kind": "continuous",
    "name": "LA4",
    "rate_hz": 512.0,
    "unit": "uV",
}
_EVENT = {"kind": "event", "name": "marks"}
_HUGE = 10**400
_INT64_MAX = 2**63 - 1
_INTEGER = "an integer from 0 to 2^63 - 1"
_ANY_INTEGER = "an integer from -2^63 to 2^63 - 1"


def test_tsz_fields_returns_the_tsz_object():
    fields = {"version": "1.0", "extra": [1]}
    assert tsz_fields({"tsz": fields, "other": {}}, "root group") == fields


def test_tsz_fields_rejects_attributes_without_tsz():
    with pytest.raises(
        ValueError,
        match=re.escape("channel 0: tsz is missing; it must be an object"),
    ):
        tsz_fields({"other": {}}, "channel 0")


@pytest.mark.parametrize("value", [None, [], "1.0", 1, True])
def test_tsz_fields_rejects_a_tsz_value_that_is_not_an_object(value):
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"channel 0 level 2: tsz must be an object, got {value!r}"
        ),
    ):
        tsz_fields({"tsz": value}, "channel 0 level 2")


def test_parse_version_reads_back_the_version_the_writer_stores():
    assert parse_version(_stored(root_attrs())) == FORMAT_VERSION


@pytest.mark.parametrize("version", ["1.0", "1.7", "1.12", "01.0"])
def test_parse_version_accepts_any_minor_version_of_major_one(version):
    assert parse_version({"tsz": {"version": version}}) == version


def test_parse_version_ignores_unknown_keys():
    attributes = {"tsz": {"version": "1.3", "producer": "x"}, "ome": {}}
    assert parse_version(attributes) == "1.3"


@pytest.mark.parametrize(
    ("version", "major"), [("2.0", 2), ("0.9", 0), ("10.1", 10)]
)
def test_parse_version_rejects_another_major_version(version, major):
    message = (
        f"bundle version {version!r} has major version {major}; "
        "this package reads major version 1"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        parse_version({"tsz": {"version": version}})


@pytest.mark.parametrize(
    "version",
    [
        "1",
        "1.0.1",
        "v1.0",
        "1.0\n",
        " 1.0",
        "1.x",
        "1.",
        "",
        "\N{ARABIC-INDIC DIGIT ONE}.0",
        1.0,
        1,
        True,
        None,
        [1, 0],
    ],
)
def test_parse_version_rejects_a_version_not_of_the_form_major_minor(version):
    with pytest.raises(
        ValueError,
        match=_invalid(
            "root group", "version", "a string of the form major.minor", version
        ),
    ):
        parse_version({"tsz": {"version": version}})


def test_parse_version_rejects_a_root_group_without_a_version():
    with pytest.raises(
        ValueError,
        match=_missing(
            "root group", "version", "a string of the form major.minor"
        ),
    ):
        parse_version({"tsz": {}})


def test_parse_version_rejects_a_root_group_without_tsz():
    with pytest.raises(ValueError, match="root group: tsz is missing"):
        parse_version({})


@pytest.mark.parametrize("kind", ["continuous", "event"])
def test_parse_channel_kind_returns_a_known_kind(kind):
    assert parse_channel_kind({"tsz": {"kind": kind}}, "channel 0") == kind


@pytest.mark.parametrize("kind", ["spectrogram", "Event", ""])
def test_parse_channel_kind_returns_none_for_an_unknown_kind(kind):
    assert parse_channel_kind({"tsz": {"kind": kind}}, "channel 0") is None


def test_parse_channel_kind_rejects_a_group_without_a_kind():
    with pytest.raises(
        ValueError, match=_missing("channel 4", "kind", "a string")
    ):
        parse_channel_kind({"tsz": {"name": "marks"}}, "channel 4")


@pytest.mark.parametrize("kind", [1, None, True, ["event"], {"kind": "event"}])
def test_parse_channel_kind_rejects_a_kind_that_is_not_a_string(kind):
    with pytest.raises(
        ValueError, match=_invalid("channel 4", "kind", "a string", kind)
    ):
        parse_channel_kind({"tsz": {"kind": kind}}, "channel 4")


def test_parse_channel_kind_rejects_a_group_without_tsz():
    with pytest.raises(ValueError, match="channel 4: tsz is missing"):
        parse_channel_kind({}, "channel 4")


@pytest.mark.parametrize(
    "provenance", [None, {"software": "acquire 4.1", "bands_hz": [80, 250]}]
)
def test_parse_continuous_attrs_reads_back_what_the_builder_writes(provenance):
    attributes = continuous_channel_attrs(
        "LA4", "N:channel:7f3a", 2000.0, 500_000, "uV", provenance
    )
    assert parse_continuous_attrs(_stored(attributes), "channel 0") == (
        ContinuousAttrs(
            name="LA4",
            id="N:channel:7f3a",
            unit="uV",
            rate_hz=2000.0,
            offset_us=500_000,
            provenance=provenance,
        )
    )


def test_parse_continuous_attrs_defaults_the_optional_keys():
    parsed = parse_continuous_attrs({"tsz": _CONTINUOUS}, "channel 0")
    assert parsed == ContinuousAttrs(
        name="LA4",
        id=None,
        unit="uV",
        rate_hz=512.0,
        offset_us=0,
        provenance=None,
    )


def test_parse_continuous_attrs_returns_an_integer_rate_as_a_float():
    parsed = parse_continuous_attrs(
        {"tsz": {**_CONTINUOUS, "rate_hz": 30000}}, "channel 0"
    )
    assert parsed.rate_hz == 30000.0
    assert type(parsed.rate_hz) is float


def test_parse_continuous_attrs_accepts_an_integral_float_offset():
    parsed = parse_continuous_attrs(
        {"tsz": {**_CONTINUOUS, "offset_us": 500000.0}}, "channel 0"
    )
    assert parsed.offset_us == 500_000
    assert type(parsed.offset_us) is int


def test_parse_continuous_attrs_keeps_an_empty_provenance_object():
    parsed = parse_continuous_attrs(
        {"tsz": {**_CONTINUOUS, "provenance": {}}}, "channel 0"
    )
    assert parsed.provenance == {}


def test_parse_continuous_attrs_ignores_unknown_keys():
    attributes = {
        "tsz": {**_CONTINUOUS, "label_names": 7, "montage": None},
        "other": [1],
    }
    assert parse_continuous_attrs(attributes, "channel 0") == (
        parse_continuous_attrs({"tsz": _CONTINUOUS}, "channel 0")
    )


@pytest.mark.parametrize(
    ("key", "expectation"),
    [
        ("name", "a string"),
        ("unit", "a string"),
        ("rate_hz", "a finite number above 0"),
    ],
)
def test_parse_continuous_attrs_rejects_a_missing_required_key(
    key, expectation
):
    fields = {name: value for name, value in _CONTINUOUS.items() if name != key}
    with pytest.raises(
        ValueError, match=_missing("channel 0", key, expectation)
    ):
        parse_continuous_attrs({"tsz": fields}, "channel 0")


@pytest.mark.parametrize(
    ("key", "value", "expectation"),
    [
        ("name", 4, "a string"),
        ("name", None, "a string"),
        ("unit", ["uV"], "a string"),
        ("id", 7, "a string"),
        ("id", None, "a string"),
        ("rate_hz", 0, "a finite number above 0"),
        ("rate_hz", -512.0, "a finite number above 0"),
        ("rate_hz", float("nan"), "a finite number above 0"),
        ("rate_hz", float("inf"), "a finite number above 0"),
        ("rate_hz", _HUGE, "a finite number above 0"),
        ("rate_hz", "512", "a finite number above 0"),
        ("rate_hz", True, "a finite number above 0"),
        ("rate_hz", None, "a finite number above 0"),
        ("offset_us", 1.5, _ANY_INTEGER),
        ("offset_us", float("nan"), _ANY_INTEGER),
        ("offset_us", float("inf"), _ANY_INTEGER),
        ("offset_us", "0", _ANY_INTEGER),
        ("offset_us", True, _ANY_INTEGER),
        ("offset_us", False, _ANY_INTEGER),
        ("offset_us", None, _ANY_INTEGER),
        ("offset_us", _INT64_MAX + 1, _ANY_INTEGER),
        ("offset_us", -_INT64_MAX - 2, _ANY_INTEGER),
        ("offset_us", float(_INT64_MAX + 1), _ANY_INTEGER),
        ("offset_us", _HUGE, _ANY_INTEGER),
        ("provenance", [1, 2], "an object"),
        ("provenance", "acquire 4.1", "an object"),
        ("provenance", True, "an object"),
    ],
)
def test_parse_continuous_attrs_rejects_an_invalid_value(
    key, value, expectation
):
    attributes = {"tsz": {**_CONTINUOUS, key: value}}
    with pytest.raises(
        ValueError, match=_invalid("channel 0", key, expectation, value)
    ):
        parse_continuous_attrs(attributes, "channel 0")


def test_parse_continuous_attrs_rejects_a_group_without_tsz():
    with pytest.raises(ValueError, match="channel 0: tsz is missing"):
        parse_continuous_attrs({}, "channel 0")


def test_parse_continuous_attrs_accepts_the_largest_int64_offset():
    parsed = parse_continuous_attrs(
        {"tsz": {**_CONTINUOUS, "offset_us": _INT64_MAX}}, "channel 0"
    )
    assert parsed.offset_us == _INT64_MAX


def test_parse_continuous_attrs_accepts_a_negative_offset():
    parsed = parse_continuous_attrs(
        {"tsz": {**_CONTINUOUS, "offset_us": -32_000}}, "channel 0"
    )
    assert parsed.offset_us == -32_000
    lowest = parse_continuous_attrs(
        {"tsz": {**_CONTINUOUS, "offset_us": -_INT64_MAX - 1}}, "channel 0"
    )
    assert lowest.offset_us == -_INT64_MAX - 1


def test_parse_continuous_attrs_rejects_an_offset_below_int64():
    with pytest.raises(
        ValueError,
        match=r"channel 0: tsz.offset_us must be an integer from -2\^63 to",
    ):
        parse_continuous_attrs(
            {"tsz": {**_CONTINUOUS, "offset_us": -_INT64_MAX - 2}}, "channel 0"
        )


@pytest.mark.parametrize("offset_us", [0, 16_000, -32_000])
def test_parse_count_level_attrs_reads_back_what_the_builder_writes(offset_us):
    attributes = _stored(level_group_attrs(16_000.0, offset_us))
    parsed = parse_count_level_attrs(attributes, "channel 4 count level 1")
    assert parsed == CountLevelAttrs(period_us=16_000.0, offset_us=offset_us)


def test_parse_count_level_attrs_defaults_the_offset_to_0():
    parsed = parse_count_level_attrs(
        {"tsz": {"period_us": 1000}}, "channel 4 count level 1"
    )
    assert parsed == CountLevelAttrs(period_us=1000.0, offset_us=0)
    assert type(parsed.period_us) is float


def test_parse_count_level_attrs_rejects_an_offset_off_the_grid():
    with pytest.raises(
        ValueError,
        match=(
            "^channel 4 count level 1: tsz.offset_us must be a multiple of "
            "period_us 1000.0, got 500$"
        ),
    ):
        parse_count_level_attrs(
            {"tsz": {"period_us": 1000, "offset_us": 500}},
            "channel 4 count level 1",
        )


def test_parse_count_level_attrs_rejects_an_offset_that_is_not_an_integer():
    with pytest.raises(ValueError, match="offset_us must be an integer"):
        parse_count_level_attrs(
            {"tsz": {"period_us": 1000, "offset_us": "0"}},
            "channel 4 count level 1",
        )


def test_parse_event_attrs_reads_back_a_minimal_group():
    attributes = event_channel_attrs("marks", None, None, None, None, None)
    assert parse_event_attrs(_stored(attributes), "channel 4") == EventAttrs(
        name="marks",
        id=None,
        unit=None,
        label_names=None,
        max_duration_us=None,
        provenance=None,
    )


def test_parse_event_attrs_reads_back_every_key_the_builder_writes():
    attributes = event_channel_attrs(
        "seizure marks",
        "N:ann:1",
        "a.u.",
        ["onset", "offset"],
        600_000_000,
        {"software": "review-tool 3.2"},
    )
    assert parse_event_attrs(_stored(attributes), "channel 4") == EventAttrs(
        name="seizure marks",
        id="N:ann:1",
        unit="a.u.",
        label_names=("onset", "offset"),
        max_duration_us=600_000_000,
        provenance={"software": "review-tool 3.2"},
    )


def test_parse_event_attrs_stores_label_names_as_a_tuple():
    parsed = parse_event_attrs(
        {"tsz": {**_EVENT, "label_names": ["a", "b", "a"]}}, "channel 4"
    )
    assert parsed.label_names == ("a", "b", "a")


def test_parse_event_attrs_keeps_a_zero_duration_bound():
    parsed = parse_event_attrs(
        {"tsz": {**_EVENT, "max_duration_us": 0}}, "channel 4"
    )
    assert parsed.max_duration_us == 0


def test_parse_event_attrs_accepts_an_integral_float_duration_bound():
    parsed = parse_event_attrs(
        {"tsz": {**_EVENT, "max_duration_us": 600000000.0}}, "channel 4"
    )
    assert parsed.max_duration_us == 600_000_000
    assert type(parsed.max_duration_us) is int


def test_parse_event_attrs_ignores_unknown_keys():
    attributes = {"tsz": {**_EVENT, "rate_hz": "fast", "offset_us": -1}, "x": 1}
    assert parse_event_attrs(attributes, "channel 4") == (
        parse_event_attrs({"tsz": _EVENT}, "channel 4")
    )


def test_parse_event_attrs_rejects_a_group_without_a_name():
    with pytest.raises(
        ValueError, match=_missing("channel 4", "name", "a string")
    ):
        parse_event_attrs({"tsz": {"kind": "event"}}, "channel 4")


@pytest.mark.parametrize(
    ("key", "value", "expectation"),
    [
        ("name", 4, "a string"),
        ("id", 7, "a string"),
        ("unit", None, "a string"),
        ("unit", 1.0, "a string"),
        ("label_names", [], "a non-empty list of strings"),
        ("label_names", "onset", "a non-empty list of strings"),
        ("label_names", [1, 2], "a non-empty list of strings"),
        ("label_names", ["onset", None], "a non-empty list of strings"),
        ("label_names", {"0": "onset"}, "a non-empty list of strings"),
        ("label_names", True, "a non-empty list of strings"),
        ("max_duration_us", -1, _INTEGER),
        ("max_duration_us", 2.5, _INTEGER),
        ("max_duration_us", "10", _INTEGER),
        ("max_duration_us", True, _INTEGER),
        ("max_duration_us", None, _INTEGER),
        ("max_duration_us", _INT64_MAX + 1, _INTEGER),
        ("provenance", [1], "an object"),
        ("provenance", None, "an object"),
    ],
)
def test_parse_event_attrs_rejects_an_invalid_value(key, value, expectation):
    attributes = {"tsz": {**_EVENT, key: value}}
    with pytest.raises(
        ValueError, match=_invalid("channel 4", key, expectation, value)
    ):
        parse_event_attrs(attributes, "channel 4")


def test_parse_event_attrs_rejects_a_group_without_tsz():
    with pytest.raises(ValueError, match="channel 4: tsz is missing"):
        parse_event_attrs({"kind": "event"}, "channel 4")


@pytest.mark.parametrize("period_us", [31.25, 125.0, 512000.0, 7812.5])
def test_parse_level_attrs_reads_back_what_the_builder_writes(period_us):
    attributes = _stored(level_group_attrs(period_us))
    assert parse_level_attrs(attributes, "channel 0 level 2") == period_us


def test_parse_level_attrs_returns_an_integer_period_as_a_float():
    period_us = parse_level_attrs(
        {"tsz": {"period_us": 1000}}, "channel 4 level 1"
    )
    assert period_us == 1000.0
    assert type(period_us) is float


def test_parse_level_attrs_ignores_unknown_keys():
    attributes = {"tsz": {"period_us": 125.0, "bins": "many"}, "x": None}
    assert parse_level_attrs(attributes, "channel 0 level 2") == 125.0


def test_parse_level_attrs_rejects_a_group_without_a_period():
    with pytest.raises(
        ValueError,
        match=_missing(
            "channel 0 level 2", "period_us", "a finite number above 0"
        ),
    ):
        parse_level_attrs({"tsz": {}}, "channel 0 level 2")


@pytest.mark.parametrize(
    "period_us",
    [
        0,
        0.0,
        -31.25,
        float("nan"),
        float("inf"),
        float("-inf"),
        _HUGE,
        True,
        "125",
        None,
        [125.0],
    ],
)
def test_parse_level_attrs_rejects_an_invalid_period(period_us):
    with pytest.raises(
        ValueError,
        match=_invalid(
            "channel 0 level 2",
            "period_us",
            "a finite number above 0",
            period_us,
        ),
    ):
        parse_level_attrs(
            {"tsz": {"period_us": period_us}}, "channel 0 level 2"
        )


def test_parse_level_attrs_rejects_a_group_without_tsz():
    with pytest.raises(ValueError, match="channel 0 level 2: tsz is missing"):
        parse_level_attrs({}, "channel 0 level 2")


@pytest.mark.parametrize(
    ("event_sample", "expected"), [(None, 0), (0, 0), (16, 16)]
)
def test_parse_waveform_attrs_reads_back_what_the_builder_writes(
    event_sample, expected
):
    attributes = _stored(waveform_array_attrs(122.0703125, event_sample))
    assert parse_waveform_attrs(attributes, "channel 3 waveforms") == (
        WaveformAttrs(period_us=122.0703125, event_sample=expected)
    )


def test_parse_waveform_attrs_defaults_event_sample_to_zero():
    parsed = parse_waveform_attrs(
        {"tsz": {"period_us": 31.25}}, "channel 3 waveforms"
    )
    assert parsed.event_sample == 0


def test_parse_waveform_attrs_accepts_an_integral_float_event_sample():
    parsed = parse_waveform_attrs(
        {"tsz": {"period_us": 31.25, "event_sample": 16.0}},
        "channel 3 waveforms",
    )
    assert parsed.event_sample == 16
    assert type(parsed.event_sample) is int


def test_parse_waveform_attrs_ignores_unknown_keys():
    attributes = {"tsz": {"period_us": 31.25, "event_sample": 3, "kind": 9}}
    assert parse_waveform_attrs(attributes, "channel 3 waveforms") == (
        WaveformAttrs(period_us=31.25, event_sample=3)
    )


def test_parse_waveform_attrs_rejects_an_array_without_a_period():
    with pytest.raises(
        ValueError,
        match=_missing(
            "channel 3 waveforms", "period_us", "a finite number above 0"
        ),
    ):
        parse_waveform_attrs(
            {"tsz": {"event_sample": 3}}, "channel 3 waveforms"
        )


@pytest.mark.parametrize(
    ("key", "value", "expectation"),
    [
        ("period_us", 0.0, "a finite number above 0"),
        ("period_us", -31.25, "a finite number above 0"),
        ("period_us", float("nan"), "a finite number above 0"),
        ("period_us", False, "a finite number above 0"),
        ("event_sample", -1, _INTEGER),
        ("event_sample", 1.5, _INTEGER),
        ("event_sample", True, _INTEGER),
        ("event_sample", "3", _INTEGER),
        ("event_sample", None, _INTEGER),
        ("event_sample", _INT64_MAX + 1, _INTEGER),
    ],
)
def test_parse_waveform_attrs_rejects_an_invalid_value(key, value, expectation):
    attributes = {"tsz": {"period_us": 31.25, "event_sample": 3, key: value}}
    with pytest.raises(
        ValueError,
        match=_invalid("channel 3 waveforms", key, expectation, value),
    ):
        parse_waveform_attrs(attributes, "channel 3 waveforms")


def test_parse_waveform_attrs_rejects_an_array_without_tsz():
    with pytest.raises(ValueError, match="channel 3 waveforms: tsz is missing"):
        parse_waveform_attrs({}, "channel 3 waveforms")


@pytest.mark.parametrize(
    ("record", "field"),
    [
        (ContinuousAttrs("LA4", None, "uV", 512.0, 0, None), "rate_hz"),
        (EventAttrs("marks", None, None, None, None, None), "name"),
        (WaveformAttrs(31.25, 0), "event_sample"),
    ],
    ids=["continuous", "event", "waveform"],
)
def test_parsed_records_reject_assignment(record, field):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(record, field, 1)
    assert not hasattr(record, "__dict__")


def test_parsers_read_the_attributes_of_a_reopened_bundle(tmp_path):
    root = open_group(tmp_path, root_attrs())
    continuous = create_group_with_attrs(
        root,
        "0",
        continuous_channel_attrs(
            "LA4", "N:channel:7f3a", 512.0, 500_000, "uV", {"software": "a 1"}
        ),
    )
    create_group_with_attrs(continuous, "1", level_group_attrs(7812.5))
    event = create_group_with_attrs(
        root,
        "3",
        event_channel_attrs("units", None, "uV", ["a", "b"], None, None),
    )
    create_array(
        event,
        "waveforms",
        (4, 8),
        np.float32,
        (4, 8),
        (4, 8),
        waveform_array_attrs(31.25, 3),
        1,
    )
    consolidate(root)

    reopened = zarr.open_group(
        str(tmp_path), mode="r", zarr_format=3, use_consolidated=True
    )

    assert parse_version(reopened.attrs) == FORMAT_VERSION
    assert parse_channel_kind(reopened["0"].attrs, "channel 0") == "continuous"
    assert parse_continuous_attrs(reopened["0"].attrs, "channel 0") == (
        ContinuousAttrs(
            "LA4", "N:channel:7f3a", "uV", 512.0, 500_000, {"software": "a 1"}
        )
    )
    assert (
        parse_level_attrs(reopened["0/1"].attrs, "channel 0 level 1") == 7812.5
    )
    assert parse_channel_kind(reopened["3"].attrs, "channel 3") == "event"
    assert parse_event_attrs(reopened["3"].attrs, "channel 3") == EventAttrs(
        "units", None, "uV", ("a", "b"), None, None
    )
    assert parse_waveform_attrs(
        reopened["3/waveforms"].attrs, "channel 3 waveforms"
    ) == WaveformAttrs(31.25, 3)
