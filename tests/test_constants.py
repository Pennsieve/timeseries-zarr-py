import numpy as np
import pytest

from timeseries_zarr.constants import (
    CANVAS_PIXELS,
    CHANNEL_KINDS,
    CHANNELS_KEY,
    COUNT_KEY,
    COUNT_LADDER_BASE_US,
    DEFAULT_DECIMATION_FACTOR,
    DURATIONS_KEY,
    EVENT_COUNT_LIMIT,
    EVENT_DIMENSION,
    EVENT_LEVEL_THRESHOLD,
    EVENTS_KEY,
    EVENTS_SEARCH_LIMIT_BYTES,
    FLOAT32_BYTES,
    FORMAT_VERSION,
    INT64_BYTES,
    LABEL_DIMENSION,
    LABELS_KEY,
    LEVEL_PERIOD_TOLERANCE,
    MAJOR_VERSION,
    MAX_LABELS,
    MEAN_KEY,
    MIN_DECIMATION_FACTOR,
    MINMAX_KEY,
    RASTER_BUDGET,
    RAW_KEY,
    RUN_DIMENSION,
    RUNS_COLUMNS,
    RUNS_KEY,
    SAMPLES_PER_BIN_LIMIT,
    SHARD_INDEX_CACHE_BYTES,
    TEXT_KEY,
    TIME_DIMENSION,
    TIME_LIMIT_US,
    UINT8_BYTES,
    UINT16_BYTES,
    UINT32_BYTES,
    UNIT_TO_UV,
    VALUES_KEY,
    VAR_KEY,
    WAVEFORM_MEAN_KEY,
    WAVEFORMS_KEY,
    integer_like,
    level_factors,
    level_widths,
    max_levels_for,
    normalize_factors,
)


def test_format_version_is_the_1_0_draft():
    assert FORMAT_VERSION == "1.0"


def test_major_version_is_the_major_part_of_the_format_version():
    assert MAJOR_VERSION == 1
    assert int(FORMAT_VERSION.split(".")[0]) == MAJOR_VERSION


def test_channel_kinds_are_continuous_and_event():
    assert CHANNEL_KINDS == ("continuous", "event")


def test_continuous_array_keys_follow_the_spec():
    assert RAW_KEY == "raw"
    assert MINMAX_KEY == "minmax"
    assert MEAN_KEY == "mean"
    assert VAR_KEY == "var"
    assert COUNT_KEY == "count"


def test_event_array_keys_follow_the_spec():
    assert EVENTS_KEY == "events"
    assert DURATIONS_KEY == "durations"
    assert LABELS_KEY == "labels"
    assert VALUES_KEY == "values"
    assert TEXT_KEY == "text"
    assert CHANNELS_KEY == "channels"
    assert WAVEFORMS_KEY == "waveforms"
    assert WAVEFORM_MEAN_KEY == "waveform_mean"


def test_dimension_names_follow_the_spec():
    assert TIME_DIMENSION == "time"
    assert EVENT_DIMENSION == "event"
    assert LABEL_DIMENSION == "label"
    assert RUN_DIMENSION == "run"


def test_runs_key_follows_the_spec():
    assert RUNS_KEY == "runs"


def test_runs_array_has_three_columns():
    assert RUNS_COLUMNS == 3


def test_events_search_limit_is_one_mebibyte_of_int64_events():
    assert EVENTS_SEARCH_LIMIT_BYTES == 2**20
    assert EVENTS_SEARCH_LIMIT_BYTES // INT64_BYTES == 131_072


def test_level_period_tolerance_is_one_part_in_a_trillion():
    assert LEVEL_PERIOD_TOLERANCE == 1e-12


def test_time_limit_is_2_to_the_53_microseconds():
    assert TIME_LIMIT_US == 2**53


def test_event_count_and_level_width_limits_are_2_to_the_32():
    assert EVENT_COUNT_LIMIT == 2**32
    assert SAMPLES_PER_BIN_LIMIT == 2**32


def test_default_decimation_factor_is_4_and_the_minimum_is_2():
    assert DEFAULT_DECIMATION_FACTOR == 4
    assert MIN_DECIMATION_FACTOR == 2


def test_shard_index_cache_budget_is_64_mebibytes():
    assert SHARD_INDEX_CACHE_BYTES == 64 * 1024 * 1024


def test_max_labels_is_the_uint16_range():
    assert np.iinfo(np.uint16).max + 1 == MAX_LABELS


def test_byte_widths_match_numpy_itemsizes():
    assert np.dtype(np.float32).itemsize == FLOAT32_BYTES
    assert np.dtype(np.int64).itemsize == INT64_BYTES
    assert np.dtype(np.uint8).itemsize == UINT8_BYTES
    assert np.dtype(np.uint16).itemsize == UINT16_BYTES
    assert np.dtype(np.uint32).itemsize == UINT32_BYTES


def test_count_level_constants_follow_the_spec():
    assert EVENT_LEVEL_THRESHOLD == 64_000
    assert RASTER_BUDGET == 10_000
    assert CANVAS_PIXELS == 2000
    assert COUNT_LADDER_BASE_US == 1000


def test_unit_to_uv_pins_the_mapping():
    assert UNIT_TO_UV == {
        "V": 1e6,
        "mV": 1e3,
        "uV": 1.0,
        "nV": 1e-3,
        "volts": 1e6,
        "millivolts": 1e3,
        "microvolts": 1.0,
        "nanovolts": 1e-3,
    }


def test_unit_to_uv_spelled_names_agree_with_symbols():
    assert UNIT_TO_UV["V"] == UNIT_TO_UV["volts"]
    assert UNIT_TO_UV["mV"] == UNIT_TO_UV["millivolts"]
    assert UNIT_TO_UV["uV"] == UNIT_TO_UV["microvolts"]
    assert UNIT_TO_UV["nV"] == UNIT_TO_UV["nanovolts"]


def test_unit_to_uv_has_no_megavolt_key():
    assert "MV" not in UNIT_TO_UV
    assert "Mv" not in UNIT_TO_UV


def test_max_levels_for_an_int_is_the_cap_of_a_one_element_ladder():
    assert max_levels_for(4) == 15
    for factor in (2, 3, 10, 1000):
        assert max_levels_for(factor) == max_levels_for((factor,))


def test_max_levels_for_counts_the_levels_of_a_ladder_that_fit_uint32():
    assert max_levels_for((8, 4)) == 15
    assert max_levels_for((3, 5)) == 14
    assert max_levels_for((2**16, 2**15)) == 2


def test_max_levels_for_stops_before_a_ladder_overflows_2_to_the_32():
    assert max_levels_for((2**16, 2**16, 2)) == 1
    assert max_levels_for((2**32,)) == 0


@pytest.mark.parametrize("factors", [(1, 4), (4, 1), [0], -4, 1])
def test_max_levels_for_rejects_a_ratio_below_2(factors):
    with pytest.raises(ValueError, match="decimation_factor"):
        max_levels_for(factors)


def test_normalize_factors_returns_an_int_as_a_one_element_tuple():
    assert normalize_factors(4, "decimation_factor") == (4,)


@pytest.mark.parametrize("value", [[8, 4], (8, 4), range(8, 3, -4)])
def test_normalize_factors_returns_a_sequence_as_a_tuple(value):
    assert normalize_factors(value, "decimation_factor") == (8, 4)


def test_normalize_factors_is_idempotent():
    once = normalize_factors([8, 4], "decimation_factor")
    assert normalize_factors(once, "decimation_factor") == once


def test_normalize_factors_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="my_field must hold at least one"):
        normalize_factors([], "my_field")


@pytest.mark.parametrize("value", [1, 0, -2, (4, 1), [1], (2, 4, 0)])
def test_normalize_factors_rejects_a_ratio_below_2(value):
    with pytest.raises(ValueError, match="my_field must .*2 or more"):
        normalize_factors(value, "my_field")


@pytest.mark.parametrize(
    "value",
    ["48", b"48", True, 4.0, None, [4.0], [True, 4], ["4"], [[4]], [(8, 4)]],
)
def test_normalize_factors_rejects_a_value_that_is_not_an_int_or_ints(value):
    with pytest.raises(TypeError, match="my_field must"):
        normalize_factors(value, "my_field")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.int64(4), (4,)),
        (np.int32(4), (4,)),
        ((np.int64(8), np.int32(4)), (8, 4)),
        ([np.int64(8), 4, np.uint8(2)], (8, 4, 2)),
        ((8, 4), (8, 4)),
    ],
)
def test_normalize_factors_returns_plain_ints_for_integer_like_values(
    value, expected
):
    ratios = normalize_factors(value, "my_field")
    assert ratios == expected
    assert all(type(ratio) is int for ratio in ratios)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        4.0,
        np.float64(4.0),
        "4",
        b"4",
        bytearray(b"\x08\x04"),
        memoryview(b"\x08\x04"),
        [[4]],
        [(8, 4)],
        [np.float64(4.0)],
        [np.bool_(True)],
        np.array([8, 4]),
        np.array(4),
    ],
)
def test_normalize_factors_rejects_values_that_are_not_integer_like(value):
    with pytest.raises(TypeError, match="my_field must"):
        normalize_factors(value, "my_field")


def test_normalize_factors_names_a_numpy_int_below_2_in_the_value_error():
    with pytest.raises(ValueError, match="my_field must be 2 or more, got 1"):
        normalize_factors(np.int64(1), "my_field")
    with pytest.raises(ValueError, match="my_field must hold ratios of 2"):
        normalize_factors([np.int64(8), np.int32(1)], "my_field")


def test_integer_like_converts_what_operator_index_accepts():
    assert type(integer_like(np.int64(4))) is int
    assert integer_like(4) == 4
    assert integer_like(True) is None
    assert integer_like(4.0) is None
    assert integer_like(None) is None


def test_ladder_helpers_agree_for_numpy_ints_and_plain_ints():
    plain = (8, 4)
    numpy_ints = (np.int64(8), np.int32(4))
    assert max_levels_for(numpy_ints) == max_levels_for(plain)
    assert max_levels_for(np.int64(4)) == max_levels_for(4)
    assert level_widths(numpy_ints, 5) == level_widths(plain, 5)
    assert level_factors(numpy_ints, 5) == level_factors(plain, 5)
    assert all(type(w) is int for w in level_widths(numpy_ints, 5))
    assert all(type(f) is int for f in level_factors(np.int64(4), 3))


def test_level_factors_repeats_the_last_ratio():
    assert level_factors((8, 4), 5) == (8, 4, 4, 4, 4)
    assert level_factors(4, 3) == (4, 4, 4)


def test_level_factors_truncates_a_ladder_longer_than_the_levels():
    assert level_factors((8, 4, 2), 2) == (8, 4)
    assert level_factors((8, 4), 0) == ()


def test_level_factors_rejects_a_negative_level_count():
    with pytest.raises(ValueError, match="levels must be 0 or more"):
        level_factors(4, -1)


def test_level_widths_are_the_cumulative_products_of_the_ratios():
    assert level_widths((8, 4), 5) == (8, 32, 128, 512, 2048)
    assert level_widths(4, 4) == (4, 16, 64, 256)
    assert level_widths((2, 3, 5), 3) == (2, 6, 30)
    assert level_widths(4, 0) == ()
