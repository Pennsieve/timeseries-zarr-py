import numpy as np

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


def test_unit_to_uv_factors():
    assert UNIT_TO_UV["volts"] == 1e6
    assert UNIT_TO_UV["v"] == 1e6
    assert UNIT_TO_UV["millivolts"] == 1e3
    assert UNIT_TO_UV["mv"] == 1e3
    assert UNIT_TO_UV["microvolts"] == 1.0
    assert UNIT_TO_UV["uv"] == 1.0


def test_unit_to_uv_aliases_agree():
    assert UNIT_TO_UV["v"] == UNIT_TO_UV["volts"]
    assert UNIT_TO_UV["mv"] == UNIT_TO_UV["millivolts"]
    assert UNIT_TO_UV["uv"] == UNIT_TO_UV["microvolts"]


def test_unit_to_uv_keys_lowercase():
    assert all(k == k.lower() for k in UNIT_TO_UV)
