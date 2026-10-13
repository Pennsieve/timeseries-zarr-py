import numpy as np

from timeseries_zarr.constants import (
    CANVAS_PIXELS,
    COUNT_COLUMN,
    COUNT_LADDER_BASE_US,
    DECIMATION_FACTOR,
    EVENT_LEVEL_THRESHOLD,
    FLOAT32_BYTES,
    FORMAT_VERSION,
    INT64_BYTES,
    MAX_COLUMN,
    MAX_LABELS,
    MAX_LEVELS,
    MEAN_COLUMN,
    MIN_COLUMN,
    RASTER_BUDGET,
    STATISTIC_COLUMNS,
    UINT8_BYTES,
    UINT16_BYTES,
    UINT32_BYTES,
    UNIT_TO_UV,
    VAR_COLUMN,
)


def test_format_version_is_the_1_0_draft():
    assert FORMAT_VERSION == "1.0"


def test_max_levels_is_the_last_level_whose_bin_count_fits_uint32():
    limit = np.iinfo(np.uint32).max
    assert limit >= DECIMATION_FACTOR**MAX_LEVELS
    assert limit < DECIMATION_FACTOR ** (MAX_LEVELS + 1)


def test_stat_columns_are_distinct_and_cover_the_block():
    columns = [MIN_COLUMN, MAX_COLUMN, MEAN_COLUMN, VAR_COLUMN, COUNT_COLUMN]
    assert sorted(columns) == list(range(STATISTIC_COLUMNS))


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
