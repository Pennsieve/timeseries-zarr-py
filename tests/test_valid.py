"""The valid member: how much of a bin was actually recorded.

A NaN bin says something is missing. valid says how much, which is what
separates one bad sample from an amplifier stop, and what tells a reader when a
montage bin is trustworthy.
"""

import numpy as np
import pytest

from timeseries_zarr.constants import (
    COUNT_COL,
    DECIMATION_FACTOR,
    MAX_LEVELS,
    MAX_VALID_COUNT,
    VALID_COL,
)
from timeseries_zarr.fold import fold_raw_block, fold_stat_block
from timeseries_zarr.planning import plan_levels
from timeseries_zarr.types import WriteOpts
from timeseries_zarr.write_continuous import write_continuous_channel
from timeseries_zarr.zarr_io import open_group

_OPTS = WriteOpts(
    min_bins=2, max_levels=7, inner_len=16, target_shard_bytes=256
)

_RATE_HZ = 32000.0
_PERIOD_US = 31.25


def _levels(num_samples):
    return plan_levels(
        num_samples, _PERIOD_US, _OPTS.max_levels, _OPTS.min_bins
    )


def _write(tmp_path, source):
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(parent, 0, source, onset_us=0, opts=_OPTS)
    return open_group(tmp_path / "bundle")["0"]


def test_the_ladder_cannot_overflow_the_valid_member():
    """MAX_LEVELS is what keeps a bin's finite count inside u2.

    A level-7 bin spans 16384 raw samples. Level 8 would span 65536, past what
    the member can hold, so raising the cap needs a wider dtype first.
    """
    assert DECIMATION_FACTOR**MAX_LEVELS <= MAX_VALID_COUNT


def test_valid_counts_finite_samples_not_slots():
    raw = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9], dtype=np.float32)
    out = fold_raw_block(raw)
    # Three bins: four slots, four slots, one slot.
    assert list(out[:, COUNT_COL]) == [4.0, 4.0, 1.0]
    assert list(out[:, VALID_COL]) == [3.0, 4.0, 1.0]


def test_valid_and_count_stay_distinct_through_the_fold():
    """The two numbers are equal only where nothing is missing."""
    raw = np.full(16, np.nan, dtype=np.float32)
    raw[0] = 1.0
    level2 = fold_stat_block(fold_raw_block(raw))
    assert level2[0, COUNT_COL] == 16.0
    assert level2[0, VALID_COL] == 1.0


def test_valid_sums_exactly_up_the_ladder(tmp_path, continuous_source):
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(1024).astype(np.float32)
    samples[[3, 400, 401, 900]] = np.nan
    grp = _write(tmp_path, continuous_source(samples, rate_hz=_RATE_HZ))

    finite = np.isfinite(samples)
    for plan in _levels(samples.shape[0]):
        span = DECIMATION_FACTOR**plan.level
        expected = [
            finite[i : i + span].sum() for i in range(0, samples.shape[0], span)
        ]
        assert list(grp[str(plan.level)]["valid"][:]) == expected


def test_a_full_bin_holding_one_nan_reports_one_short(
    tmp_path, continuous_source
):
    samples = np.ones(1024, dtype=np.float32)
    samples[5] = np.nan
    grp = _write(tmp_path, continuous_source(samples, rate_hz=_RATE_HZ))

    # Level 1 bin 1 covers samples 4..7, so it is three of four.
    assert grp["1"]["valid"][1] == 3
    assert grp["1"]["valid"][0] == 4
    # The mean is still NaN: valid describes the bin, it does not rescue it.
    assert np.isnan(grp["1"]["mean"][1])


def test_a_gap_reports_zero_while_its_bins_stay_full_width(
    tmp_path, continuous_source
):
    """A dropout and a single bad sample must not look alike."""
    samples = np.ones(1024, dtype=np.float32)
    samples[256:512] = np.nan
    grp = _write(tmp_path, continuous_source(samples, rate_hz=_RATE_HZ))

    valid = grp["1"]["valid"][:]
    assert valid[0] == 4
    assert np.all(valid[64:128] == 0)
    # The bins still exist and still span their slots; only the content is gone.
    assert grp["1"]["env"].shape[0] == 256


def test_a_clean_recording_writes_no_valid(tmp_path, continuous_source):
    """The spec writes valid only for channels containing non-finite samples."""
    samples = np.arange(1024, dtype=np.float32)
    grp = _write(tmp_path, continuous_source(samples, rate_hz=_RATE_HZ))
    for plan in _levels(samples.shape[0]):
        assert "valid" not in grp[str(plan.level)].array_keys()


def test_one_non_finite_sample_writes_valid_at_every_level(
    tmp_path, continuous_source
):
    samples = np.arange(1024, dtype=np.float32)
    samples[700] = np.inf
    grp = _write(tmp_path, continuous_source(samples, rate_hz=_RATE_HZ))
    for plan in _levels(samples.shape[0]):
        assert "valid" in grp[str(plan.level)].array_keys()


def test_valid_is_u2_and_fills_with_zero(tmp_path, continuous_source):
    samples = np.arange(1024, dtype=np.float32)
    samples[0] = np.nan
    grp = _write(tmp_path, continuous_source(samples, rate_hz=_RATE_HZ))
    valid = grp["1"]["valid"]
    assert valid.dtype == np.uint16
    # Integers fill 0, which reads as "nothing recorded here" rather than NaN.
    assert valid.fill_value == 0


def test_a_short_trailing_bin_reports_its_own_width(
    tmp_path, continuous_source
):
    samples = np.arange(1013, dtype=np.float32)
    # One gap at the start, so the channel carries valid at all.
    samples[0] = np.nan
    grp = _write(tmp_path, continuous_source(samples, rate_hz=_RATE_HZ))
    for plan in _levels(samples.shape[0]):
        span = DECIMATION_FACTOR**plan.level
        stored = grp[str(plan.level)]["valid"][:]
        assert stored[-1] == 1013 - (stored.shape[0] - 1) * span
        assert stored[0] == span - 1
        assert np.all(stored[1:-1] == span)


@pytest.mark.parametrize("nan_at", [0, 1, 2, 3])
def test_valid_is_unaffected_by_where_in_the_bin_the_nan_sits(
    tmp_path, continuous_source, nan_at
):
    samples = np.ones(1024, dtype=np.float32)
    samples[nan_at] = np.nan
    grp = _write(tmp_path, continuous_source(samples, rate_hz=_RATE_HZ))
    assert grp["1"]["valid"][0] == 3
