"""The count pyramid a dense event channel carries.

A sparse channel answers every zoom from its own events. A dense one cannot be
rastered at overview zoom, so it carries per-bin counts folded by summation,
which stays exact however far it is coarsened.
"""

import numpy as np
import pytest

from timeseries_zarr.counts import (
    base_period_us,
    count_shape,
    fold_counts_block,
    iter_count_blocks,
    level_bins,
    plan_count_levels,
)
from timeseries_zarr.types import WriteOpts
from timeseries_zarr.write_unit import write_unit_channel
from timeseries_zarr.zarr_io import open_group

_TARGET_BYTES = 16 * 2**20


def _blocks(times, labels=None, size=3):
    """Split events into windows the way a source's reads arrive."""
    times = np.asarray(times, dtype=np.int64)
    labels = None if labels is None else np.asarray(labels, dtype=np.uint16)
    for start in range(0, times.shape[0], size):
        stop = start + size
        yield (
            times[start:stop],
            None if labels is None else labels[start:stop],
        )


def _counted(times, labels, n_labels, period_us, bins, block_bins=4):
    out = list(
        iter_count_blocks(
            _blocks(times, labels), n_labels, period_us, bins, block_bins
        )
    )
    return np.concatenate(out, axis=0) if out else np.empty((0,))


def test_count_shape_drops_the_label_axis_when_there_are_no_labels():
    assert count_shape(10, 3) == (10, 3)
    assert count_shape(10, 0) == (10,)


def test_base_period_shrinks_as_labels_multiply():
    """The label axis is what makes the finest level expensive."""
    one = base_period_us(1_000_000_000, 1, _TARGET_BYTES)
    many = base_period_us(1_000_000_000, 100, _TARGET_BYTES)
    assert many > one


def test_base_period_has_a_floor():
    assert base_period_us(10, 1, _TARGET_BYTES) == 1.0


def test_level_bins_keeps_a_partial_last_bin():
    assert level_bins(10, 4.0) == 3
    assert level_bins(12, 4.0) == 3
    assert level_bins(0, 4.0) == 0


def test_plan_count_levels_steps_up_by_four():
    plans = plan_count_levels(10**9, 4, 7, 16, _TARGET_BYTES)
    assert [p.level for p in plans] == list(range(1, len(plans) + 1))
    for below, above in zip(plans, plans[1:], strict=False):
        assert above.period_us == pytest.approx(below.period_us * 4)
        assert above.shape[0] <= below.shape[0]


def test_plan_count_levels_stops_below_the_bin_floor():
    plans = plan_count_levels(10**9, 4, 7, 1024, _TARGET_BYTES)
    assert plans[-1].shape[0] >= 1
    assert all(p.shape[1] == 4 for p in plans)


def test_plan_count_levels_of_an_empty_span_is_empty():
    assert plan_count_levels(0, 4, 7, 1024, _TARGET_BYTES) == []


def test_fold_counts_sums_rather_than_averages():
    counts = np.arange(8, dtype=np.uint32).reshape(8, 1)
    out = fold_counts_block(counts)
    assert list(out[:, 0]) == [0 + 1 + 2 + 3, 4 + 5 + 6 + 7]


def test_fold_counts_keeps_a_partial_trailing_group():
    counts = np.ones((6, 2), dtype=np.uint32)
    out = fold_counts_block(counts)
    assert out.shape == (2, 2)
    assert list(out[0]) == [4, 4]
    assert list(out[1]) == [2, 2]


def test_fold_counts_is_exact_over_repeated_folds():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 50, size=(64, 3)).astype(np.uint32)
    once = fold_counts_block(counts)
    twice = fold_counts_block(once)
    # Nothing is averaged away, so the total survives every level.
    assert twice.sum() == counts.sum()


def test_fold_counts_empty():
    assert fold_counts_block(np.empty((0, 2), dtype=np.uint32)).shape == (0, 2)


def test_counting_places_events_in_their_bin():
    times = np.array([0, 3, 4, 11], dtype=np.int64)
    labels = np.zeros(4, dtype=np.uint16)
    out = _counted(times, labels, 1, 4.0, 3)
    assert list(out[:, 0]) == [2, 1, 1]


def test_counting_separates_labels_into_columns():
    times = np.array([0, 1, 5], dtype=np.int64)
    labels = np.array([0, 2, 1], dtype=np.uint16)
    out = _counted(times, labels, 3, 4.0, 2)
    assert out.shape == (2, 3)
    assert list(out[0]) == [1, 0, 1]
    assert list(out[1]) == [0, 1, 0]


def test_counting_covers_bins_no_event_falls_in():
    times = np.array([0, 40], dtype=np.int64)
    labels = np.zeros(2, dtype=np.uint16)
    out = _counted(times, labels, 1, 4.0, 11)
    assert out.shape == (11, 1)
    assert out[:, 0].sum() == 2
    assert np.all(out[1:10, 0] == 0)


def test_counting_is_unlabelled_when_the_channel_is():
    times = np.array([0, 1, 9], dtype=np.int64)
    out = _counted(times, None, 0, 4.0, 3)
    assert out.ndim == 1
    assert list(out) == [2, 0, 1]


@pytest.mark.parametrize("block_bins", [1, 2, 3, 4, 7, 64])
def test_counting_is_block_invariant(block_bins):
    rng = np.random.default_rng(1)
    times = np.sort(rng.integers(0, 400, size=200)).astype(np.int64)
    labels = rng.integers(0, 3, size=200).astype(np.uint16)
    out = _counted(times, labels, 3, 4.0, 100, block_bins=block_bins)
    assert out.shape == (100, 3)
    assert out.sum() == 200


def test_counting_totals_every_event():
    rng = np.random.default_rng(2)
    times = np.sort(rng.integers(0, 1000, size=500)).astype(np.int64)
    labels = rng.integers(0, 5, size=500).astype(np.uint16)
    out = _counted(times, labels, 5, 10.0, 100, block_bins=8)
    assert out.sum() == 500
    for label in range(5):
        assert out[:, label].sum() == (labels == label).sum()


def test_counting_rejects_a_nonpositive_block():
    with pytest.raises(ValueError, match="positive"):
        list(iter_count_blocks(_blocks([0]), 1, 4.0, 4, 0))


_OPTS = WriteOpts(
    min_bins=4,
    max_levels=7,
    inner_len=16,
    target_shard_bytes=4096,
    event_level_threshold=100,
)


def _write_channel(tmp_path, source, opts=_OPTS):
    parent = open_group(tmp_path / "bundle")
    write_unit_channel(parent, 0, source, onset_us=0, opts=opts)
    return open_group(tmp_path / "bundle")["0"]


def test_a_sparse_channel_gets_no_count_levels(tmp_path, unit_source):
    events = np.arange(50, dtype=np.int64) * 1000
    grp = _write_channel(tmp_path, unit_source(events, num_labels=1))
    # Below the threshold a reader rasters the events themselves.
    assert list(grp.group_keys()) == []


def test_a_dense_channel_gets_count_levels(tmp_path, unit_source):
    rng = np.random.default_rng(3)
    events = np.sort(rng.integers(0, 10**8, size=500)).astype(np.int64)
    labels = rng.integers(0, 4, size=500).astype(np.uint16)
    grp = _write_channel(
        tmp_path, unit_source(events, labels=labels, num_labels=4)
    )
    levels = sorted(int(k) for k in grp.group_keys())
    assert levels
    assert levels == list(range(1, len(levels) + 1))
    for level in levels:
        assert list(grp[str(level)].array_keys()) == ["counts"]
        assert grp[str(level)]["counts"].dtype == np.uint32


def test_every_count_level_totals_the_event_count(tmp_path, unit_source):
    rng = np.random.default_rng(4)
    events = np.sort(rng.integers(0, 10**8, size=500)).astype(np.int64)
    labels = rng.integers(0, 4, size=500).astype(np.uint16)
    grp = _write_channel(
        tmp_path, unit_source(events, labels=labels, num_labels=4)
    )
    for key in grp.group_keys():
        # The sum fold is exact, so no level loses or invents an event.
        assert grp[key]["counts"][:].sum() == 500


def test_each_count_level_is_the_fold_of_the_one_below(tmp_path, unit_source):
    rng = np.random.default_rng(5)
    events = np.sort(rng.integers(0, 10**8, size=500)).astype(np.int64)
    labels = rng.integers(0, 4, size=500).astype(np.uint16)
    grp = _write_channel(
        tmp_path, unit_source(events, labels=labels, num_labels=4)
    )
    levels = sorted(int(k) for k in grp.group_keys())
    for level in levels[1:]:
        below = grp[str(level - 1)]["counts"][:]
        assert np.array_equal(
            grp[str(level)]["counts"][:], fold_counts_block(below)
        )


def test_count_levels_carry_period_us_on_the_group(tmp_path, unit_source):
    rng = np.random.default_rng(6)
    events = np.sort(rng.integers(0, 10**8, size=500)).astype(np.int64)
    labels = rng.integers(0, 4, size=500).astype(np.uint16)
    grp = _write_channel(
        tmp_path, unit_source(events, labels=labels, num_labels=4)
    )
    levels = sorted(int(k) for k in grp.group_keys())
    periods = [grp[str(level)].attrs["period_us"] for level in levels]
    for below, above in zip(periods, periods[1:], strict=False):
        assert above == pytest.approx(below * 4)


def test_counts_keep_the_label_axis_unchunked(tmp_path, unit_source):
    """A window read wants every label in it; splitting them multiplies reads."""
    rng = np.random.default_rng(7)
    events = np.sort(rng.integers(0, 10**8, size=500)).astype(np.int64)
    labels = rng.integers(0, 4, size=500).astype(np.uint16)
    grp = _write_channel(
        tmp_path, unit_source(events, labels=labels, num_labels=4)
    )
    for key in grp.group_keys():
        counts = grp[key]["counts"]
        assert counts.chunks[1] == 4
        assert counts.shards[1] == 4


def test_count_levels_are_onset_relative(tmp_path, unit_source):
    """The pyramid shares the timeline the events are written on."""
    onset = 1_000_000
    events = (np.arange(200, dtype=np.int64) * 1000) + onset
    parent = open_group(tmp_path / "bundle")
    write_unit_channel(
        parent,
        0,
        unit_source(events, num_labels=1, start_us=onset),
        onset_us=onset,
        opts=_OPTS,
    )
    grp = open_group(tmp_path / "bundle")["0"]
    counts = grp["1"]["counts"][:]
    # Every event lands inside the span, none before bin 0.
    assert counts.sum() == 200
    assert counts[0].sum() >= 1


def test_count_bins_start_where_period_us_says(tmp_path, unit_source):
    """A reader puts bin i at offset_us + i * period_us, so counting must use that period."""
    rng = np.random.default_rng(8)
    # A span the bin budget does not divide, so an unrounded period is fractional.
    events = np.sort(rng.integers(0, 10**8 + 12_345, size=2000)).astype(
        np.int64
    )
    labels = rng.integers(0, 4, size=2000).astype(np.uint16)
    grp = _write_channel(
        tmp_path, unit_source(events, labels=labels, num_labels=4)
    )
    stored = grp["events"][:] - grp.attrs["offset_us"]
    levels = sorted(int(k) for k in grp.group_keys())
    assert levels
    for level in levels:
        period = grp[str(level)].attrs["period_us"]
        counts = grp[str(level)]["counts"][:]
        expected = np.bincount(stored // int(period), minlength=counts.shape[0])
        assert float(period).is_integer()
        assert np.array_equal(counts.sum(axis=1), expected)
