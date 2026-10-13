import math

import numpy as np
import pytest
from zarr import Array

from timeseries_zarr.attrs import channel_group_attrs
from timeseries_zarr.fold import fold_block, fold_raw_block, split_members
from timeseries_zarr.planning import plan_levels, sample_period_us
from timeseries_zarr.sizing import level_sizing
from timeseries_zarr.types import ChunkShard, LevelPlan, WriteOpts
from timeseries_zarr.write_continuous import (
    COUNT_KEY,
    MEAN_KEY,
    MINMAX_KEY,
    RAW_KEY,
    VAR_KEY,
    create_level,
    write_continuous_channel,
    write_level0,
    write_level_from_previous,
    write_raw,
    write_statistics_blocks,
)
from timeseries_zarr.zarr_io import create_array, open_group, write_region


def _plan(n, period_us=31.25):
    return LevelPlan(level=0, bins=n, period_us=period_us)


def _sizing():
    return ChunkShard(chunk_shape=(4,), shard_shape=(8,))


def _all_plans(num_samples=64, rate_hz=32000.0):
    """Return the raw plan followed by the planned folded levels."""
    return [
        LevelPlan(
            level=0, bins=num_samples, period_us=sample_period_us(rate_hz)
        ),
        *plan_levels(num_samples, rate_hz, 8, 2),
    ]


def _make_prev(group, data):
    """Create and fill a "prev" array on disk matching data's shape."""
    grid = (max(1, data.shape[0]), *data.shape[1:])
    arr = create_array(group, "prev", data.shape, np.float32, grid, grid, {}, 5)
    if data.shape[0]:
        write_region(arr, 0, data)
    return arr


def _fold_plan(level, n_prev_rows, chunk0):
    """Return the plan and sizing for folding n_prev_rows by 4 into level."""
    bins = math.ceil(n_prev_rows / 4)
    plan = LevelPlan(level=level, bins=bins, period_us=125.0)
    sizing = ChunkShard(chunk_shape=(chunk0, 2), shard_shape=(chunk0, 2))
    return plan, sizing


def test_write_level0_round_trips_samples(tmp_path, continuous_source):
    samples = np.arange(10, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    write_level0(group, continuous_source(samples), _plan(10), _sizing(), 5)
    stored = open_group(tmp_path / "bundle")["0"][:]
    assert np.array_equal(stored, samples)
    assert stored.dtype == np.float32


def test_write_level0_creates_named_array_with_shape_and_dtype(
    tmp_path, continuous_source
):
    samples = np.arange(10, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    write_level0(group, continuous_source(samples), _plan(10), _sizing(), 5)
    arr = open_group(tmp_path / "bundle")["0"]
    assert arr.shape == (10,)
    assert arr.dtype == np.float32


def test_write_level0_sets_period_us_attr(tmp_path, continuous_source):
    samples = np.arange(10, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    write_level0(
        group, continuous_source(samples), _plan(10, 62.5), _sizing(), 5
    )
    assert dict(open_group(tmp_path / "bundle")["0"].attrs) == {
        "period_us": 62.5
    }


def test_write_level0_chunk_and_shard_grid(tmp_path, continuous_source):
    samples = np.arange(10, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    write_level0(group, continuous_source(samples), _plan(10), _sizing(), 5)
    arr = open_group(tmp_path / "bundle")["0"]
    assert arr.shards == (8,)
    assert arr.chunks == (4,)


def test_write_level0_returns_the_array(tmp_path, continuous_source):
    samples = np.arange(10, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    result = write_level0(
        group, continuous_source(samples), _plan(10), _sizing(), 5
    )
    assert isinstance(result, Array)
    assert result.shape == (10,)


def test_write_level0_empty_source_creates_array_writes_nothing(
    tmp_path, continuous_source
):
    group = open_group(tmp_path / "bundle")
    write_level0(
        group,
        continuous_source(np.empty(0, np.float32)),
        _plan(0),
        _sizing(),
        5,
    )
    arr = open_group(tmp_path / "bundle")["0"]
    assert arr.shape == (0,)
    assert arr[:].shape == (0,)


def test_write_level0_rejects_non_raw_plan(tmp_path, continuous_source):
    group = open_group(tmp_path / "bundle")
    non_raw = LevelPlan(level=1, bins=5, period_us=125.0)
    with pytest.raises(ValueError, match="raw"):
        write_level0(
            group,
            continuous_source(np.arange(5, dtype=np.float32)),
            non_raw,
            _sizing(),
            5,
        )


def test_write_level_from_previous_folds_level0_to_level1(tmp_path):
    data = np.arange(20, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    plan, sizing = _fold_plan(1, data.shape[0], 4)
    write_level_from_previous(group, _make_prev(group, data), plan, sizing, 5)
    stored = open_group(tmp_path / "bundle")["1"][:]
    assert np.array_equal(stored, fold_block(data))
    assert stored.dtype == np.float32


def test_write_level_from_previous_folds_envelope_level(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((13, 2)).astype(np.float32)
    group = open_group(tmp_path / "bundle")
    plan, sizing = _fold_plan(2, data.shape[0], 4)
    write_level_from_previous(group, _make_prev(group, data), plan, sizing, 5)
    stored = open_group(tmp_path / "bundle")["2"][:]
    assert np.array_equal(stored, fold_block(data))


@pytest.mark.parametrize("chunk0", [1, 3, 4, 5, 7, 16, 1000])
def test_write_level_from_previous_chunk_boundary_exact(tmp_path, chunk0):
    data = np.arange(37, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    plan, sizing = _fold_plan(1, data.shape[0], chunk0)
    write_level_from_previous(group, _make_prev(group, data), plan, sizing, 5)
    stored = open_group(tmp_path / "bundle")["1"][:]
    assert np.array_equal(stored, fold_block(data))


def test_write_level_from_previous_sets_period_us_attr(tmp_path):
    data = np.arange(20, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    plan, sizing = _fold_plan(1, data.shape[0], 4)
    write_level_from_previous(group, _make_prev(group, data), plan, sizing, 5)
    assert dict(open_group(tmp_path / "bundle")["1"].attrs) == {
        "period_us": 125.0
    }


def test_write_level_from_previous_returns_the_array(tmp_path):
    data = np.arange(20, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    plan, sizing = _fold_plan(1, data.shape[0], 4)
    result = write_level_from_previous(
        group, _make_prev(group, data), plan, sizing, 5
    )
    assert isinstance(result, Array)
    assert result.shape == (5, 2)


def test_write_level_from_previous_empty_prev(tmp_path):
    data = np.empty((0,), dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    plan, sizing = _fold_plan(1, 0, 4)
    write_level_from_previous(group, _make_prev(group, data), plan, sizing, 5)
    arr = open_group(tmp_path / "bundle")["1"]
    assert arr.shape == (0, 2)
    assert arr[:].shape == (0, 2)


def test_write_level_from_previous_rejects_level0_plan(tmp_path):
    data = np.arange(20, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    sizing = ChunkShard(chunk_shape=(4, 2), shard_shape=(4, 2))
    with pytest.raises(ValueError, match="level"):
        write_level_from_previous(
            group, _make_prev(group, data), _plan(5), sizing, 5
        )


_MULTI_OPTS = WriteOpts(
    min_bins=2, max_levels=8, inner_len=16, target_shard_bytes=256
)


def test_write_continuous_channel_creates_subgroup_with_attrs(
    tmp_path, continuous_source
):
    samples = np.arange(64, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    src = continuous_source(samples, id="N:ch:xyz", rate_hz=32000.0, start_us=7)
    write_continuous_channel(parent, 3, src, opts=_MULTI_OPTS)
    grp = open_group(tmp_path / "bundle")["3"]
    assert dict(grp.attrs) == channel_group_attrs(
        "N:ch:xyz", 32000.0, 7, "continuous", src.name, src.unit
    )


def test_write_continuous_channel_level0_round_trips(
    tmp_path, continuous_source
):
    samples = np.arange(64, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(
        parent, 0, continuous_source(samples), opts=_MULTI_OPTS
    )
    assert np.array_equal(open_group(tmp_path / "bundle")["0"]["0"][:], samples)


def test_write_continuous_channel_each_level_folds_from_below(
    tmp_path, continuous_source
):
    samples = np.arange(64, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(
        parent, 0, continuous_source(samples), opts=_MULTI_OPTS
    )
    grp = open_group(tmp_path / "bundle")["0"]
    n_levels = len(_all_plans())
    assert n_levels >= 3
    for k in range(1, n_levels):
        below = grp[str(k - 1)][:]
        assert np.array_equal(grp[str(k)][:], fold_block(below))


def test_write_continuous_channel_level_arrays_and_periods(
    tmp_path, continuous_source
):
    samples = np.arange(64, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(
        parent, 0, continuous_source(samples), opts=_MULTI_OPTS
    )
    grp = open_group(tmp_path / "bundle")["0"]
    plans = _all_plans()
    assert sorted(grp.array_keys()) == sorted(str(p.level) for p in plans)
    for p in plans:
        assert dict(grp[str(p.level)].attrs) == {"period_us": p.period_us}
        assert grp[str(p.level)].shape[0] == p.bins


def test_write_continuous_channel_degenerate_single_level(
    tmp_path, continuous_source
):
    samples = np.arange(3, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(
        parent, 0, continuous_source(samples), opts=WriteOpts()
    )
    grp = open_group(tmp_path / "bundle")["0"]
    assert list(grp.array_keys()) == ["0"]
    assert np.array_equal(grp["0"][:], samples)


def test_write_continuous_channel_empty_source_writes_empty_level0(
    tmp_path, continuous_source
):
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(parent, 0, continuous_source([]), opts=WriteOpts())
    grp = open_group(tmp_path / "bundle")["0"]
    assert list(grp.array_keys()) == ["0"]
    assert grp["0"].shape == (0,)


def test_write_continuous_channel_all_zero_source_writes_every_shard(
    tmp_path, continuous_source
):
    samples = np.zeros(64, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(
        parent, 0, continuous_source(samples), opts=_MULTI_OPTS
    )
    grp = open_group(tmp_path / "bundle")["0"]
    assert len(list(grp.array_keys())) >= 3
    for key in grp.array_keys():
        level_dir = tmp_path / "bundle" / "0" / key
        shard_files = [
            f
            for f in level_dir.rglob("*")
            if f.is_file() and f.name != "zarr.json"
        ]
        assert shard_files, f"level {key} has no shard file"


def test_write_continuous_channel_returns_none(tmp_path, continuous_source):
    samples = np.arange(64, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    result = write_continuous_channel(
        parent, 0, continuous_source(samples), opts=_MULTI_OPTS
    )
    assert result is None


def _record_writes(monkeypatch, module):
    """Record (start, rows) for every write_region call made by module."""
    writes = []

    def spy(array, start, block):
        writes.append((start, block.shape[0]))
        write_region(array, start, block)

    monkeypatch.setattr(f"{module}.write_region", spy)
    return writes


def test_write_level0_writes_one_whole_shard_per_write(
    tmp_path, continuous_source, monkeypatch
):
    writes = _record_writes(monkeypatch, "timeseries_zarr.write_continuous")
    samples = np.arange(26, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    write_level0(group, continuous_source(samples), _plan(26), _sizing(), 5)
    assert writes == [(0, 8), (8, 8), (16, 8), (24, 2)]
    assert np.array_equal(open_group(tmp_path / "bundle")["0"][:], samples)


def test_write_level_from_previous_writes_one_whole_shard_per_write(
    tmp_path, monkeypatch
):
    group = open_group(tmp_path / "bundle")
    data = np.arange(64, dtype=np.float32)
    prev = _make_prev(group, data)
    plan = LevelPlan(level=1, bins=16, period_us=125.0)
    sizing = ChunkShard(chunk_shape=(4, 2), shard_shape=(8, 2))
    writes = _record_writes(monkeypatch, "timeseries_zarr.write_continuous")
    write_level_from_previous(group, prev, plan, sizing, 5)
    assert writes == [(0, 8), (8, 8)]
    assert np.array_equal(
        open_group(tmp_path / "bundle")["1"][:], fold_block(data)
    )


def _gappy_samples(n=26):
    rng = np.random.default_rng(0)
    samples = (5000.0 + 8.0 * rng.standard_normal(n)).astype(np.float32)
    samples[3] = np.nan
    samples[10] = np.inf
    samples[11] = -np.inf
    return samples


def test_write_raw_round_trips_samples_unchanged(tmp_path, continuous_source):
    samples = _gappy_samples()
    group = open_group(tmp_path / "bundle")
    write_raw(group, continuous_source(samples), _sizing(), 5)
    stored = open_group(tmp_path / "bundle")[RAW_KEY][:]
    assert stored.dtype == np.float32
    assert np.array_equal(stored, samples, equal_nan=True)


def test_write_raw_creates_one_array_named_raw_with_no_attributes(
    tmp_path, continuous_source
):
    group = open_group(tmp_path / "bundle")
    write_raw(group, continuous_source(_gappy_samples()), _sizing(), 5)
    stored = open_group(tmp_path / "bundle")
    assert list(stored.array_keys()) == ["raw"]
    assert dict(stored["raw"].attrs) == {}


def test_write_raw_declares_shape_time_dimension_and_nan_fill(
    tmp_path, continuous_source
):
    group = open_group(tmp_path / "bundle")
    write_raw(group, continuous_source(_gappy_samples()), _sizing(), 5)
    arr = open_group(tmp_path / "bundle")["raw"]
    assert arr.shape == (26,)
    assert arr.metadata.dimension_names == ("time",)
    assert np.isnan(arr.fill_value)


def test_write_raw_chunk_and_shard_grid(tmp_path, continuous_source):
    group = open_group(tmp_path / "bundle")
    write_raw(group, continuous_source(_gappy_samples()), _sizing(), 5)
    arr = open_group(tmp_path / "bundle")["raw"]
    assert arr.chunks == (4,)
    assert arr.shards == (8,)


def test_write_raw_returns_the_array(tmp_path, continuous_source):
    group = open_group(tmp_path / "bundle")
    result = write_raw(group, continuous_source(_gappy_samples()), _sizing(), 5)
    assert isinstance(result, Array)
    assert result.shape == (26,)


def test_write_raw_empty_source_creates_a_zero_length_array(
    tmp_path, continuous_source
):
    group = open_group(tmp_path / "bundle")
    write_raw(group, continuous_source(np.empty(0, np.float32)), _sizing(), 5)
    arr = open_group(tmp_path / "bundle")["raw"]
    assert arr.shape == (0,)
    assert arr[:].shape == (0,)


def test_write_raw_writes_one_whole_shard_per_write(
    tmp_path, continuous_source, monkeypatch
):
    writes = _record_writes(monkeypatch, "timeseries_zarr.write_continuous")
    samples = _gappy_samples(26)
    group = open_group(tmp_path / "bundle")
    write_raw(group, continuous_source(samples), _sizing(), 5)
    assert writes == [(0, 8), (8, 8), (16, 8), (24, 2)]
    stored = open_group(tmp_path / "bundle")["raw"][:]
    assert np.array_equal(stored, samples, equal_nan=True)


def test_write_raw_writes_every_shard_of_an_all_gap_source(
    tmp_path, continuous_source
):
    samples = np.full(20, np.nan, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    write_raw(group, continuous_source(samples), _sizing(), 5)
    shards = [
        f
        for f in (tmp_path / "bundle" / "raw").rglob("*")
        if f.is_file() and f.name != "zarr.json"
    ]
    assert len(shards) == 3


def _level_plan(level=1, bins=20):
    return LevelPlan(level=level, bins=bins, period_us=125.0)


def _level_sizing(bins=20):
    """Chunks of 4 rows and shards of 8 rows, for a (bins, 2) float32 minmax."""
    return level_sizing(bins, inner_len=4, target_shard_bytes=64)


def test_create_level_creates_a_group_named_by_the_level_with_its_period(
    tmp_path,
):
    parent = open_group(tmp_path / "bundle")
    create_level(parent, _level_plan(level=3), _level_sizing(), 5, False)
    group = open_group(tmp_path / "bundle")["3"]
    assert dict(group.attrs) == {"tsz": {"period_us": 125.0}}


@pytest.mark.parametrize(
    ("write_var", "expected"),
    [
        (False, [COUNT_KEY, MEAN_KEY, MINMAX_KEY]),
        (True, [COUNT_KEY, MEAN_KEY, MINMAX_KEY, VAR_KEY]),
    ],
)
def test_create_level_holds_the_members_write_var_selects(
    tmp_path, write_var, expected
):
    parent = open_group(tmp_path / "bundle")
    members = create_level(parent, _level_plan(), _level_sizing(), 5, write_var)
    group = open_group(tmp_path / "bundle")["1"]
    assert sorted(group.array_keys()) == sorted(expected)
    assert (members.var is None) == (not write_var)


def test_create_level_member_shapes_and_dtypes(tmp_path):
    parent = open_group(tmp_path / "bundle")
    members = create_level(parent, _level_plan(), _level_sizing(), 5, True)
    assert members.minmax.shape == (20, 2)
    assert members.minmax.dtype == np.float32
    for array in (members.mean, members.var):
        assert array.shape == (20,)
        assert array.dtype == np.float32
    assert members.count.shape == (20,)
    assert members.count.dtype == np.uint32


def test_create_level_names_the_bin_axis_time(tmp_path):
    parent = open_group(tmp_path / "bundle")
    members = create_level(parent, _level_plan(), _level_sizing(), 5, True)
    assert members.minmax.metadata.dimension_names == ("time", None)
    for array in (members.mean, members.var, members.count):
        assert array.metadata.dimension_names == ("time",)


def test_create_level_floats_fill_nan_and_count_fills_zero(tmp_path):
    parent = open_group(tmp_path / "bundle")
    members = create_level(parent, _level_plan(), _level_sizing(), 5, True)
    for array in (members.minmax, members.mean, members.var):
        assert np.isnan(array.fill_value)
    assert members.count.fill_value == 0


def test_create_level_members_share_the_minmax_row_counts(tmp_path):
    parent = open_group(tmp_path / "bundle")
    members = create_level(parent, _level_plan(), _level_sizing(), 5, True)
    assert (members.minmax.chunks, members.minmax.shards) == ((4, 2), (8, 2))
    for array in (members.mean, members.var, members.count):
        assert (array.chunks, array.shards) == ((4,), (8,))


def test_create_level_rejects_a_level_below_one(tmp_path):
    parent = open_group(tmp_path / "bundle")
    with pytest.raises(ValueError, match="level"):
        create_level(parent, _level_plan(level=0), _level_sizing(), 5, False)


def _statistics(rows=20, seed=0):
    rng = np.random.default_rng(seed)
    raw = (5000.0 + 8.0 * rng.standard_normal(4 * rows)).astype(np.float32)
    raw[6] = np.nan
    raw[30:38] = np.nan
    raw[50] = np.inf
    return fold_raw_block(raw)


def _cut_rows(statistics, sizes):
    out, start = [], 0
    for size in sizes:
        out.append(statistics[start : start + size])
        start += size
    return out


def test_write_statistics_blocks_stores_every_member_as_split(tmp_path):
    parent = open_group(tmp_path / "bundle")
    members = create_level(parent, _level_plan(), _level_sizing(), 5, True)
    statistics = _statistics()
    write_statistics_blocks(members, _cut_rows(statistics, [8, 8, 4]))
    group = open_group(tmp_path / "bundle")["1"]
    minmax, mean, var, count = split_members(statistics, True)
    assert np.array_equal(group[MINMAX_KEY][:], minmax, equal_nan=True)
    assert np.array_equal(group[MEAN_KEY][:], mean, equal_nan=True)
    assert np.array_equal(group[VAR_KEY][:], var, equal_nan=True)
    assert np.array_equal(group[COUNT_KEY][:], count)


def test_write_statistics_blocks_without_var_drops_the_variance_column(
    tmp_path,
):
    parent = open_group(tmp_path / "bundle")
    members = create_level(parent, _level_plan(), _level_sizing(), 5, False)
    statistics = _statistics()
    write_statistics_blocks(members, [statistics])
    group = open_group(tmp_path / "bundle")["1"]
    minmax, mean, _, count = split_members(statistics, False)
    assert VAR_KEY not in list(group.array_keys())
    assert np.array_equal(group[MEAN_KEY][:], mean, equal_nan=True)
    assert np.array_equal(group[COUNT_KEY][:], count)


def test_write_statistics_blocks_writes_each_block_at_a_running_offset(
    tmp_path, monkeypatch
):
    writes = _record_writes(monkeypatch, "timeseries_zarr.write_continuous")
    parent = open_group(tmp_path / "bundle")
    members = create_level(parent, _level_plan(), _level_sizing(), 5, True)
    write_statistics_blocks(members, _cut_rows(_statistics(), [8, 8, 4]))
    assert writes == [(0, 8)] * 4 + [(8, 8)] * 4 + [(16, 4)] * 4


def test_write_statistics_blocks_without_var_writes_three_members_per_block(
    tmp_path, monkeypatch
):
    writes = _record_writes(monkeypatch, "timeseries_zarr.write_continuous")
    parent = open_group(tmp_path / "bundle")
    members = create_level(parent, _level_plan(), _level_sizing(), 5, False)
    write_statistics_blocks(members, _cut_rows(_statistics(), [8, 12]))
    assert writes == [(0, 8)] * 3 + [(8, 12)] * 3


def test_write_statistics_blocks_empty_stream_leaves_the_fill_value(tmp_path):
    parent = open_group(tmp_path / "bundle")
    members = create_level(parent, _level_plan(), _level_sizing(), 5, True)
    write_statistics_blocks(members, [])
    assert np.isnan(members.minmax[:]).all()
    assert not members.count[:].any()


def test_write_statistics_blocks_writes_every_shard_of_an_all_gap_level(
    tmp_path,
):
    parent = open_group(tmp_path / "bundle")
    members = create_level(parent, _level_plan(), _level_sizing(), 5, True)
    gap = fold_raw_block(np.full(80, np.nan, dtype=np.float32))
    write_statistics_blocks(members, _cut_rows(gap, [8, 8, 4]))
    for key in (MINMAX_KEY, MEAN_KEY, VAR_KEY, COUNT_KEY):
        shards = [
            f
            for f in (tmp_path / "bundle" / "1" / key).rglob("*")
            if f.is_file() and f.name != "zarr.json"
        ]
        assert len(shards) == 3, key
