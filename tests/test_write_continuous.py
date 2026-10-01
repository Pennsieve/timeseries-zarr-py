import math

import numpy as np
import pytest
from zarr import Array

from timeseries_zarr.attrs import channel_group_attrs
from timeseries_zarr.constants import (
    MAX_COL,
    MEAN_COL,
    MIN_COL,
    VALID_COL,
)
from timeseries_zarr.fold import fold_block, fold_raw_block, fold_stat_block
from timeseries_zarr.planning import plan_levels
from timeseries_zarr.streaming import (
    _rebuffer_and_fold,
    iter_level_stat_blocks,
    iter_offset_removed_blocks,
)
from timeseries_zarr.types import ChunkShard, LevelPlan, WriteOpts
from timeseries_zarr.write_continuous import (
    write_continuous_channel,
    write_level,
    write_raw,
)
from timeseries_zarr.zarr_io import create_array, open_group, write_region


def _sizing():
    return ChunkShard(chunk_shape=(4,), shard_shape=(8,))


def _make_prev(group, data):
    """Create and fill a "prev" array on disk matching data's shape."""
    grid = (max(1, data.shape[0]), *data.shape[1:])
    arr = create_array(group, "prev", data.shape, np.float32, grid, grid, {}, 5)
    if data.shape[0]:
        write_region(arr, 0, data)
    return arr


def _raw_blocks(group, data, block_len=1024, offset_uv=0.0):
    """Yield the stat blocks folding a raw array on disk into level 1."""
    arr = _make_prev(group, data)
    return _rebuffer_and_fold(
        iter_offset_removed_blocks(arr, offset_uv, block_len), fold_block
    )


def _fold_plan(level, n_prev_rows, chunk0):
    """Return the plan and sizing for folding n_prev_rows by 4 into level."""
    bins = math.ceil(n_prev_rows / 4)
    plan = LevelPlan(level=level, shape=(bins, 2), period_us=125.0)
    sizing = ChunkShard(chunk_shape=(chunk0, 2), shard_shape=(chunk0, 2))
    return plan, sizing


def test_write_raw_round_trips_samples(tmp_path, continuous_source):
    samples = np.arange(10, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    write_raw(group, continuous_source(samples), _sizing(), 5)
    stored = open_group(tmp_path / "bundle")["raw"][:]
    assert np.array_equal(stored, samples)
    assert stored.dtype == np.float32


def test_write_raw_creates_named_array_with_shape_and_dtype(
    tmp_path, continuous_source
):
    samples = np.arange(10, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    write_raw(group, continuous_source(samples), _sizing(), 5)
    arr = open_group(tmp_path / "bundle")["raw"]
    assert arr.shape == (10,)
    assert arr.dtype == np.float32


def test_write_raw_carries_no_attributes(tmp_path, continuous_source):
    """The sample period is the channel's rate_hz; raw does not restate it."""
    samples = np.arange(10, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    write_raw(group, continuous_source(samples), _sizing(), 5)
    assert dict(open_group(tmp_path / "bundle")["raw"].attrs) == {}


def test_write_raw_chunk_and_shard_grid(tmp_path, continuous_source):
    samples = np.arange(10, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    write_raw(group, continuous_source(samples), _sizing(), 5)
    arr = open_group(tmp_path / "bundle")["raw"]
    assert arr.shards == (8,)
    assert arr.chunks == (4,)


def test_write_raw_returns_the_array(tmp_path, continuous_source):
    samples = np.arange(10, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    result, _ = write_raw(group, continuous_source(samples), _sizing(), 5)
    assert isinstance(result, Array)
    assert result.shape == (10,)


def test_write_raw_empty_source_creates_array_writes_nothing(
    tmp_path, continuous_source
):
    group = open_group(tmp_path / "bundle")
    write_raw(group, continuous_source(np.empty(0, np.float32)), _sizing(), 5)
    arr = open_group(tmp_path / "bundle")["raw"]
    assert arr.shape == (0,)
    assert arr[:].shape == (0,)


def _env(block):
    """The env columns of a stat block, as they reach disk."""
    return block[:, MIN_COL : MAX_COL + 1].astype(np.float32)


def _mean(block):
    """The mean column of a stat block, as it reaches disk."""
    return block[:, MEAN_COL].astype(np.float32)


def _valid(block):
    """The valid column of a stat block, as it reaches disk."""
    return block[:, VALID_COL].astype(np.uint16)


def test_write_level_folds_raw_into_env_and_mean(tmp_path):
    data = np.arange(20, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    plan, sizing = _fold_plan(1, data.shape[0], 4)
    write_level(group, _raw_blocks(group, data), plan, sizing, 5)
    level = open_group(tmp_path / "bundle")["1"]
    expected = fold_raw_block(data)
    assert np.array_equal(level["env"][:], _env(expected))
    assert np.array_equal(level["mean"][:], _mean(expected))
    assert level["env"].dtype == np.float32
    assert level["mean"].dtype == np.float32


def test_write_level_folds_a_level_into_the_next(tmp_path):
    raw = np.arange(52, dtype=np.float32)
    below = fold_raw_block(raw)
    group = open_group(tmp_path / "bundle")
    env_arr = _make_prev(group, _env(below))
    mean_arr = create_array(
        group,
        "prev_mean",
        (below.shape[0],),
        np.float32,
        (below.shape[0],),
        (below.shape[0],),
        {},
        5,
    )
    write_region(mean_arr, 0, _mean(below))
    valid_arr = create_array(
        group,
        "prev_valid",
        (below.shape[0],),
        np.uint16,
        (below.shape[0],),
        (below.shape[0],),
        {},
        5,
    )
    write_region(valid_arr, 0, _valid(below))

    plan, sizing = _fold_plan(2, below.shape[0], 4)
    blocks = _rebuffer_and_fold(
        iter_level_stat_blocks(env_arr, mean_arr, valid_arr, 52, 1, 1024),
        fold_block,
    )
    write_level(group, blocks, plan, sizing, 5)

    level = open_group(tmp_path / "bundle")["2"]
    expected = fold_stat_block(below)
    assert np.array_equal(level["env"][:], _env(expected))
    assert np.allclose(level["mean"][:], _mean(expected))
    assert np.array_equal(level["valid"][:], _valid(expected))


@pytest.mark.parametrize("chunk0", [1, 3, 4, 5, 7, 16, 1000])
def test_write_level_chunk_boundary_exact(tmp_path, chunk0):
    data = np.arange(37, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    plan, sizing = _fold_plan(1, data.shape[0], chunk0)
    write_level(group, _raw_blocks(group, data), plan, sizing, 5)
    level = open_group(tmp_path / "bundle")["1"]
    expected = fold_raw_block(data)
    assert np.array_equal(level["env"][:], _env(expected))
    assert np.array_equal(level["mean"][:], _mean(expected))


def test_write_level_sets_period_us_on_the_group(tmp_path):
    """period_us belongs to the level, not to any one member of it."""
    data = np.arange(20, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    plan, sizing = _fold_plan(1, data.shape[0], 4)
    write_level(group, _raw_blocks(group, data), plan, sizing, 5)
    level = open_group(tmp_path / "bundle")["1"]
    assert dict(level.attrs) == {"period_us": 125.0}
    assert dict(level["env"].attrs) == {}
    assert dict(level["mean"].attrs) == {}
    assert dict(level["valid"].attrs) == {}


def test_write_level_returns_every_member(tmp_path):
    data = np.arange(20, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    plan, sizing = _fold_plan(1, data.shape[0], 4)
    env, mean, valid = write_level(
        group, _raw_blocks(group, data), plan, sizing, 5
    )
    assert all(isinstance(a, Array) for a in (env, mean, valid))
    assert env.shape == (5, 2)
    assert mean.shape == (5,)
    assert valid.shape == (5,)
    assert valid.dtype == np.uint16


def test_write_level_empty_prev(tmp_path):
    data = np.empty((0,), dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    plan, sizing = _fold_plan(1, 0, 4)
    write_level(group, _raw_blocks(group, data), plan, sizing, 5)
    level = open_group(tmp_path / "bundle")["1"]
    assert level["env"].shape == (0, 2)
    assert level["mean"].shape == (0,)
    assert level["valid"].shape == (0,)


def test_write_level_rejects_a_level_below_one(tmp_path):
    data = np.arange(20, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    sizing = ChunkShard(chunk_shape=(4, 2), shard_shape=(4, 2))
    plan = LevelPlan(level=0, shape=(5,), period_us=31.25)
    with pytest.raises(ValueError, match="raw is not a level"):
        write_level(group, _raw_blocks(group, data), plan, sizing, 5)


_MULTI_OPTS = WriteOpts(
    min_bins=2, max_levels=7, inner_len=16, target_shard_bytes=256
)


def test_write_continuous_channel_creates_subgroup_with_attrs(
    tmp_path, continuous_source
):
    samples = np.arange(64, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    src = continuous_source(samples, id="N:ch:xyz", rate_hz=32000.0, start_us=7)
    write_continuous_channel(parent, 3, src, onset_us=0, opts=_MULTI_OPTS)
    grp = open_group(tmp_path / "bundle")["3"]
    assert dict(grp.attrs) == channel_group_attrs(
        "N:ch:xyz", 32000.0, 7, "continuous", src.name, src.unit
    )


def test_write_continuous_channel_raw_round_trips(tmp_path, continuous_source):
    samples = np.arange(64, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(
        parent, 0, continuous_source(samples), onset_us=0, opts=_MULTI_OPTS
    )
    stored = open_group(tmp_path / "bundle")["0"]["raw"][:]
    assert np.array_equal(stored, samples)


def test_write_continuous_channel_level1_folds_from_raw(
    tmp_path, continuous_source
):
    samples = np.arange(64, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(
        parent, 0, continuous_source(samples), onset_us=0, opts=_MULTI_OPTS
    )
    grp = open_group(tmp_path / "bundle")["0"]
    expected = fold_raw_block(grp["raw"][:])
    assert np.array_equal(grp["1"]["env"][:], _env(expected))
    assert np.array_equal(grp["1"]["mean"][:], _mean(expected))
    assert "valid" not in grp["1"].array_keys()


def test_write_continuous_channel_each_level_folds_from_below(
    tmp_path, continuous_source
):
    samples = np.arange(64, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(
        parent, 0, continuous_source(samples), onset_us=0, opts=_MULTI_OPTS
    )
    grp = open_group(tmp_path / "bundle")["0"]
    n_levels = len(plan_levels(64, 31.25, 7, 2))
    assert n_levels >= 2
    for k in range(2, n_levels + 1):
        below = grp[str(k - 1)]["env"][:]
        expected = np.stack(
            [
                below.reshape(-1, 4, 2)[:, :, 0].min(axis=1),
                below.reshape(-1, 4, 2)[:, :, 1].max(axis=1),
            ],
            axis=1,
        )
        assert np.array_equal(grp[str(k)]["env"][:], expected)


def test_write_continuous_channel_levels_are_groups_not_arrays(
    tmp_path, continuous_source
):
    """Numeric keys are level groups; raw is the channel's only array."""
    samples = np.arange(64, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(
        parent, 0, continuous_source(samples), onset_us=0, opts=_MULTI_OPTS
    )
    grp = open_group(tmp_path / "bundle")["0"]
    plans = plan_levels(64, 31.25, 7, 2)
    assert list(grp.array_keys()) == ["raw"]
    assert sorted(grp.group_keys()) == sorted(str(p.level) for p in plans)


def test_write_continuous_channel_level_members_and_periods(
    tmp_path, continuous_source
):
    samples = np.arange(64, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(
        parent, 0, continuous_source(samples), onset_us=0, opts=_MULTI_OPTS
    )
    grp = open_group(tmp_path / "bundle")["0"]
    for p in plan_levels(64, 31.25, 7, 2):
        level = grp[str(p.level)]
        assert dict(level.attrs) == {"period_us": p.period_us}
        assert sorted(level.array_keys()) == ["env", "mean"]
        assert level["env"].shape == p.shape
        assert level["mean"].shape == (p.shape[0],)


def test_write_continuous_channel_short_source_gets_raw_and_no_levels(
    tmp_path, continuous_source
):
    samples = np.arange(3, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(
        parent, 0, continuous_source(samples), onset_us=0, opts=WriteOpts()
    )
    grp = open_group(tmp_path / "bundle")["0"]
    assert list(grp.array_keys()) == ["raw"]
    assert list(grp.group_keys()) == []
    assert np.array_equal(grp["raw"][:], samples)


def test_write_continuous_channel_empty_source_writes_empty_raw(
    tmp_path, continuous_source
):
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(
        parent, 0, continuous_source([]), onset_us=0, opts=WriteOpts()
    )
    grp = open_group(tmp_path / "bundle")["0"]
    assert list(grp.array_keys()) == ["raw"]
    assert list(grp.group_keys()) == []
    assert grp["raw"].shape == (0,)


def test_write_continuous_channel_all_zero_source_writes_every_shard(
    tmp_path, continuous_source
):
    samples = np.zeros(64, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(
        parent, 0, continuous_source(samples), onset_us=0, opts=_MULTI_OPTS
    )
    channel_dir = tmp_path / "bundle" / "0"
    keys = ["raw", *(str(p.level) for p in plan_levels(64, 31.25, 7, 2))]
    assert len(keys) >= 3
    for key in keys:
        member_dir = channel_dir / key
        shard_files = [
            f
            for f in member_dir.rglob("*")
            if f.is_file() and f.name != "zarr.json"
        ]
        assert shard_files, f"{key} has no shard file"


def test_write_continuous_channel_returns_none(tmp_path, continuous_source):
    samples = np.arange(64, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    result = write_continuous_channel(
        parent, 0, continuous_source(samples), onset_us=0, opts=_MULTI_OPTS
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


def test_write_raw_writes_one_whole_shard_per_write(
    tmp_path, continuous_source, monkeypatch
):
    writes = _record_writes(monkeypatch, "timeseries_zarr.write_continuous")
    samples = np.arange(26, dtype=np.float32)
    group = open_group(tmp_path / "bundle")
    write_raw(group, continuous_source(samples), _sizing(), 5)
    assert writes == [(0, 8), (8, 8), (16, 8), (24, 2)]
    assert np.array_equal(open_group(tmp_path / "bundle")["raw"][:], samples)


def test_write_level_writes_one_whole_shard_per_member_per_write(
    tmp_path, monkeypatch
):
    """env and mean share a row geometry, so both land on shard boundaries."""
    group = open_group(tmp_path / "bundle")
    data = np.arange(64, dtype=np.float32)
    # Read 4 shards of raw at a time, so one folded block fills one shard.
    blocks = _raw_blocks(group, data, block_len=32)
    plan = LevelPlan(level=1, shape=(16, 2), period_us=125.0)
    sizing = ChunkShard(chunk_shape=(4, 2), shard_shape=(8, 2))
    writes = _record_writes(monkeypatch, "timeseries_zarr.write_continuous")
    write_level(group, blocks, plan, sizing, 5)
    # One write per member per block, all at the same offsets.
    assert writes == [(0, 8)] * 3 + [(8, 8)] * 3
    level = open_group(tmp_path / "bundle")["1"]
    expected = fold_raw_block(data)
    assert np.array_equal(level["env"][:], _env(expected))
    assert np.array_equal(level["mean"][:], _mean(expected))
