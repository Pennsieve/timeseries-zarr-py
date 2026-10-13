import math

import numpy as np
import pytest
from zarr import Array

from timeseries_zarr.attrs import channel_group_attrs
from timeseries_zarr.fold import fold_block
from timeseries_zarr.planning import plan_levels, sample_period_us
from timeseries_zarr.types import ChunkShard, LevelPlan, WriteOpts
from timeseries_zarr.write_continuous import (
    write_continuous_channel,
    write_level0,
    write_level_from_previous,
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
