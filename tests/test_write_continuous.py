import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from zarr import Array

from tests.conftest import ArrayContinuousSource, ArrayMultiChannelSource
from timeseries_zarr.constants import (
    COUNT_KEY,
    MEAN_KEY,
    MINMAX_KEY,
    RAW_KEY,
    VAR_KEY,
)
from timeseries_zarr.fold import fold_raw_block, split_members
from timeseries_zarr.planning import level_period_us, plan_levels
from timeseries_zarr.sizing import level_sizing, raw_sizing
from timeseries_zarr.streaming import scale_block
from timeseries_zarr.types import ChunkShard, LevelPlan, WriteOpts
from timeseries_zarr.write_continuous import (
    create_channel_group,
    create_level,
    create_raw,
    write_continuous_channel,
    write_continuous_channels,
    write_levels,
    write_raw,
    write_statistics_blocks,
)
from timeseries_zarr.zarr_io import open_group, write_region


def _sizing():
    return ChunkShard(chunk_shape=(4,), shard_shape=(8,))


def _record_writes(monkeypatch, module):
    """Record (start, rows) for every write_region call made by module."""
    writes = []

    def spy(array, start, block):
        writes.append((start, block.shape[0]))
        write_region(array, start, block)

    monkeypatch.setattr(f"{module}.write_region", spy)
    return writes


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


def test_create_channel_group_carries_the_continuous_attributes_and_offset(
    tmp_path: Path, continuous_source: type[ArrayContinuousSource]
) -> None:
    channel = continuous_source(
        np.arange(64, dtype=np.float32),
        id="N:ch:xyz",
        rate_hz=32000.0,
        start_us=7,
        name="ch name",
        unit="volts",
        provenance={"software": "acquire", "gain": 2},
    )
    parent = open_group(tmp_path / "bundle")
    create_channel_group(parent, 3, channel, reference_us=5, unit="uV")
    group = open_group(tmp_path / "bundle")["3"]
    assert dict(group.attrs) == {
        "tsz": {
            "kind": "continuous",
            "name": "ch name",
            "id": "N:ch:xyz",
            "rate_hz": 32000.0,
            "offset_us": 2,
            "unit": "uV",
            "provenance": {"software": "acquire", "gain": 2},
        }
    }
    assert list(group.array_keys()) == []


def test_create_raw_declares_shape_dtype_and_time_dimension(
    tmp_path: Path,
) -> None:
    group = open_group(tmp_path / "bundle")
    created = create_raw(group, 26, _sizing(), 5)
    stored = open_group(tmp_path / "bundle")[RAW_KEY]
    assert created.shape == (26,)
    assert stored.shape == (26,)
    assert stored.dtype == np.float32
    assert stored.metadata.dimension_names == ("time",)
    assert (stored.chunks, stored.shards) == ((4,), (8,))
    assert dict(stored.attrs) == {}
    assert np.isnan(stored[:]).all()


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


_OPTS = WriteOpts(min_bins=2, max_levels=8, inner_len=8, target_shard_bytes=128)
_OPTS_ONE_PASS = dataclasses.replace(_OPTS, write_var=True)


_WIDE = WriteOpts(
    min_bins=2, max_levels=8, inner_len=32, target_shard_bytes=4096
)
_WIDE_ONE_PASS = dataclasses.replace(_WIDE, write_var=True)


@pytest.fixture(params=[_OPTS, _OPTS_ONE_PASS], ids=["read_back", "one_pass"])
def opts(request):
    return request.param


@pytest.fixture(params=[_WIDE, _WIDE_ONE_PASS], ids=["read_back", "one_pass"])
def wide_opts(request):
    """Options with shards of 512 rows, which keep long series quick to write."""
    return request.param


def _series(n, offset=5000.0, scale=8.0, seed=0):
    rng = np.random.default_rng(seed)
    raw = (offset + scale * rng.standard_normal(n)).astype(np.float32)
    raw[5] = np.nan
    raw[40] = np.inf
    raw[100:140] = np.nan
    return raw


def _direct(raw, width):
    """Per-bin statistics straight from the definition, in float64."""
    values = raw.astype(np.float64)
    finite = np.isfinite(values)
    values[~finite] = np.nan
    rows = []
    for i in range(0, len(values), width):
        block, ok = values[i : i + width], finite[i : i + width]
        rows.append(
            [block.min(), block.max(), block.mean(), block.var(), ok.sum()]
        )
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def _close(got, want):
    """Compare to a relative 1e-5 with the generator's absolute floor."""
    want = np.asarray(want, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    if not np.array_equal(np.isnan(got), np.isnan(want)):
        return False
    finite = np.isfinite(want)
    if not finite.any():
        return True
    atol = 1e-7 * np.max(np.abs(want[finite]))
    return bool(np.allclose(got[finite], want[finite], rtol=1e-5, atol=atol))


def test_write_levels_writes_every_planned_level_from_the_stored_raw(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    opts: WriteOpts,
) -> None:
    samples = _series(1000)
    parent = open_group(tmp_path / "bundle")
    sizing = raw_sizing(1000, opts.inner_len, opts.target_shard_bytes)
    raw = write_raw(parent, continuous_source(samples), sizing, opts.zstd_level)
    write_levels(parent, raw, 32000.0, opts)
    group = open_group(tmp_path / "bundle")
    plans = plan_levels(1000, 32000.0, opts.max_levels, opts.min_bins)
    assert len(plans) >= 2
    assert sorted(group.group_keys()) == sorted(str(p.level) for p in plans)
    for plan in plans:
        want = _direct(samples, 4**plan.level)
        level = group[str(plan.level)]
        assert dict(level.attrs) == {
            "tsz": {"period_us": level_period_us(32000.0, plan.level)}
        }
        assert np.array_equal(
            level[MINMAX_KEY][:], want[:, :2].astype(np.float32), equal_nan=True
        )
        assert np.array_equal(level[COUNT_KEY][:], want[:, 4])
        assert _close(level[MEAN_KEY][:], want[:, 2]), plan.level
        if opts.write_var:
            assert _close(level[VAR_KEY][:], want[:, 3]), plan.level
        else:
            assert VAR_KEY not in list(level.array_keys())


def test_write_levels_writes_nothing_for_a_channel_too_short_for_a_level(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    opts: WriteOpts,
) -> None:
    samples = np.arange(3, dtype=np.float32)
    parent = open_group(tmp_path / "bundle")
    sizing = raw_sizing(3, opts.inner_len, opts.target_shard_bytes)
    raw = write_raw(parent, continuous_source(samples), sizing, opts.zstd_level)
    write_levels(parent, raw, 32000.0, opts)
    group = open_group(tmp_path / "bundle")
    assert list(group.array_keys()) == [RAW_KEY]
    assert list(group.group_keys()) == []


def _write(tmp_path, source, opts, reference_us=0, name="bundle"):
    parent = open_group(tmp_path / name)
    write_continuous_channel(
        parent, 0, source, reference_us=reference_us, opts=opts
    )
    return open_group(tmp_path / name)["0"]


def test_write_continuous_channel_attributes_sit_under_tsz_with_the_offset(
    tmp_path, continuous_source
):
    src = continuous_source(
        np.arange(64, dtype=np.float32),
        id="N:ch:xyz",
        rate_hz=32000.0,
        start_us=7,
        name="ch name",
        unit="uV",
    )
    group = _write(tmp_path, src, _OPTS, reference_us=5)
    assert dict(group.attrs) == {
        "tsz": {
            "kind": "continuous",
            "name": "ch name",
            "id": "N:ch:xyz",
            "rate_hz": 32000.0,
            "offset_us": 2,
            "unit": "uV",
        }
    }


def test_write_continuous_channel_offset_is_zero_at_the_reference_time(
    tmp_path, continuous_source
):
    src = continuous_source(np.arange(64, dtype=np.float32), start_us=9)
    group = _write(tmp_path, src, _OPTS, reference_us=9)
    assert dict(group.attrs)["tsz"]["offset_us"] == 0


_PROVENANCE = {
    "software": "acquire 4.1",
    "settings": {"gain": 2, "bands": [1, 70]},
}


def test_write_continuous_channel_writes_the_source_provenance_under_tsz(
    tmp_path, continuous_source
):
    src = continuous_source(
        np.arange(64, dtype=np.float32), provenance=_PROVENANCE
    )
    group = _write(tmp_path, src, _OPTS)
    assert dict(group.attrs)["tsz"]["provenance"] == _PROVENANCE


def test_write_continuous_channel_writes_every_other_attribute_beside_provenance(
    tmp_path, continuous_source
):
    src = continuous_source(
        np.arange(64, dtype=np.float32),
        id="N:ch:xyz",
        rate_hz=32000.0,
        start_us=7,
        name="ch name",
        unit="uV",
        provenance=_PROVENANCE,
    )
    group = _write(tmp_path, src, _OPTS, reference_us=5)
    assert dict(group.attrs) == {
        "tsz": {
            "kind": "continuous",
            "name": "ch name",
            "id": "N:ch:xyz",
            "rate_hz": 32000.0,
            "offset_us": 2,
            "unit": "uV",
            "provenance": _PROVENANCE,
        }
    }


def test_write_continuous_channel_writes_no_provenance_key_without_provenance(
    tmp_path, continuous_source
):
    src = continuous_source(np.arange(64, dtype=np.float32))
    group = _write(tmp_path, src, _OPTS)
    assert "provenance" not in dict(group.attrs)["tsz"]


def test_write_continuous_channel_keeps_provenance_off_the_arrays_and_levels(
    tmp_path, continuous_source, opts
):
    src = continuous_source(
        np.arange(256, dtype=np.float32), provenance=_PROVENANCE
    )
    group = _write(tmp_path, src, opts)
    levels = [group[key] for key in sorted(group.group_keys())]
    assert len(levels) >= 2
    nodes = [group[RAW_KEY], *levels]
    nodes += [level[key] for level in levels for key in level.array_keys()]
    assert all("provenance" not in str(dict(node.attrs)) for node in nodes)


def test_write_continuous_channel_writes_provenance_on_a_channel_with_no_level(
    tmp_path, continuous_source
):
    src = continuous_source(
        np.arange(3, dtype=np.float32), provenance=_PROVENANCE
    )
    group = _write(tmp_path, src, _OPTS)
    assert list(group.group_keys()) == []
    assert dict(group.attrs)["tsz"]["provenance"] == _PROVENANCE


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_write_continuous_channel_rejects_a_non_finite_provenance_number(
    tmp_path, continuous_source, value
):
    src = continuous_source(
        np.arange(64, dtype=np.float32), provenance={"threshold": value}
    )
    with pytest.raises(ValueError, match="provenance"):
        _write(tmp_path, src, _OPTS)


def test_write_continuous_channel_before_the_reference_time_has_a_negative_offset(
    tmp_path, continuous_source
):
    src = continuous_source(np.arange(64, dtype=np.float32), start_us=7)
    group = _write(tmp_path, src, _OPTS, reference_us=8)
    assert dict(group.attrs)["tsz"]["offset_us"] == -1


def test_write_continuous_channel_holds_raw_and_level_groups_from_one(
    tmp_path, continuous_source, opts
):
    samples = _series(1000)
    group = _write(tmp_path, continuous_source(samples), opts)
    levels = plan_levels(1000, 32000.0, opts.max_levels, opts.min_bins)
    assert list(group.array_keys()) == [RAW_KEY]
    assert sorted(group.group_keys()) == sorted(str(p.level) for p in levels)
    assert "0" not in list(group.group_keys()) + list(group.array_keys())


def test_write_continuous_channel_stores_the_raw_samples_unchanged(
    tmp_path, continuous_source, opts
):
    samples = _series(1000)
    group = _write(tmp_path, continuous_source(samples), opts)
    assert np.array_equal(group[RAW_KEY][:], samples, equal_nan=True)


def test_write_continuous_channel_level_groups_carry_period_and_bins(
    tmp_path, continuous_source, opts
):
    group = _write(tmp_path, continuous_source(_series(1000)), opts)
    for plan in plan_levels(1000, 32000.0, opts.max_levels, opts.min_bins):
        level = group[str(plan.level)]
        assert dict(level.attrs) == {
            "tsz": {"period_us": level_period_us(32000.0, plan.level)}
        }
        assert level[MINMAX_KEY].shape == (plan.bins, 2)
        assert level[MEAN_KEY].shape == (plan.bins,)
        assert level[COUNT_KEY].shape == (plan.bins,)


def test_write_continuous_channel_members_equal_the_direct_definition_over_six_levels(
    tmp_path, continuous_source, wide_opts
):
    samples = _series(8199)
    group = _write(tmp_path, continuous_source(samples), wide_opts)
    levels = plan_levels(
        8199, 32000.0, wide_opts.max_levels, wide_opts.min_bins
    )
    assert len(levels) == 6
    for plan in levels:
        want = _direct(samples, 4**plan.level)
        level = group[str(plan.level)]
        minmax = level[MINMAX_KEY][:]
        assert np.array_equal(
            minmax, want[:, :2].astype(np.float32), equal_nan=True
        )
        assert np.array_equal(level[COUNT_KEY][:], want[:, 4])
        assert _close(level[MEAN_KEY][:], want[:, 2]), plan.level
        if wide_opts.write_var:
            assert _close(level[VAR_KEY][:], want[:, 3]), plan.level


def test_write_continuous_channel_one_pass_variance_is_exact_beyond_a_large_offset(
    tmp_path, continuous_source
):
    samples = _series(8199, offset=5000.0, scale=1.0, seed=3)
    group = _write(tmp_path, continuous_source(samples), _WIDE_ONE_PASS)
    for plan in plan_levels(8199, 32000.0, 8, 2):
        want = _direct(samples, 4**plan.level)[:, 3]
        assert _close(group[str(plan.level)][VAR_KEY][:], want), plan.level


def test_write_continuous_channel_var_members_exist_only_when_write_var_is_on(
    tmp_path, continuous_source
):
    samples = _series(1000)
    without = _write(tmp_path, continuous_source(samples), _OPTS, name="a")
    with_var = _write(
        tmp_path, continuous_source(samples), _OPTS_ONE_PASS, name="b"
    )
    for plan in plan_levels(1000, 32000.0, 8, 2):
        assert VAR_KEY not in list(without[str(plan.level)].array_keys())
        assert VAR_KEY in list(with_var[str(plan.level)].array_keys())


def test_write_continuous_channel_both_modes_agree_on_minmax_count_and_mean(
    tmp_path, continuous_source
):
    samples = _series(8199)
    a = _write(tmp_path, continuous_source(samples), _WIDE, name="a")
    b = _write(tmp_path, continuous_source(samples), _WIDE_ONE_PASS, name="b")
    for plan in plan_levels(8199, 32000.0, 8, 2):
        left, right = a[str(plan.level)], b[str(plan.level)]
        assert np.array_equal(
            left[MINMAX_KEY][:], right[MINMAX_KEY][:], equal_nan=True
        )
        assert np.array_equal(left[COUNT_KEY][:], right[COUNT_KEY][:])
        assert _close(left[MEAN_KEY][:], right[MEAN_KEY][:])


def test_write_continuous_channel_too_short_for_a_level_gets_raw_only(
    tmp_path, continuous_source
):
    group = _write(
        tmp_path,
        continuous_source(np.arange(3, dtype=np.float32)),
        WriteOpts(),
    )
    assert list(group.array_keys()) == [RAW_KEY]
    assert list(group.group_keys()) == []


def test_write_continuous_channel_empty_source_writes_an_empty_raw_array(
    tmp_path, continuous_source, opts
):
    group = _write(tmp_path, continuous_source(np.empty(0, np.float32)), opts)
    assert group[RAW_KEY].shape == (0,)
    assert list(group.group_keys()) == []


def test_write_continuous_channel_all_zero_source_writes_every_shard(
    tmp_path, continuous_source, opts
):
    group = _write(
        tmp_path, continuous_source(np.zeros(1000, np.float32)), opts
    )
    arrays = [(RAW_KEY, group[RAW_KEY])]
    for level in group.group_keys():
        arrays += [(f"{level}/{k}", a) for k, a in group[level].arrays()]
    assert len(arrays) > 4
    for path, array in arrays:
        shards = [
            f
            for f in (tmp_path / "bundle" / "0" / path).rglob("*")
            if f.is_file() and f.name != "zarr.json"
        ]
        wanted = -(-array.shape[0] // array.shards[0])
        assert len(shards) == wanted, path


def test_write_continuous_channel_every_write_covers_a_whole_shard(
    tmp_path, continuous_source, monkeypatch, opts
):
    writes = []

    def spy(array, start, block):
        writes.append(
            (array.name, start, block.shape[0], array.shape[0], array.shards[0])
        )
        write_region(array, start, block)

    monkeypatch.setattr("timeseries_zarr.write_continuous.write_region", spy)
    _write(tmp_path, continuous_source(_series(1000)), opts)
    assert len(writes) > 10
    for name, start, rows, length, shard_rows in writes:
        assert start % shard_rows == 0, name
        assert rows == min(shard_rows, length - start), name


def test_write_continuous_channel_returns_none(tmp_path, continuous_source):
    parent = open_group(tmp_path / "bundle")
    result = write_continuous_channel(
        parent,
        0,
        continuous_source(np.arange(64, dtype=np.float32)),
        reference_us=0,
        opts=_OPTS,
    )
    assert result is None


def test_write_continuous_channel_scales_a_volts_source_to_microvolts(
    tmp_path, continuous_source, opts
):
    samples = _series(1000, offset=0.005, scale=0.001)
    group = _write(tmp_path, continuous_source(samples, unit="volts"), opts)
    want = samples * np.float32(1e6)
    assert group.attrs["tsz"]["unit"] == "uV"
    assert np.array_equal(group[RAW_KEY][:], want, equal_nan=True)


def test_write_continuous_channel_folds_levels_from_the_scaled_samples(
    tmp_path, continuous_source, opts
):
    samples = _series(1000, offset=0.005, scale=0.001)
    group = _write(tmp_path, continuous_source(samples, unit="volts"), opts)
    scaled = (samples * np.float32(1e6)).astype(np.float32)
    want = _direct(scaled, 4)[:, 2]
    assert _close(group["1"]["mean"][:], want)


def test_write_continuous_channel_leaves_an_unknown_unit_unchanged(
    tmp_path, continuous_source, opts
):
    samples = _series(1000)
    group = _write(tmp_path, continuous_source(samples, unit="pA"), opts)
    assert group.attrs["tsz"]["unit"] == "pA"
    assert np.array_equal(group[RAW_KEY][:], samples, equal_nan=True)


class _WindowSource:
    """A multi-channel source over prepared windows.

    It counts the calls to iter_windows.
    """

    def __init__(
        self, channels: Sequence[Any], windows: Sequence[Sequence[Any]]
    ) -> None:
        self._channels = list(channels)
        self._windows = list(windows)
        self.calls = 0

    def channels(self) -> list[Any]:
        return self._channels

    def iter_windows(self) -> Any:
        self.calls += 1
        return iter(self._windows)


class _MetadataChannel:
    """A channel that declares its metadata and has no read_samples."""

    def __init__(self, id: str, num_samples: int) -> None:
        self.id = id
        self.name = id
        self.unit = "uV"
        self._num_samples = num_samples

    def rate_hz(self) -> float:
        return 32000.0

    def start_us(self) -> int:
        return 0

    def num_samples(self) -> int:
        return self._num_samples


def _write_many(
    tmp_path: Path,
    source: Any,
    opts: WriteOpts,
    first_key: int = 0,
    reference_us: int = 0,
    name: str = "bundle",
) -> Any:
    """Write source below the root group at tmp_path / name and return it."""
    parent = open_group(tmp_path / name)
    write_continuous_channels(
        parent, first_key, source, reference_us=reference_us, opts=opts
    )
    return open_group(tmp_path / name)


def _files(root: Path) -> dict[str, bytes]:
    """Return the content of every file below root by relative path."""
    return {
        file.relative_to(root).as_posix(): file.read_bytes()
        for file in sorted(root.rglob("*"))
        if file.is_file()
    }


def _record_level_writes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the group name of every write_levels call."""
    folded: list[str] = []

    def spy(group: Any, raw: Any, rate_hz: float, opts: WriteOpts) -> None:
        folded.append(group.name)
        write_levels(group, raw, rate_hz, opts)

    monkeypatch.setattr("timeseries_zarr.write_continuous.write_levels", spy)
    return folded


def _assert_levels_equal_the_definition(
    group: Any, samples: np.ndarray, rate_hz: float, opts: WriteOpts
) -> None:
    plans = plan_levels(
        samples.shape[0], rate_hz, opts.max_levels, opts.min_bins
    )
    assert plans
    assert sorted(group.group_keys()) == sorted(str(p.level) for p in plans)
    for plan in plans:
        want = _direct(samples, 4**plan.level)
        level = group[str(plan.level)]
        assert dict(level.attrs) == {
            "tsz": {"period_us": level_period_us(rate_hz, plan.level)}
        }
        assert np.array_equal(
            level[MINMAX_KEY][:], want[:, :2].astype(np.float32), equal_nan=True
        )
        assert np.array_equal(level[COUNT_KEY][:], want[:, 4])
        assert _close(level[MEAN_KEY][:], want[:, 2]), plan.level
        if opts.write_var:
            assert _close(level[VAR_KEY][:], want[:, 3]), plan.level
        else:
            assert VAR_KEY not in list(level.array_keys())


def test_write_continuous_channels_stores_each_channels_raw_samples_unchanged(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    samples = [_series(1000, seed=1), _series(1000, seed=2), _series(256)]
    channels = [
        continuous_source(each, id=f"c{i}") for i, each in enumerate(samples)
    ]
    root = _write_many(tmp_path, multi_channel_source(channels), _OPTS)
    for index, expected in enumerate(samples):
        stored = root[str(index)][RAW_KEY][:]
        assert stored.dtype == np.float32
        assert np.array_equal(stored, expected, equal_nan=True)


def test_write_continuous_channels_names_the_groups_from_the_first_key_in_channel_order(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    channels = [
        continuous_source(
            np.arange(64, dtype=np.float32),
            id=name,
            name=f"name {name}",
            start_us=start_us,
        )
        for name, start_us in (("a", 10), ("b", 20), ("c", 30))
    ]
    root = _write_many(
        tmp_path,
        multi_channel_source(channels),
        _OPTS,
        first_key=4,
        reference_us=5,
    )
    assert sorted(root.group_keys()) == ["4", "5", "6"]
    stored = [dict(root[key].attrs)["tsz"] for key in ("4", "5", "6")]
    assert [(tsz["id"], tsz["name"]) for tsz in stored] == [
        ("a", "name a"),
        ("b", "name b"),
        ("c", "name c"),
    ]
    assert [tsz["offset_us"] for tsz in stored] == [5, 15, 25]


@pytest.mark.parametrize("window_rows", [1, 3, 7, 8, 13, 1000])
def test_write_continuous_channels_every_write_covers_a_whole_shard_whatever_the_window_length(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
    monkeypatch: pytest.MonkeyPatch,
    opts: WriteOpts,
    window_rows: int,
) -> None:
    writes: list[tuple[str, int, int, int, int]] = []

    def spy(array: Any, start: int, block: np.ndarray) -> None:
        writes.append(
            (array.name, start, block.shape[0], array.shape[0], array.shards[0])
        )
        write_region(array, start, block)

    monkeypatch.setattr("timeseries_zarr.write_continuous.write_region", spy)
    samples = [_series(1000, seed=1), _series(700, seed=2)]
    channels = [
        continuous_source(each, id=f"c{i}") for i, each in enumerate(samples)
    ]
    root = _write_many(
        tmp_path, multi_channel_source(channels, window_rows), opts
    )
    assert len(writes) > 10
    for name, start, rows, length, shard_rows in writes:
        assert start % shard_rows == 0, name
        assert rows == min(shard_rows, length - start), name
    for index, expected in enumerate(samples):
        shard_rows = raw_sizing(
            expected.shape[0], opts.inner_len, opts.target_shard_bytes
        ).shard_shape[0]
        raw_starts = [
            start for name, start, *_ in writes if name == f"/{index}/raw"
        ]
        assert raw_starts == list(range(0, expected.shape[0], shard_rows))
        stored = root[str(index)][RAW_KEY][:]
        assert np.array_equal(stored, expected, equal_nan=True)


def test_write_continuous_channels_scales_each_channel_by_its_own_unit(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    samples = _series(300, offset=0.005, scale=0.001)
    cases = [
        ("volts", 1e6, "uV"),
        ("mV", 1e3, "uV"),
        ("uV", 1.0, "uV"),
        ("pA", 1.0, "pA"),
    ]
    channels = [
        continuous_source(samples, id=f"c{i}", unit=unit)
        for i, (unit, _, _) in enumerate(cases)
    ]
    root = _write_many(tmp_path, multi_channel_source(channels), _OPTS)
    for index, (_, factor, stored_unit) in enumerate(cases):
        group = root[str(index)]
        want = samples * np.float32(factor) if factor != 1.0 else samples
        assert group.attrs["tsz"]["unit"] == stored_unit
        assert np.array_equal(group[RAW_KEY][:], want, equal_nan=True)


def test_write_continuous_channels_writes_each_channels_provenance_on_its_group(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    channels = [
        continuous_source(
            np.arange(64, dtype=np.float32), id="a", provenance=_PROVENANCE
        ),
        continuous_source(np.arange(64, dtype=np.float32), id="b"),
    ]
    root = _write_many(tmp_path, multi_channel_source(channels), _OPTS)
    assert dict(root["0"].attrs)["tsz"]["provenance"] == _PROVENANCE
    assert "provenance" not in dict(root["1"].attrs)["tsz"]


def test_write_continuous_channels_writes_an_empty_raw_array_for_a_channel_without_samples(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
    opts: WriteOpts,
) -> None:
    samples = _series(200)
    channels = [
        continuous_source(np.empty(0, np.float32), id="empty"),
        continuous_source(samples, id="full"),
    ]
    root = _write_many(tmp_path, multi_channel_source(channels), opts)
    assert root["0"][RAW_KEY].shape == (0,)
    assert list(root["0"].group_keys()) == []
    assert np.array_equal(root["1"][RAW_KEY][:], samples, equal_nan=True)


def test_write_continuous_channels_accepts_channels_of_different_lengths_and_rates(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    lengths = [1000, 333, 700]
    rates = [32000.0, 1000.0, 250.0]
    samples = [_series(n, seed=n) for n in lengths]
    channels = [
        continuous_source(each, id=f"c{i}", rate_hz=rate)
        for i, (each, rate) in enumerate(zip(samples, rates))
    ]
    source = multi_channel_source(channels, window_rows=[7, 5, 13])
    root = _write_many(tmp_path, source, _OPTS)
    for index, (each, rate) in enumerate(zip(samples, rates)):
        group = root[str(index)]
        plans = plan_levels(
            each.shape[0], rate, _OPTS.max_levels, _OPTS.min_bins
        )
        assert group.attrs["tsz"]["rate_hz"] == rate
        assert np.array_equal(group[RAW_KEY][:], each, equal_nan=True)
        assert sorted(group.group_keys()) == sorted(str(p.level) for p in plans)


def test_write_continuous_channels_folds_the_levels_of_each_channel(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
    opts: WriteOpts,
) -> None:
    samples = [_series(1000, seed=1), _series(600, seed=2)]
    rates = [32000.0, 1000.0]
    channels = [
        continuous_source(each, id=f"c{i}", rate_hz=rate)
        for i, (each, rate) in enumerate(zip(samples, rates))
    ]
    source = multi_channel_source(channels, window_rows=[11, 40])
    root = _write_many(tmp_path, source, opts)
    for index, (each, rate) in enumerate(zip(samples, rates)):
        _assert_levels_equal_the_definition(root[str(index)], each, rate, opts)


def test_write_continuous_channels_matches_write_continuous_channel_for_each_channel(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
    opts: WriteOpts,
) -> None:
    channels = [
        continuous_source(
            _series(1000, seed=1),
            id="a",
            start_us=10,
            provenance=_PROVENANCE,
        ),
        continuous_source(
            _series(333, offset=0.005, scale=0.001, seed=2),
            id="b",
            rate_hz=1000.0,
            start_us=50,
            unit="volts",
        ),
        continuous_source(
            _series(700, seed=3), id="c", rate_hz=250.0, unit="pA"
        ),
    ]
    single = open_group(tmp_path / "single")
    for key, channel in enumerate(channels):
        write_continuous_channel(
            single, key, channel, reference_us=10, opts=opts
        )
    source = multi_channel_source(channels, window_rows=[7, 5, 13])
    _write_many(tmp_path, source, opts, reference_us=10, name="sweep")
    left, right = _files(tmp_path / "single"), _files(tmp_path / "sweep")
    assert len(left) > 20
    assert left.keys() == right.keys()
    assert [name for name in left if left[name] != right[name]] == []


@pytest.mark.parametrize("window_rows", [7, 100])
def test_write_continuous_channels_accepts_strided_column_views(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    window_rows: int,
) -> None:
    matrix = np.stack([_series(300, seed=i) for i in range(3)], axis=1)
    channels = [continuous_source(matrix[:, i], id=f"c{i}") for i in range(3)]
    windows = [
        list(matrix[start : start + window_rows].T)
        for start in range(0, 300, window_rows)
    ]
    assert not windows[0][0].flags["C_CONTIGUOUS"]
    root = _write_many(tmp_path, _WindowSource(channels, windows), _OPTS)
    for index in range(3):
        stored = root[str(index)][RAW_KEY][:]
        assert np.array_equal(stored, matrix[:, index], equal_nan=True)


def test_write_continuous_channels_casts_a_float64_block_to_float32(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
) -> None:
    wide = np.random.default_rng(3).standard_normal(300)
    narrow = wide.astype(np.float32)
    channels = [continuous_source(narrow, id="a", unit="volts")]
    windows = [[wide[:100]], [wide[100:]]]
    root = _write_many(tmp_path, _WindowSource(channels, windows), _OPTS)
    stored = root["0"][RAW_KEY][:]
    assert stored.dtype == np.float32
    assert np.array_equal(stored, narrow * np.float32(1e6))


def test_write_continuous_channels_needs_only_channel_metadata(
    tmp_path: Path,
) -> None:
    channels = [_MetadataChannel("a", 40), _MetadataChannel("b", 10)]
    samples = [np.arange(40, dtype=np.float32), np.arange(10, dtype=np.float32)]
    windows = [[samples[0][:25], samples[1]], [samples[0][25:], samples[1][:0]]]
    root = _write_many(tmp_path, _WindowSource(channels, windows), _OPTS)
    assert np.array_equal(root["0"][RAW_KEY][:], samples[0])
    assert np.array_equal(root["1"][RAW_KEY][:], samples[1])


def test_write_continuous_channels_reads_the_windows_once(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
) -> None:
    samples = _series(300)
    channels = [continuous_source(samples, id="a")]
    windows = [[samples[:100]], [samples[100:]]]
    source = _WindowSource(channels, windows)
    _write_many(tmp_path, source, _OPTS)
    assert source.calls == 1


def test_write_continuous_channels_with_no_channels_writes_nothing_and_reads_no_window(
    tmp_path: Path,
) -> None:
    source = _WindowSource([], [])
    root = _write_many(tmp_path, source, _OPTS, first_key=5)
    assert list(root.group_keys()) == []
    assert list(root.array_keys()) == []
    assert source.calls == 0


@pytest.mark.parametrize("count", [1, 3])
def test_write_continuous_channels_rejects_a_window_with_the_wrong_block_count(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> None:
    writes = _record_writes(monkeypatch, "timeseries_zarr.write_continuous")
    channels = [
        continuous_source(_series(1000, seed=1), id="a"),
        continuous_source(_series(1000, seed=2), id="b"),
    ]
    short = np.zeros(8, dtype=np.float32)
    long = np.zeros(100, dtype=np.float32)
    source = _WindowSource(channels, [[short, short], [long] * count])
    with pytest.raises(
        ValueError,
        match=rf"window 1 holds {count} blocks and the source has 2 channels",
    ):
        _write_many(tmp_path, source, _OPTS)
    assert writes == []


@pytest.mark.parametrize(
    ("shape", "rank"), [((4, 2), 2), ((), 0)], ids=["matrix", "scalar"]
)
def test_write_continuous_channels_rejects_a_block_that_is_not_rank_one(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    shape: tuple[int, ...],
    rank: int,
) -> None:
    channels = [
        continuous_source(_series(1000, seed=1), id="a"),
        continuous_source(_series(1000, seed=2), id="b"),
    ]
    good = np.zeros(8, dtype=np.float32)
    source = _WindowSource(channels, [[good, np.zeros(shape, np.float32)]])
    with pytest.raises(
        ValueError, match=rf"channel 4 \('b'\) gives a block of rank {rank}"
    ):
        _write_many(tmp_path, source, _OPTS, first_key=3)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (
            (1024, 1000),
            r"channel 3 \('short'\) yields 1000 samples and declares 1024",
        ),
        (None, r"channel 2 \('ok'\) yields 0 samples and declares 1024"),
    ],
    ids=["short window", "no window"],
)
def test_write_continuous_channels_rejects_a_channel_with_too_few_samples(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    monkeypatch: pytest.MonkeyPatch,
    counts: tuple[int, int] | None,
    message: str,
) -> None:
    folded = _record_level_writes(monkeypatch)
    channels = [
        continuous_source(_series(1024, seed=1), id="ok"),
        continuous_source(_series(1024, seed=2), id="short"),
    ]
    windows = (
        []
        if counts is None
        else [
            [
                channel.read_samples(0, count)
                for channel, count in zip(channels, counts)
            ]
        ]
    )
    with pytest.raises(ValueError, match=message):
        _write_many(
            tmp_path, _WindowSource(channels, windows), _OPTS, first_key=2
        )
    assert folded == []


@pytest.mark.parametrize(
    ("declared", "sizes", "expected_writes", "received"),
    [(40, (32, 16), [(0, 32)], 48), (40, (41,), [], 41), (0, (1,), [], 1)],
    ids=["after a shard", "in one block", "no samples declared"],
)
def test_write_continuous_channels_rejects_too_many_samples_before_writing_past_the_array(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    monkeypatch: pytest.MonkeyPatch,
    declared: int,
    sizes: tuple[int, ...],
    expected_writes: list[tuple[int, int]],
    received: int,
) -> None:
    writes = _record_writes(monkeypatch, "timeseries_zarr.write_continuous")
    folded = _record_level_writes(monkeypatch)
    scaled: list[int] = []

    def spy_scale(block: np.ndarray, factor: float) -> np.ndarray:
        scaled.append(block.shape[0])
        return scale_block(block, factor)

    monkeypatch.setattr(
        "timeseries_zarr.write_continuous.scale_block", spy_scale
    )
    channel = continuous_source(np.zeros(declared, np.float32), id="x")
    windows = [[np.ones(size, np.float32)] for size in sizes]
    with pytest.raises(
        ValueError,
        match=rf"channel 0 \('x'\) yields at least {received} samples "
        rf"and declares {declared}",
    ):
        _write_many(tmp_path, _WindowSource([channel], windows), _OPTS)
    assert writes == expected_writes
    assert all(start + rows <= declared for start, rows in writes)
    assert scaled == list(sizes[:-1])
    assert folded == []


def test_write_continuous_channels_rejects_a_volts_unit_under_require_uv(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
) -> None:
    channels = [
        continuous_source(np.arange(64, dtype=np.float32), id="a", unit="uV"),
        continuous_source(
            np.arange(64, dtype=np.float32), id="b", unit="volts"
        ),
    ]
    source = _WindowSource(channels, [])
    strict = dataclasses.replace(_OPTS, voltage_policy="require_uv")
    with pytest.raises(ValueError, match="voltage unit"):
        _write_many(tmp_path, source, strict)
    assert source.calls == 0


def test_write_continuous_channels_returns_none(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    parent = open_group(tmp_path / "bundle")
    source = multi_channel_source(
        [continuous_source(np.arange(64, dtype=np.float32))]
    )
    result = write_continuous_channels(
        parent, 0, source, reference_us=0, opts=_OPTS
    )
    assert result is None
