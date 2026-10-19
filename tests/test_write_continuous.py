import dataclasses

import numpy as np
import pytest
from zarr import Array

from timeseries_zarr.constants import (
    COUNT_KEY,
    MEAN_KEY,
    MINMAX_KEY,
    RAW_KEY,
    VAR_KEY,
)
from timeseries_zarr.fold import fold_raw_block, split_members
from timeseries_zarr.planning import level_period_us, plan_levels
from timeseries_zarr.sizing import level_sizing
from timeseries_zarr.types import ChunkShard, LevelPlan, WriteOpts
from timeseries_zarr.write_continuous import (
    create_level,
    write_continuous_channel,
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
