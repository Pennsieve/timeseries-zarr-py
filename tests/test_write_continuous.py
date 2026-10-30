import dataclasses
import resource
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.conftest import ArrayContinuousSource, ArrayMultiChannelSource
from timeseries_zarr.bundle import write_bundle
from timeseries_zarr.constants import (
    COUNT_KEY,
    MEAN_KEY,
    MINMAX_KEY,
    RAW_KEY,
    VAR_KEY,
)
from timeseries_zarr.planning import level_period_us, plan_levels
from timeseries_zarr.shards import ArrayWriter, FileShardSink
from timeseries_zarr.sizing import level_sizing
from timeseries_zarr.streaming import scale_block
from timeseries_zarr.types import ChunkShard, LevelPlan, WriteOpts
from timeseries_zarr.write_continuous import (
    ChannelWriter,
    create_channel_group,
    create_level,
    create_raw,
    write_continuous_channel,
    write_continuous_channels,
)
from timeseries_zarr.zarr_io import create_nodes, open_group


def _sizing():
    return ChunkShard(chunk_shape=(4,), shard_shape=(8,))


def _record_shard_files(monkeypatch):
    """Record the path of every shard file as it closes, and the bytes of each."""
    closed = []
    pending = {}
    append = FileShardSink.append
    close = FileShardSink.close

    def spy_append(self, data):
        pending[self._path] = pending.get(self._path, 0) + len(data)
        append(self, data)

    def spy_close(self):
        closed.append((self._path, pending.pop(self._path, 0)))
        close(self)

    monkeypatch.setattr(FileShardSink, "append", spy_append)
    monkeypatch.setattr(FileShardSink, "close", spy_close)
    return closed


def _gappy_samples(n=26):
    rng = np.random.default_rng(0)
    samples = (5000.0 + 8.0 * rng.standard_normal(n)).astype(np.float32)
    samples[3] = np.nan
    samples[10] = np.inf
    samples[11] = -np.inf
    return samples


def _write_raw_only(tmp_path, source):
    """Write source as channel 0 with no levels and return its group."""
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(
        parent,
        0,
        source,
        reference_us=0,
        opts=dataclasses.replace(_OPTS, max_levels=0),
    )
    return open_group(tmp_path / "bundle")["0"]


def test_write_continuous_channel_round_trips_samples_unchanged(
    tmp_path, continuous_source
):
    samples = _gappy_samples()
    stored = _write_raw_only(tmp_path, continuous_source(samples))[RAW_KEY][:]
    assert stored.dtype == np.float32
    assert np.array_equal(stored, samples, equal_nan=True)


def test_write_continuous_channel_with_no_levels_holds_one_array_named_raw_with_no_attributes(
    tmp_path, continuous_source
):
    group = _write_raw_only(tmp_path, continuous_source(_gappy_samples()))
    assert list(group.array_keys()) == ["raw"]
    assert dict(group["raw"].attrs) == {}


def test_write_continuous_channel_declares_raw_shape_time_dimension_and_nan_fill(
    tmp_path, continuous_source
):
    arr = _write_raw_only(tmp_path, continuous_source(_gappy_samples()))["raw"]
    assert arr.shape == (26,)
    assert arr.metadata.dimension_names == ("time",)
    assert np.isnan(arr.fill_value)


def test_write_continuous_channel_raw_chunk_and_shard_grid(
    tmp_path, continuous_source
):
    arr = _write_raw_only(tmp_path, continuous_source(_gappy_samples()))["raw"]
    assert arr.chunks == (8,)
    assert arr.shards == (32,)


def test_write_continuous_channel_empty_source_creates_a_zero_length_raw_array(
    tmp_path, continuous_source
):
    group = _write_raw_only(
        tmp_path, continuous_source(np.empty(0, np.float32))
    )
    assert group["raw"].shape == (0,)
    assert group["raw"][:].shape == (0,)


def test_write_continuous_channel_writes_each_raw_shard_once_in_order(
    tmp_path, continuous_source, monkeypatch
):
    closed = _record_shard_files(monkeypatch)
    samples = _gappy_samples(70)
    group = _write_raw_only(tmp_path, continuous_source(samples))
    root = tmp_path / "bundle" / "0" / "raw" / "c"
    assert [path for path, _ in closed] == [root / "0", root / "1", root / "2"]
    assert np.array_equal(group[RAW_KEY][:], samples, equal_nan=True)


def test_write_continuous_channel_writes_every_raw_shard_of_an_all_gap_source(
    tmp_path, continuous_source
):
    samples = np.full(70, np.nan, dtype=np.float32)
    _write_raw_only(tmp_path, continuous_source(samples))
    shards = [
        f
        for f in (tmp_path / "bundle" / "0" / "raw").rglob("*")
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
    """Per-bin statistics of the finite samples, straight from the definition."""
    values = raw.astype(np.float64)
    rows = []
    for i in range(0, len(values), width):
        block = values[i : i + width]
        block = block[np.isfinite(block)]
        if block.size:
            rows.append(
                [
                    block.min(),
                    block.max(),
                    block.mean(),
                    block.var(),
                    block.size,
                ]
            )
        else:
            rows.append([np.nan, np.nan, np.nan, np.nan, 0.0])
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


def test_write_continuous_channel_writes_every_planned_level(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    opts: WriteOpts,
) -> None:
    samples = _series(1000)
    parent = open_group(tmp_path / "bundle")
    write_continuous_channel(
        parent, 0, continuous_source(samples), reference_us=0, opts=opts
    )
    group = open_group(tmp_path / "bundle")["0"]
    plans = plan_levels(1000, 32000.0, opts.level_cap, opts.min_bins)
    assert len(plans) >= 2
    assert sorted(group.group_keys()) == sorted(str(p.level) for p in plans)
    for plan in plans:
        want = _direct(samples, 4**plan.level)
        level = group[str(plan.level)]
        assert dict(level.attrs) == {
            "tsz": {"period_us": level_period_us(32000.0, 4**plan.level)}
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


def _write(tmp_path, source, opts, reference_us=0, name="bundle"):
    parent = open_group(tmp_path / name)
    write_continuous_channel(
        parent, 0, source, reference_us=reference_us, opts=opts
    )
    return open_group(tmp_path / name)["0"]


def test_write_bundle_opens_more_arrays_than_the_file_descriptor_limit(
    tmp_path,
):
    sources = [
        ArrayContinuousSource(
            _series(3000, seed=index),
            id=f"c{index}",
            name=f"c{index}",
            rate_hz=512.0,
            start_us=0,
        )
        for index in range(12)
    ]
    limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    in_use = len(list(Path("/dev/fd").iterdir()))
    resource.setrlimit(resource.RLIMIT_NOFILE, (in_use + 48, limit[1]))
    try:
        write_bundle(
            [ArrayMultiChannelSource(sources, window_rows=1000)],
            [],
            output_path=tmp_path / "bundle",
            opts=_WIDE_ONE_PASS,
        )
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, limit)
    raw = open_group(tmp_path / "bundle")["11"]["raw"]
    np.testing.assert_array_equal(raw[:], _series(3000, seed=11))


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
    levels = plan_levels(1000, 32000.0, opts.level_cap, opts.min_bins)
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
    for plan in plan_levels(1000, 32000.0, opts.level_cap, opts.min_bins):
        level = group[str(plan.level)]
        assert dict(level.attrs) == {
            "tsz": {"period_us": level_period_us(32000.0, 4**plan.level)}
        }
        assert level[MINMAX_KEY].shape == (plan.bins, 2)
        assert level[MEAN_KEY].shape == (plan.bins,)
        assert level[COUNT_KEY].shape == (plan.bins,)


def test_write_continuous_channel_members_equal_the_direct_definition_over_six_levels(
    tmp_path, continuous_source, wide_opts
):
    samples = _series(8199)
    group = _write(tmp_path, continuous_source(samples), wide_opts)
    levels = plan_levels(8199, 32000.0, wide_opts.level_cap, wide_opts.min_bins)
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


@pytest.mark.parametrize("factor", [2, 3, 10])
def test_write_continuous_channel_members_equal_the_direct_definition_at_another_factor(
    tmp_path, continuous_source, wide_opts, factor
):
    samples = _series(8199)
    opts = dataclasses.replace(wide_opts, decimation_factor=factor)
    group = _write(tmp_path, continuous_source(samples), opts)
    plans = plan_levels(
        8199,
        32000.0,
        opts.level_cap,
        opts.min_bins,
        decimation_factor=factor,
    )
    assert len(plans) >= 3
    assert sorted(group.group_keys()) == sorted(str(p.level) for p in plans)
    for plan in plans:
        width = factor**plan.level
        want = _direct(samples, width)
        level = group[str(plan.level)]
        assert dict(level.attrs) == {
            "tsz": {"period_us": level_period_us(32000.0, width)}
        }
        assert level[MINMAX_KEY].shape == (plan.bins, 2)
        assert np.array_equal(
            level[MINMAX_KEY][:], want[:, :2].astype(np.float32), equal_nan=True
        )
        assert np.array_equal(level[COUNT_KEY][:], want[:, 4])
        assert _close(level[MEAN_KEY][:], want[:, 2]), plan.level
        if opts.write_var:
            assert _close(level[VAR_KEY][:], want[:, 3]), plan.level


def test_write_continuous_channel_at_2x_gets_more_than_15_levels_by_default(
    tmp_path, continuous_source
):
    samples = _series(2**16 + 5)
    opts = WriteOpts(
        min_bins=1,
        decimation_factor=2,
        write_var=True,
        inner_len=1024,
        target_shard_bytes=65_536,
    )
    group = _write(tmp_path, continuous_source(samples), opts)
    assert sorted(group.group_keys(), key=int) == [str(k) for k in range(1, 17)]
    top = group["16"]
    want = _direct(samples, 2**16)
    assert top[COUNT_KEY].shape == (2,)
    assert np.array_equal(top[COUNT_KEY][:], want[:, 4])
    assert np.array_equal(
        top[MINMAX_KEY][:], want[:, :2].astype(np.float32), equal_nan=True
    )
    assert dict(top.attrs) == {
        "tsz": {"period_us": level_period_us(32000.0, 2**16)}
    }


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


def _assert_each_shard_file_written_once(root, closed):
    """Check that every shard file below root closed once, after all its bytes."""
    paths = [path for path, _ in closed]
    on_disk = {
        file
        for file in root.rglob("*")
        if file.is_file() and file.name != "zarr.json"
    }
    assert len(paths) > 10
    assert len(set(paths)) == len(paths)
    assert set(paths) == on_disk
    for path, size in closed:
        assert path.stat().st_size == size, path


def test_write_continuous_channel_writes_each_shard_file_once(
    tmp_path, continuous_source, monkeypatch, opts
):
    closed = _record_shard_files(monkeypatch)
    _write(tmp_path, continuous_source(_series(1000)), opts)
    _assert_each_shard_file_written_once(tmp_path / "bundle", closed)


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
    first_index: int = 0,
    reference_us: int = 0,
    name: str = "bundle",
) -> Any:
    """Write source below the root group at tmp_path / name and return it."""
    parent = open_group(tmp_path / name)
    write_continuous_channels(
        parent, first_index, source, reference_us=reference_us, opts=opts
    )
    return open_group(tmp_path / name)


def _files(root: Path) -> dict[str, bytes]:
    """Return the content of every file below root by relative path."""
    return {
        file.relative_to(root).as_posix(): file.read_bytes()
        for file in sorted(root.rglob("*"))
        if file.is_file()
    }


def _one_window_per_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch the jobs of each window before the next window is read."""
    monkeypatch.setattr("timeseries_zarr.write_continuous.WINDOW_BATCH_ROWS", 1)


def _record_writes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the samples of every job that writes a channel."""
    written: list[int] = []
    write = ChannelWriter._write

    def spy(self: ChannelWriter, samples: np.ndarray) -> None:
        written.append(samples.shape[0])
        write(self, samples)

    monkeypatch.setattr(ChannelWriter, "_write", spy)
    return written


def _record_pushes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the length of every block pushed to a channel writer."""
    pushed: list[int] = []
    push = ChannelWriter.push

    def spy(self: ChannelWriter, block: Any) -> None:
        pushed.append(int(np.shape(block)[0]))
        push(self, block)

    monkeypatch.setattr(ChannelWriter, "push", spy)
    return pushed


def _assert_levels_equal_the_definition(
    group: Any, samples: np.ndarray, rate_hz: float, opts: WriteOpts
) -> None:
    plans = plan_levels(
        samples.shape[0], rate_hz, opts.level_cap, opts.min_bins
    )
    assert plans
    assert sorted(group.group_keys()) == sorted(str(p.level) for p in plans)
    for plan in plans:
        want = _direct(samples, 4**plan.level)
        level = group[str(plan.level)]
        assert dict(level.attrs) == {
            "tsz": {"period_us": level_period_us(rate_hz, 4**plan.level)}
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


def test_write_continuous_channels_names_the_groups_from_the_first_index_in_channel_order(
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
        first_index=4,
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
def test_write_continuous_channels_writes_each_shard_file_once_whatever_the_window_length(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
    monkeypatch: pytest.MonkeyPatch,
    opts: WriteOpts,
    window_rows: int,
) -> None:
    closed = _record_shard_files(monkeypatch)
    samples = [_series(1000, seed=1), _series(700, seed=2)]
    channels = [
        continuous_source(each, id=f"c{i}") for i, each in enumerate(samples)
    ]
    root = _write_many(
        tmp_path, multi_channel_source(channels, window_rows), opts
    )
    _assert_each_shard_file_written_once(tmp_path / "bundle", closed)
    for index, expected in enumerate(samples):
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
            each.shape[0], rate, _OPTS.level_cap, _OPTS.min_bins
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
    root = _write_many(tmp_path, source, _OPTS, first_index=5)
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
    _one_window_per_job(monkeypatch)
    pushed = _record_pushes(monkeypatch)
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
    assert pushed == [8, 8]


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
        _write_many(tmp_path, source, _OPTS, first_index=3)


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
    counts: tuple[int, int] | None,
    message: str,
) -> None:
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
            tmp_path, _WindowSource(channels, windows), _OPTS, first_index=2
        )


@pytest.mark.parametrize(
    ("declared", "sizes", "closed_shards", "received"),
    [(40, (32, 16), 1, 48), (40, (41,), 0, 41), (0, (1,), 0, 1)],
    ids=["after a shard", "in one block", "no samples declared"],
)
def test_write_continuous_channels_rejects_too_many_samples_before_writing_past_the_array(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    monkeypatch: pytest.MonkeyPatch,
    declared: int,
    sizes: tuple[int, ...],
    closed_shards: int,
    received: int,
) -> None:
    _one_window_per_job(monkeypatch)
    closed = _record_shard_files(monkeypatch)
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
    assert [path.parent.parent.name for path, _ in closed] == (
        ["raw"] * closed_shards
    )
    assert all(size > 0 for _, size in closed)
    assert scaled == list(sizes[:-1])


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
    assert list(open_group(tmp_path / "bundle").keys()) == []


def test_write_continuous_channels_creates_every_node_in_one_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    calls: list[list[str]] = []

    def spy(root: Any, nodes: Any) -> Any:
        calls.append(sorted(nodes))
        return create_nodes(root, nodes)

    monkeypatch.setattr("timeseries_zarr.write_continuous.create_nodes", spy)
    samples = [_series(1000, seed=1), np.arange(3, dtype=np.float32)]
    channels = [
        continuous_source(each, id=f"c{i}") for i, each in enumerate(samples)
    ]
    _write_many(tmp_path, multi_channel_source(channels), _OPTS, first_index=5)
    levels = len(plan_levels(1000, 32000.0, _OPTS.level_cap, _OPTS.min_bins))
    assert levels >= 2
    members = [COUNT_KEY, MEAN_KEY, MINMAX_KEY]
    assert calls == [
        sorted(
            ["5", "5/raw", "6", "6/raw"]
            + [
                path
                for level in range(1, levels + 1)
                for path in [f"5/{level}"]
                + [f"5/{level}/{member}" for member in members]
            ]
        )
    ]


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


@pytest.mark.parametrize("write_var", [False, True])
@pytest.mark.parametrize("window_rows", [37, 1000])
def test_a_written_bundle_equals_the_definition_over_six_levels(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
    write_var: bool,
    window_rows: int,
) -> None:
    opts = WriteOpts(
        min_bins=2,
        max_levels=8,
        inner_len=16,
        target_shard_bytes=512,
        write_var=write_var,
    )
    lengths = [16389, 16387]
    samples = []
    for seed, length in enumerate(lengths):
        each = _series(length, offset=5000.0, scale=3.0, seed=seed)
        each[1000:1010] = np.nan
        each[7777] = np.inf
        each[9000] = -np.inf
        each[-1] = np.nan
        samples.append(each)
    channels = [
        continuous_source(each, id=f"c{i}") for i, each in enumerate(samples)
    ]
    write_bundle(
        [multi_channel_source(channels, window_rows)],
        [],
        output_path=tmp_path / "bundle",
        opts=opts,
    )
    root = open_group(tmp_path / "bundle")
    for index, each in enumerate(samples):
        group = root[str(index)]
        plans = plan_levels(each.shape[0], 32000.0, opts.level_cap, 2)
        assert len(plans) == 6
        assert np.array_equal(group[RAW_KEY][:], each, equal_nan=True)
        for plan in plans:
            want = _direct(each, 4**plan.level)
            level = group[str(plan.level)]
            minmax = level[MINMAX_KEY][:]
            assert minmax.shape == (plan.bins, 2)
            assert np.array_equal(
                minmax, want[:, :2].astype(np.float32), equal_nan=True
            )
            assert np.array_equal(level[COUNT_KEY][:], want[:, 4])
            assert _close(level[MEAN_KEY][:], want[:, 2]), plan.level
            if write_var:
                assert _close(level[VAR_KEY][:], want[:, 3]), plan.level
            else:
                assert VAR_KEY not in list(level.array_keys())


def _three_channels(
    continuous_source: type[ArrayContinuousSource],
) -> list[ArrayContinuousSource]:
    return [
        continuous_source(_series(1000, seed=1), id="a"),
        continuous_source(
            _series(333, offset=0.005, scale=0.001, seed=2),
            id="b",
            rate_hz=1000.0,
            unit="volts",
        ),
        continuous_source(_series(700, seed=3), id="c", rate_hz=250.0),
    ]


@pytest.mark.parametrize("channel_count", [1, 3, 8])
@pytest.mark.parametrize("workers", [2, 4])
def test_write_continuous_channels_writes_identical_bundles_at_any_worker_count(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
    workers: int,
    channel_count: int,
) -> None:
    channels = (_three_channels(continuous_source) * 3)[:channel_count]
    opts = dataclasses.replace(_OPTS_ONE_PASS, max_workers=1)
    source = multi_channel_source(channels, window_rows=[7, 5, 13] * 3)
    _write_many(tmp_path, source, opts, name="one")
    pooled = dataclasses.replace(opts, max_workers=workers)
    _write_many(tmp_path, source, pooled, name="many")
    left, right = _files(tmp_path / "one"), _files(tmp_path / "many")
    assert len(left) > 20
    assert left == right


def test_write_continuous_channels_encodes_chunk_groups_on_the_pool_for_a_narrow_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    monkeypatch.setattr("timeseries_zarr.shards.ENCODE_GROUP_BYTES", 64)
    threads: set[str] = set()
    encode_group = ArrayWriter._encode_group

    def spy(self: ArrayWriter, chunks: Any) -> Any:
        threads.add(threading.current_thread().name)
        return encode_group(self, chunks)

    monkeypatch.setattr(ArrayWriter, "_encode_group", spy)
    source = multi_channel_source(
        [continuous_source(_series(1000, seed=1))], window_rows=500
    )
    one = dataclasses.replace(_OPTS_ONE_PASS, max_workers=1)
    _write_many(tmp_path, source, one, name="one")
    assert threads == set()
    four = dataclasses.replace(_OPTS_ONE_PASS, max_workers=4)
    _write_many(tmp_path, source, four, name="four")
    assert threads
    assert all(name.startswith("timeseries-zarr") for name in threads)
    assert _files(tmp_path / "one") == _files(tmp_path / "four")


@pytest.mark.parametrize("workers", [1, 4])
def test_write_continuous_channel_writes_identical_bundles_at_any_worker_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    continuous_source: type[ArrayContinuousSource],
    workers: int,
) -> None:
    monkeypatch.setattr("timeseries_zarr.shards.ENCODE_GROUP_BYTES", 64)
    channel = continuous_source(_series(3000, seed=4), id="a")
    for name, count in (("one", 1), ("many", workers)):
        opts = dataclasses.replace(_OPTS_ONE_PASS, max_workers=count)
        write_continuous_channel(
            open_group(tmp_path / name), 0, channel, reference_us=0, opts=opts
        )
    assert _files(tmp_path / "one") == _files(tmp_path / "many")


def test_write_continuous_channels_raises_a_failed_job_with_its_channel_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    _one_window_per_job(monkeypatch)
    push = ChannelWriter.push

    def failing(self: ChannelWriter, block: Any) -> None:
        if self.label.startswith("channel 4 "):
            raise RuntimeError("disk full")
        push(self, block)

    monkeypatch.setattr(ChannelWriter, "push", failing)
    channels = [
        continuous_source(_series(1000, seed=index), id=f"c{index}")
        for index in range(3)
    ]
    source = multi_channel_source(channels, window_rows=100)
    opts = dataclasses.replace(_OPTS, max_workers=3)
    with pytest.raises(RuntimeError, match="disk full") as caught:
        _write_many(tmp_path, source, opts, first_index=3)
    assert caught.value.__notes__ == ["while writing channel 4 ('c1')"]


def test_write_continuous_channels_stops_dispatching_after_a_failed_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    _one_window_per_job(monkeypatch)
    pushed = _record_pushes(monkeypatch)
    push = ChannelWriter.push

    def failing(self: ChannelWriter, block: Any) -> None:
        if len(pushed) > 2:
            raise RuntimeError("disk full")
        push(self, block)

    monkeypatch.setattr(ChannelWriter, "push", failing)
    channels = [continuous_source(_series(1000, seed=1), id="a")]
    source = multi_channel_source(channels, window_rows=100)
    opts = dataclasses.replace(_OPTS, max_workers=1)
    with pytest.raises(RuntimeError, match="disk full"):
        _write_many(tmp_path, source, opts)
    assert len(pushed) == 3


@pytest.mark.parametrize(
    ("batch_rows", "batch_bytes", "written"),
    [(100, 2**26, [120, 110]), (10**6, 240, [60, 60, 60, 50])],
    ids=["rows", "bytes"],
)
def test_write_continuous_channels_gathers_windows_into_one_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    continuous_source: type[ArrayContinuousSource],
    batch_rows: int,
    batch_bytes: int,
    written: list[int],
) -> None:
    monkeypatch.setattr(
        "timeseries_zarr.write_continuous.WINDOW_BATCH_ROWS", batch_rows
    )
    monkeypatch.setattr(
        "timeseries_zarr.write_continuous.WINDOW_BATCH_BYTES", batch_bytes
    )
    writes = _record_writes(monkeypatch)
    samples = _series(230, seed=3)
    windows = [[samples[start : start + 30]] for start in range(0, 230, 30)]
    channel = continuous_source(samples, id="a")
    root = _write_many(tmp_path, _WindowSource([channel], windows), _OPTS)
    assert writes == written
    assert np.array_equal(root["0"][RAW_KEY][:], samples, equal_nan=True)


def test_write_continuous_channels_writes_the_same_bundle_with_and_without_gathering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    channels = [
        continuous_source(_series(1000, seed=index), id=f"c{index}")
        for index in range(3)
    ]
    opts = dataclasses.replace(_OPTS_ONE_PASS, max_workers=3)
    monkeypatch.setattr(
        "timeseries_zarr.write_continuous.WINDOW_BATCH_ROWS", 150
    )
    source = multi_channel_source(channels, window_rows=37)
    _write_many(tmp_path, source, opts, name="gathered")
    _one_window_per_job(monkeypatch)
    _write_many(tmp_path, source, opts, name="alone")
    left, right = _files(tmp_path / "gathered"), _files(tmp_path / "alone")
    assert len(left) > 20
    assert left == right


def _matrix_windows(
    samples: Sequence[np.ndarray], rows: int, dtype: Any = np.float32
) -> list[np.ndarray]:
    """Cut equal-length channels into (rows, channels) windows."""
    data = np.stack(samples, axis=1).astype(dtype)
    return [data[start : start + rows] for start in range(0, len(data), rows)]


@pytest.mark.parametrize("workers", [1, 3])
def test_write_continuous_channels_writes_a_rank_2_window_as_its_columns(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    workers: int,
) -> None:
    samples = [_series(1000, seed=index) for index in range(3)]
    channels = [
        continuous_source(each, id=f"c{index}")
        for index, each in enumerate(samples)
    ]
    windows = _matrix_windows(samples, 64)
    opts = dataclasses.replace(_OPTS_ONE_PASS, max_workers=workers)
    _write_many(tmp_path, _WindowSource(channels, windows), opts, name="matrix")
    as_lists = [list(window.T) for window in windows]
    _write_many(tmp_path, _WindowSource(channels, as_lists), opts, name="lists")
    left, right = _files(tmp_path / "matrix"), _files(tmp_path / "lists")
    assert len(left) > 20
    assert left == right


def test_write_continuous_channels_hands_each_job_a_contiguous_block_of_a_rank_2_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    continuous_source: type[ArrayContinuousSource],
) -> None:
    _one_window_per_job(monkeypatch)
    flags: list[tuple[bool, tuple[int, ...]]] = []
    push = ChannelWriter.push

    def spy(self: ChannelWriter, block: Any) -> None:
        flags.append((block.flags.c_contiguous, block.shape))
        push(self, block)

    monkeypatch.setattr(ChannelWriter, "push", spy)
    samples = [_series(200, seed=index) for index in range(2)]
    channels = [continuous_source(each) for each in samples]
    source = _WindowSource(channels, _matrix_windows(samples, 100))
    _write_many(tmp_path, source, _OPTS)
    assert flags == [(True, (100,))] * 4


@pytest.mark.parametrize("dtype", [np.int16, np.float64])
def test_write_continuous_channels_casts_a_rank_2_window_to_float32(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    dtype: Any,
) -> None:
    samples = [np.arange(300, dtype=np.float32) + 10 * i for i in range(2)]
    channels = [continuous_source(each) for each in samples]
    source = _WindowSource(channels, _matrix_windows(samples, 128, dtype))
    root = _write_many(tmp_path, source, _OPTS)
    for index, each in enumerate(samples):
        assert np.array_equal(root[str(index)][RAW_KEY][:], each)


def test_write_continuous_channels_accepts_an_empty_rank_2_window(
    tmp_path: Path, continuous_source: type[ArrayContinuousSource]
) -> None:
    samples = [np.arange(40, dtype=np.float32) for _ in range(2)]
    channels = [continuous_source(each) for each in samples]
    windows = [np.zeros((0, 2), np.float32), *_matrix_windows(samples, 40)]
    root = _write_many(tmp_path, _WindowSource(channels, windows), _OPTS)
    assert np.array_equal(root["1"][RAW_KEY][:], samples[1])


@pytest.mark.parametrize("columns", [1, 3])
def test_write_continuous_channels_rejects_a_rank_2_window_with_the_wrong_column_count(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    columns: int,
) -> None:
    channels = [
        continuous_source(_series(1000, seed=1), id="a"),
        continuous_source(_series(1000, seed=2), id="b"),
    ]
    source = _WindowSource(channels, [np.zeros((8, columns), np.float32)])
    with pytest.raises(
        ValueError,
        match=rf"window 0 holds {columns} blocks and the source has 2 channels",
    ):
        _write_many(tmp_path, source, _OPTS)


_RUN_RATE_HZ = 1000.0
_RUN_FIRST = 300
_RUN_SECOND = 200
_RUN_RESUME_US = 700_300
_RUN_RESUME_INDEX = 700
_RUN_GRID_LENGTH = 900


def _clean(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (5000.0 + 8.0 * rng.standard_normal(n)).astype(np.float32)


def _two_runs(
    runs_source: Any, seed: int = 0, start_us: int = 0, **options: Any
) -> Any:
    """Return a source whose second run resumes 0.3 sample periods off the grid."""
    return runs_source(
        [
            (start_us, _clean(_RUN_FIRST, seed)),
            (start_us + _RUN_RESUME_US, _clean(_RUN_SECOND, seed + 1)),
        ],
        rate_hz=_RUN_RATE_HZ,
        **options,
    )


def _two_run_grid(seed: int = 0) -> np.ndarray:
    grid = np.full(_RUN_GRID_LENGTH, np.nan, dtype=np.float32)
    grid[:_RUN_FIRST] = _clean(_RUN_FIRST, seed)
    grid[_RUN_RESUME_INDEX:] = _clean(_RUN_SECOND, seed + 1)
    return grid


def test_write_continuous_channel_writes_nan_between_runs_and_the_runs_array(
    tmp_path: Path, runs_source: Any, opts: WriteOpts
) -> None:
    group = _write(tmp_path, _two_runs(runs_source), opts)
    assert group[RAW_KEY].shape == (_RUN_GRID_LENGTH,)
    assert np.array_equal(group[RAW_KEY][:], _two_run_grid(), equal_nan=True)
    runs = group["runs"]
    assert runs.shape == (2, 3)
    assert runs.dtype == np.int64
    assert runs.metadata.dimension_names == ("run", None)
    assert runs.chunks == runs.shards == (2, 3)
    assert runs[:].tolist() == [
        [0, _RUN_FIRST, 0],
        [_RUN_RESUME_INDEX, _RUN_SECOND, _RUN_RESUME_US],
    ]


def test_write_continuous_channel_offset_is_the_time_of_the_first_run(
    tmp_path: Path, runs_source: Any
) -> None:
    group = _write(
        tmp_path,
        _two_runs(runs_source, start_us=5000),
        _OPTS,
        reference_us=2000,
    )
    assert group.attrs["tsz"]["offset_us"] == 3000
    assert group["runs"][:, 2].tolist() == [3000, 3000 + _RUN_RESUME_US]


def test_write_continuous_channel_levels_over_a_pause_equal_the_fold_of_the_grid(
    tmp_path: Path, runs_source: Any, opts: WriteOpts
) -> None:
    group = _write(tmp_path, _two_runs(runs_source), opts)
    _assert_levels_equal_the_definition(
        group, _two_run_grid(), _RUN_RATE_HZ, opts
    )
    count = group["1"][COUNT_KEY][:]
    assert count[_RUN_FIRST // 4 : _RUN_RESUME_INDEX // 4].sum() == 0
    assert count[: _RUN_FIRST // 4].min() == 4


def test_write_continuous_channel_pushes_a_pause_in_pieces_of_one_raw_shard(
    tmp_path: Path, runs_source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pieces: list[int] = []
    push = ChannelWriter._push_rows

    def spy(self: ChannelWriter, rows: np.ndarray) -> None:
        if np.isnan(rows).all():
            pieces.append(rows.shape[0])
        push(self, rows)

    monkeypatch.setattr(ChannelWriter, "_push_rows", spy)
    group = _write(tmp_path, _two_runs(runs_source), _OPTS)
    assert group[RAW_KEY].shards == (32,)
    gap = _RUN_RESUME_INDEX - _RUN_FIRST
    assert pieces == [32] * (gap // 32) + [gap % 32]


def test_write_continuous_channel_writes_each_shard_file_once_over_a_pause(
    tmp_path: Path,
    runs_source: Any,
    monkeypatch: pytest.MonkeyPatch,
    opts: WriteOpts,
) -> None:
    closed = _record_shard_files(monkeypatch)
    _write(tmp_path, _two_runs(runs_source), opts)
    root = tmp_path / "bundle"
    paths = [path for path, _ in closed]
    on_disk = {
        file
        for file in root.rglob("*")
        if file.is_file()
        and file.name != "zarr.json"
        and "runs" not in file.relative_to(root).parts
    }
    assert len(paths) > 10
    assert len(set(paths)) == len(paths)
    assert set(paths) == on_disk
    for path, size in closed:
        assert path.stat().st_size == size, path


def test_write_continuous_channel_one_run_writes_the_bytes_of_a_source_without_runs(
    tmp_path: Path, runs_source: Any, continuous_source: Any, opts: WriteOpts
) -> None:
    samples = _clean(500, 4)
    _write(tmp_path, continuous_source(samples, rate_hz=1000.0), opts, name="a")
    _write(
        tmp_path,
        runs_source([(0, samples)], rate_hz=1000.0),
        opts,
        name="b",
    )
    left, right = _files(tmp_path / "a"), _files(tmp_path / "b")
    assert "0/runs/zarr.json" not in right
    assert left == right


def test_write_continuous_channel_records_a_pause_shorter_than_a_period_without_nan(
    tmp_path: Path, runs_source: Any
) -> None:
    source = runs_source(
        [(0, _clean(40, 1)), (40_300, _clean(30, 2))], rate_hz=1000.0
    )
    group = _write(tmp_path, source, _OPTS)
    assert group[RAW_KEY].shape == (70,)
    assert not np.isnan(group[RAW_KEY][:]).any()
    assert group["runs"][:].tolist() == [[0, 40, 0], [40, 30, 40_300]]


def test_write_continuous_channel_rejects_runs_that_do_not_total_num_samples(
    tmp_path: Path, runs_source: Any
) -> None:
    source = _two_runs(runs_source)
    source.num_samples = lambda: _RUN_FIRST + _RUN_SECOND + 1
    with pytest.raises(ValueError, match="the runs hold 500 samples"):
        _write(tmp_path, source, _OPTS)
    assert not (tmp_path / "bundle" / "0").exists()


def test_write_bundle_rejects_runs_that_start_after_start_us_before_writing(
    tmp_path: Path, runs_source: Any
) -> None:
    source = _two_runs(runs_source)
    source.start_us = lambda: 7
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="channel 0 .* must be equal"):
        write_bundle([source], [], output_path=output, opts=_OPTS)
    assert not output.exists()


def test_write_bundle_rejects_a_run_whose_last_sample_passes_the_time_limit(
    tmp_path: Path, runs_source: Any, continuous_source: Any
) -> None:
    anchor = continuous_source(_clean(1, 1), start_us=0, rate_hz=1.0)
    source = runs_source(
        [(640_992, _clean(1, 2)), (2**53 - 700_000, _clean(2, 3))],
        rate_hz=1.0,
        id="late",
        name="late",
    )
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="channel 1 .*last sample of run 1"):
        write_bundle([anchor, source], [], output_path=output, opts=_OPTS)
    assert not output.exists()


def test_write_bundle_rejects_runs_that_arrive_faster_than_the_sample_rate(
    tmp_path: Path, runs_source: Any
) -> None:
    source = runs_source(
        [(0, _clean(40, 1)), (39_000, _clean(30, 2))], rate_hz=1000.0
    )
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="at least one sample period"):
        write_bundle([source], [], output_path=output, opts=_OPTS)
    assert not output.exists()


def test_write_continuous_channel_with_runs_places_a_channel_in_a_bundle_by_its_first_run(
    tmp_path: Path, runs_source: Any, continuous_source: Any
) -> None:
    early = continuous_source(_clean(100, 1), rate_hz=1000.0, start_us=-4000)
    gapped = _two_runs(runs_source, start_us=10_000, id="g")
    output = tmp_path / "bundle"
    write_bundle([early, gapped], [], output_path=output, opts=_OPTS)
    root = open_group(output)
    assert root["1"].attrs["tsz"]["offset_us"] == 14_000
    assert root["1"]["runs"][:, 2].tolist() == [14_000, 14_000 + _RUN_RESUME_US]


def test_write_continuous_channel_straddling_block_lands_in_both_runs(
    tmp_path: Path, runs_source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pushed = _record_pushes(monkeypatch)
    group = _write(tmp_path, _two_runs(runs_source), _OPTS)
    shard = group[RAW_KEY].shards[0]
    assert _RUN_FIRST % shard != 0
    assert sum(pushed[: _RUN_FIRST // shard + 1]) > _RUN_FIRST
    assert np.array_equal(group[RAW_KEY][:], _two_run_grid(), equal_nan=True)


def test_write_continuous_channels_with_runs_matches_the_single_channel_write(
    tmp_path: Path,
    runs_source: Any,
    continuous_source: Any,
    multi_channel_source: type[ArrayMultiChannelSource],
    opts: WriteOpts,
) -> None:
    channels = [
        _two_runs(runs_source, seed=1, id="a"),
        continuous_source(
            _clean(_RUN_GRID_LENGTH, 9), id="b", rate_hz=_RUN_RATE_HZ
        ),
        _two_runs(runs_source, seed=5, id="c", start_us=20),
    ]
    single = open_group(tmp_path / "single")
    for key, channel in enumerate(channels):
        write_continuous_channel(
            single, key, channel, reference_us=0, opts=opts
        )
    # Rows 288 through 319 of channel 0 straddle the resume at row 300.
    source = multi_channel_source(channels, window_rows=[32, 7, 50])
    root = _write_many(tmp_path, source, opts, name="sweep")
    left, right = _files(tmp_path / "single"), _files(tmp_path / "sweep")
    assert "0/runs/zarr.json" in right
    assert "1/runs/zarr.json" not in right
    assert left.keys() == right.keys()
    assert [name for name in left if left[name] != right[name]] == []
    assert np.array_equal(
        root["0"][RAW_KEY][:], _two_run_grid(1), equal_nan=True
    )


def test_write_continuous_channels_splits_a_rank_2_window_at_a_run_boundary(
    tmp_path: Path, runs_source: Any
) -> None:
    channels = [
        _two_runs(runs_source, seed=1, id="a"),
        _two_runs(runs_source, seed=3, id="b"),
    ]
    recorded = np.stack(
        [channel.read_samples(0, 500) for channel in channels], axis=1
    )
    windows = [recorded[:130], recorded[130:310], recorded[310:]]
    source = _WindowSource(channels, windows)
    root = _write_many(tmp_path, source, _OPTS)
    assert np.array_equal(
        root["0"][RAW_KEY][:], _two_run_grid(1), equal_nan=True
    )
    assert np.array_equal(
        root["1"][RAW_KEY][:], _two_run_grid(3), equal_nan=True
    )


@pytest.mark.parametrize(
    ("total", "message"),
    [
        (499, r"channel 0 \('a'\) yields 499 samples and declares 500"),
        (501, r"channel 0 \('a'\) yields at least 501 samples"),
    ],
    ids=["too few", "too many"],
)
def test_write_continuous_channels_with_runs_counts_the_recorded_samples(
    tmp_path: Path, runs_source: Any, total: int, message: str
) -> None:
    channel = _two_runs(runs_source, id="a")
    blocks = [np.zeros(total - 100, np.float32), np.zeros(100, np.float32)]
    source = _WindowSource([channel], [[block] for block in blocks])
    with pytest.raises(ValueError, match=message):
        _write_many(tmp_path, source, _OPTS)
