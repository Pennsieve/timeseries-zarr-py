import dataclasses
import re
import sys

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from tests.conftest import (
    ODD_NAN_RUN,
    ODD_OFFSET_US,
    ODD_RATE_HZ,
    RUNS_GRID_LENGTH,
    RUNS_LENGTHS,
    RUNS_RESUME_US,
    RUNS_START_US,
    RUNS_TABLE,
    runs_samples,
)
from tests.test_bundle import _ACQUIRE
from timeseries_zarr import read_continuous
from timeseries_zarr.attrs import continuous_channel_attrs, level_group_attrs
from timeseries_zarr.constants import (
    COUNT_KEY,
    MEAN_KEY,
    MINMAX_KEY,
    RAW_KEY,
    RUNS_KEY,
    TSZ_KEY,
    VAR_KEY,
)
from timeseries_zarr.planning import level_bins, level_period_us
from timeseries_zarr.read_continuous import (
    Bins,
    Level,
    Samples,
    parse_continuous_channel,
)
from timeseries_zarr.zarr_io import descendants, open_root

_HAND_RATE_HZ = 512.0
_HAND_SAMPLES = 64
_ODD_END_US = 1_955_079
"""End of channel 0 of the odd bundle in microseconds.

It is the duration of 1001 samples at 512 Hz, rounded up to a whole
microsecond.
"""


def _below(nodes, prefix):
    return {
        path.removeprefix(prefix): node
        for path, node in nodes.items()
        if path.startswith(prefix)
    }


def _channel(location, key):
    """Return channel key of a bundle, parsed from descendants of its root."""
    root = open_root(location)
    groups, arrays = descendants(root)
    prefix = f"{key}/"
    return parse_continuous_channel(
        key,
        groups[str(key)].attrs,
        _below(groups, prefix),
        _below(arrays, prefix),
    )


def _hand_nodes(num_samples=_HAND_SAMPLES, levels=2, widths=None):
    """Return the nodes of a valid channel keyed by path below the channel.

    A dict is the attributes of a group and an array is the data of an
    array. Each level holds minmax only. widths lists the samples per bin of
    each level, and the default is 4 to the power of the level number.
    """
    if widths is None:
        widths = [4**level for level in range(1, levels + 1)]
    nodes = {RAW_KEY: np.arange(num_samples, dtype=np.float32)}
    for level, width in enumerate(widths, start=1):
        bins = level_bins(num_samples, width)
        nodes[str(level)] = level_group_attrs(
            level_period_us(_HAND_RATE_HZ, width)
        )
        nodes[f"{level}/{MINMAX_KEY}"] = np.zeros((bins, 2), np.float32)
    return nodes


def _hand_attrs():
    return continuous_channel_attrs(
        "hand", "hand-0", _HAND_RATE_HZ, 0, "uV", None
    )


def _hand_build(nodes, attributes=None):
    """Write nodes under channel 0 of a MemoryStore and return the root."""
    root = zarr.open_group(MemoryStore(), mode="w", zarr_format=3)
    channel = root.create_group(
        "0", attributes=_hand_attrs() if attributes is None else attributes
    )
    for path in sorted(nodes, key=lambda name: name.count("/")):
        node = nodes[path]
        if isinstance(node, dict):
            channel.create_group(path, attributes=node)
        else:
            channel.create_array(path, data=node)
    return root


def _hand_parse(nodes, attributes=None):
    """Write nodes under channel 0 of a MemoryStore and parse the channel."""
    channel = _hand_build(nodes, attributes)["0"]
    groups, arrays = descendants(channel)
    return parse_continuous_channel(0, channel.attrs, groups, arrays)


# Parsed fields


def test_odd_channel_without_id_or_offset_parses_with_defaults(odd_bundle):
    channel = _channel(odd_bundle, 0)
    assert channel.key == 0
    assert channel.kind == "continuous"
    assert channel.name == "short"
    assert channel.id is None
    assert channel.unit == "uV"
    assert channel.rate_hz == ODD_RATE_HZ
    assert channel.offset_us == 0
    assert channel.provenance is None
    assert channel.num_samples == 1001
    assert channel.period_us == 1e6 / ODD_RATE_HZ
    assert channel.start_us == 0
    assert channel.end_us == _ODD_END_US


def test_level_without_optional_members_holds_minmax_only(odd_bundle):
    level = _channel(odd_bundle, 0).levels[1]
    assert isinstance(level, Level)
    assert level.number == 1
    assert level.period_us == level_period_us(ODD_RATE_HZ, 4)
    assert level.samples_per_bin == 4
    assert level.bins == 251
    assert level.minmax.shape == (251, 2)
    assert (level.mean, level.var, level.count) == (None, None, None)


def test_level_with_count_and_no_mean_parses_count_only(odd_bundle):
    level = _channel(odd_bundle, 0).levels[2]
    assert level.samples_per_bin == 16
    assert level.bins == 63
    assert level.count.shape == (63,)
    assert (level.mean, level.var) == (None, None)


def test_level_with_every_member_exposes_each_array(odd_bundle):
    channel = _channel(odd_bundle, 2)
    level = channel.levels[1]
    assert channel.id == "two"
    assert level.mean.shape == level.var.shape == level.count.shape == (750,)
    assert level.count.dtype == np.uint32


def test_levels_is_a_read_only_mapping_in_level_order(odd_bundle):
    channel = _channel(odd_bundle, 0)
    assert list(channel.levels) == [1, 2]
    with pytest.raises(TypeError):
        channel.levels[3] = channel.levels[1]


def test_records_are_frozen(odd_bundle):
    channel = _channel(odd_bundle, 0)
    samples = channel.read(0, 10_000)
    with pytest.raises(dataclasses.FrozenInstanceError):
        channel.name = "renamed"
    with pytest.raises(dataclasses.FrozenInstanceError):
        channel.levels[1].number = 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        samples.first = 1


def test_records_compare_by_identity(odd_bundle):
    first = _channel(odd_bundle, 0)
    second = _channel(odd_bundle, 0)
    assert first != second
    assert {first: "first"}[first] == "first"
    assert first.read(0, 10_000) != first.read(0, 10_000)
    assert first.read(level=1) != first.read(level=1)
    assert len({first, second}) == 2


def test_records_have_no_instance_dict(odd_bundle):
    channel = _channel(odd_bundle, 0)
    records = [
        channel,
        channel.levels[1],
        channel.read(0, 10_000),
        channel.read(level=1),
    ]
    for record in records:
        assert not hasattr(record, "__dict__"), type(record).__name__


def test_repr_names_the_key_name_rate_samples_and_level_count(odd_bundle):
    assert repr(_channel(odd_bundle, 0)) == (
        "ContinuousChannel(key=0, name='short', rate_hz=512.0, "
        "num_samples=1001, levels=2)"
    )


def test_mixed_channel_carries_its_provenance(mixed_bundle):
    channel = _channel(mixed_bundle, 0)
    assert channel.id == "LA1"
    assert channel.provenance == _ACQUIRE
    assert list(channel.levels) == [1, 2]


def test_mixed_channel_carries_its_offset(mixed_bundle):
    channel = _channel(mixed_bundle, 1)
    assert channel.offset_us == 500_000
    assert channel.end_us == 32_500_000
    assert list(channel.levels) == [1, 2, 3]


def test_channel_with_no_samples_has_no_levels_and_an_empty_window(
    odd_bundle,
):
    channel = _channel(odd_bundle, 7)
    assert channel.num_samples == 0
    assert dict(channel.levels) == {}
    assert channel.end_us == channel.start_us == 0


# Raw reads


def test_read_without_a_window_returns_every_sample(odd_bundle):
    channel = _channel(odd_bundle, 0)
    samples = channel.read()
    assert isinstance(samples, Samples)
    assert (samples.offset_us, samples.first) == (0, 0)
    assert samples.period_us == channel.period_us
    assert len(samples) == 1001
    assert samples.values.dtype == np.float32


def test_read_from_start_us_to_end_us_covers_every_sample(odd_bundle):
    channel = _channel(odd_bundle, 0)
    samples = channel.read(channel.start_us, channel.end_us)
    assert (samples.first, len(samples)) == (0, channel.num_samples)


def test_read_at_level_0_equals_a_raw_read(odd_bundle):
    channel = _channel(odd_bundle, 0)
    at_zero = channel.read(10_000, 90_000, level=0)
    raw = channel.read(10_000, 90_000)
    assert isinstance(at_zero, Samples)
    assert at_zero.first == raw.first
    np.testing.assert_array_equal(at_zero.values, raw.values)


def test_raw_read_equals_a_direct_zarr_slice(odd_bundle):
    channel = _channel(odd_bundle, 2)
    samples = channel.read(300_000, 1_000_000)
    stop = samples.first + len(samples)
    direct = zarr.open_group(odd_bundle, mode="r")["2/raw"]
    assert (samples.first, stop) == (153, 512)
    np.testing.assert_array_equal(
        samples.values, np.asarray(direct[samples.first : stop])
    )


def test_raw_read_keeps_a_nan_run(odd_bundle):
    samples = _channel(odd_bundle, 0).read()
    finite = np.isfinite(samples.values)
    assert np.flatnonzero(~finite).tolist() == list(range(*ODD_NAN_RUN))


def test_sample_times_count_from_the_channel_offset(odd_bundle):
    samples = _channel(odd_bundle, 5).read()
    assert samples.offset_us == ODD_OFFSET_US
    assert samples.time_us[0] == 2_000_000.0
    assert samples.time_us[1] == 2_001_953.125
    assert samples.time_s[512] == 3.0
    assert samples.time_us.dtype == samples.time_s.dtype == np.float64


def test_read_snaps_a_window_on_an_offset_channel_to_whole_samples(
    odd_bundle,
):
    channel = _channel(odd_bundle, 5)
    assert channel.start_us == ODD_OFFSET_US
    assert channel.end_us == 4_000_000
    assert len(channel.read(0, 2_000_000)) == 0
    assert len(channel.read(0, 2_000_001)) == 1
    assert channel.read(3_000_000).first == 512


def test_read_places_30_khz_samples_and_bins_on_the_exact_grid():
    attributes = continuous_channel_attrs(
        "fast", "fast-0", 30_000.0, 0, "uV", None
    )
    nodes = {
        RAW_KEY: np.zeros(300, np.float32),
        "1": level_group_attrs(level_period_us(30_000.0, 4)),
        f"1/{MINMAX_KEY}": np.zeros((75, 2), np.float32),
    }
    channel = _hand_parse(nodes, attributes)
    samples = channel.read(1000, 2000)
    assert (samples.first, len(samples)) == (30, 30)
    assert channel.read(4000, 8000, level=1).first == 30


def test_read_before_the_channel_is_empty(odd_bundle):
    samples = _channel(odd_bundle, 5).read(0, 1_000_000)
    assert (samples.first, len(samples)) == (0, 0)
    assert samples.time_us.shape == (0,)


def test_read_after_the_channel_is_empty(odd_bundle):
    channel = _channel(odd_bundle, 0)
    samples = channel.read(10**9, 2 * 10**9)
    bins = channel.read(10**9, 2 * 10**9, level=2)
    assert (samples.first, len(samples)) == (1001, 0)
    assert (bins.first, len(bins)) == (63, 0)
    assert bins.min.shape == bins.max.shape == bins.count.shape == (0,)


def test_read_of_equal_bounds_is_empty(odd_bundle):
    assert len(_channel(odd_bundle, 0).read(50_000, 50_000)) == 0


def test_read_of_a_channel_with_no_samples_is_empty(odd_bundle):
    channel = _channel(odd_bundle, 7)
    for samples in (channel.read(), channel.read(0, 10), channel.read(5)):
        assert (samples.first, len(samples)) == (0, 0)
        assert samples.values.shape == (0,)


def test_read_accepts_numpy_integer_bounds(odd_bundle):
    channel = _channel(odd_bundle, 0)
    samples = channel.read(np.int64(10_000), np.int64(90_000))
    assert (samples.first, len(samples)) == (5, 42)


# Level reads


def test_level_read_returns_bins_from_the_minmax_columns(odd_bundle):
    channel = _channel(odd_bundle, 2)
    bins = channel.read(300_000, 1_000_000, level=1)
    direct = zarr.open_group(odd_bundle, mode="r")["2/1"]
    first, stop = bins.first, bins.first + len(bins)
    assert isinstance(bins, Bins)
    assert (bins.level, bins.offset_us, first, stop) == (1, 0, 38, 128)
    assert bins.period_us == level_period_us(ODD_RATE_HZ, 4)
    minmax = np.asarray(direct[MINMAX_KEY][first:stop])
    np.testing.assert_array_equal(bins.min, minmax[:, 0])
    np.testing.assert_array_equal(bins.max, minmax[:, 1])
    for name in (MEAN_KEY, VAR_KEY, COUNT_KEY):
        np.testing.assert_array_equal(
            getattr(bins, name), np.asarray(direct[name][first:stop])
        )


def test_level_read_sets_absent_members_to_none(odd_bundle):
    bins = _channel(odd_bundle, 0).read(level=1)
    assert (bins.mean, bins.var, bins.count) == (None, None, None)
    assert len(bins) == 251


def test_level_read_keeps_the_short_last_bin(odd_bundle):
    channel = _channel(odd_bundle, 0)
    raw = channel.read().values
    level_one = channel.read(level=1)
    level_two = channel.read(level=2)
    assert level_one.min[-1] == level_one.max[-1] == raw[1000]
    assert level_two.min[-1] == raw[992:].min()
    assert level_two.max[-1] == raw[992:].max()
    assert level_two.count[-1] == 9
    assert level_two.time_us[-1] == 62 * 31_250.0


def test_level_read_passes_nan_bins_and_counts_through(odd_bundle):
    channel = _channel(odd_bundle, 0)
    level_one = channel.read(level=1)
    level_two = channel.read(level=2)
    assert np.flatnonzero(np.isnan(level_one.min)).tolist() == list(
        range(25, 35)
    )
    assert np.flatnonzero(np.isnan(level_two.max)).tolist() == [6, 7, 8]
    assert level_two.count[5:10].tolist() == [16, 4, 0, 4, 16]


def test_bin_times_count_from_the_channel_offset(mixed_bundle):
    bins = _channel(mixed_bundle, 1).read(0, 1_000_000, level=1)
    assert (bins.first, len(bins)) == (0, 64)
    assert bins.time_us[:2].tolist() == [500_000.0, 507_812.5]
    assert bins.time_s[0] == 0.5


def test_read_accepts_a_numpy_integer_level(odd_bundle):
    bins = _channel(odd_bundle, 0).read(level=np.int64(2))
    assert (bins.level, len(bins)) == (2, 63)


def _recorded_paths(monkeypatch):
    """Record the path of every array that read_continuous.read_rows reads."""
    paths = []
    read_rows = read_continuous.read_rows

    def counting(array, start, stop):
        paths.append(array.path)
        return read_rows(array, start, stop)

    monkeypatch.setattr(read_continuous, "read_rows", counting)
    return paths


@pytest.mark.parametrize("members", [(), ("minmax",)])
def test_level_read_of_no_other_member_holds_min_and_max_alone(
    odd_bundle, members
):
    channel = _channel(odd_bundle, 2)
    full = channel.read(300_000, 1_000_000, level=1)
    bins = channel.read(300_000, 1_000_000, level=1, members=members)
    assert (bins.first, len(bins)) == (full.first, len(full))
    np.testing.assert_array_equal(bins.min, full.min)
    np.testing.assert_array_equal(bins.max, full.max)
    assert (bins.mean, bins.var, bins.count) == (None, None, None)


def test_level_read_returns_only_the_named_members(odd_bundle):
    channel = _channel(odd_bundle, 2)
    full = channel.read(level=1)
    bins = channel.read(level=1, members=("mean", "count"))
    np.testing.assert_array_equal(bins.mean, full.mean)
    np.testing.assert_array_equal(bins.count, full.count)
    assert bins.var is None


def test_level_read_of_no_other_member_reads_only_minmax(
    odd_bundle, monkeypatch
):
    channel = _channel(odd_bundle, 2)
    paths = _recorded_paths(monkeypatch)
    channel.read(level=1, members=())
    assert paths == [channel.levels[1].minmax.path]


def test_level_read_reads_a_repeated_member_once(odd_bundle, monkeypatch):
    channel = _channel(odd_bundle, 2)
    paths = _recorded_paths(monkeypatch)
    channel.read(level=1, members=("mean", "mean"))
    assert paths == [
        channel.levels[1].minmax.path,
        channel.levels[1].mean.path,
    ]


@pytest.mark.parametrize("name", ["mean", "median"])
def test_level_read_rejects_a_member_the_level_lacks(odd_bundle, name):
    channel = _channel(odd_bundle, 0)
    message = (
        "channel 0 level 1 has members ('minmax',); members must name only "
        f"these, got '{name}'"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        channel.read(level=1, members=(name,))


@pytest.mark.parametrize("level", [0, None])
def test_raw_read_rejects_members(odd_bundle, level):
    message = "level 0 holds raw samples and takes members None, got ()"
    with pytest.raises(ValueError, match=re.escape(message)):
        _channel(odd_bundle, 0).read(level=level, members=())


def test_level_read_rejects_a_string_of_members(odd_bundle):
    message = (
        "members must be a collection of member names, got the string 'mean'"
    )
    with pytest.raises(TypeError, match=re.escape(message)):
        _channel(odd_bundle, 2).read(level=1, members="mean")


def test_level_read_of_members_none_reads_every_member(odd_bundle):
    channel = _channel(odd_bundle, 2)
    bins = channel.read(level=1, members=None)
    full = channel.read(level=1)
    for name, values in full.columns().items():
        np.testing.assert_array_equal(bins.columns()[name], values)


def test_level_read_with_members_accepts_numpy_integers(odd_bundle):
    bins = _channel(odd_bundle, 2).read(
        np.int64(0), np.int64(1_000_000), level=np.int64(1), members=()
    )
    assert (bins.level, bins.mean) == (1, None)
    assert len(bins) == 128


def test_read_rejects_float_bounds(odd_bundle):
    channel = _channel(odd_bundle, 0)
    with pytest.raises(TypeError, match="start_us must be an integer"):
        channel.read(1.5)
    with pytest.raises(TypeError, match="end_us must be an integer"):
        channel.read(0, 2.0, level=1)


def test_read_rejects_a_float_level(odd_bundle):
    with pytest.raises(
        TypeError, match=r"level must be an integer or None, got 1\.0"
    ):
        _channel(odd_bundle, 0).read(level=1.0)


def test_read_takes_level_only_as_a_keyword(odd_bundle):
    with pytest.raises(TypeError, match="positional argument"):
        _channel(odd_bundle, 0).read(0, 1_000_000, 1)


def test_read_rejects_an_end_before_the_start(odd_bundle):
    with pytest.raises(ValueError, match="end_us must be at or after start_us"):
        _channel(odd_bundle, 0).read(2_000, 1_000)


@pytest.mark.parametrize("level", [3, -1])
def test_read_rejects_a_level_the_channel_does_not_hold(odd_bundle, level):
    message = (
        "channel 0 has levels 1 through 2; level must be one of them, "
        "or 0 or None for raw samples"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _channel(odd_bundle, 0).read(level=level)


def test_read_rejects_a_level_on_a_channel_with_one_level(odd_bundle):
    message = "channel 2 has level 1; level must be 1, or 0 or None for raw"
    with pytest.raises(ValueError, match=re.escape(message)):
        _channel(odd_bundle, 2).read(level=2)


def test_read_rejects_a_level_on_a_channel_without_levels(odd_bundle):
    message = "channel 5 has no levels; level must be 0 or None for raw"
    with pytest.raises(ValueError, match=re.escape(message)):
        _channel(odd_bundle, 5).read(level=1)


# Level choice


def test_level_for_picks_the_coarsest_level_with_enough_bins(odd_bundle):
    channel = _channel(odd_bundle, 0)
    assert channel.level_for(bins=63) == 2
    assert channel.level_for(bins=64) == 1
    assert channel.level_for(bins=252) == 0
    assert channel.level_for(0, 500_000, bins=16) == 2
    assert channel.level_for(0, 500_000, bins=17) == 1


def test_level_for_counts_the_bins_a_read_returns(odd_bundle):
    channel = _channel(odd_bundle, 0)
    level = channel.level_for(12_345, 456_789, bins=50)
    assert level == 1
    assert len(channel.read(12_345, 456_789, level=level)) >= 50
    assert len(channel.read(12_345, 456_789, level=2)) < 50


def test_level_for_is_0_outside_the_channel(odd_bundle):
    channel = _channel(odd_bundle, 5)
    assert channel.level_for(0, 1_000_000, bins=1) == 0
    assert channel.level_for(5_000_000, 6_000_000, bins=1) == 0


def test_level_for_is_0_on_a_channel_with_no_samples(odd_bundle):
    assert _channel(odd_bundle, 7).level_for(bins=1) == 0


def test_level_for_rejects_float_bounds_on_a_channel_without_levels(
    odd_bundle,
):
    with pytest.raises(TypeError, match="start_us must be an integer"):
        _channel(odd_bundle, 7).level_for(1.5, bins=1)


def test_level_for_rejects_a_float_end(odd_bundle):
    with pytest.raises(TypeError, match="end_us must be an integer"):
        _channel(odd_bundle, 0).level_for(0, 2.0, bins=1)


def test_level_for_rejects_bins_below_1(odd_bundle):
    with pytest.raises(ValueError, match="bins must be at least 1, got 0"):
        _channel(odd_bundle, 0).level_for(bins=0)


@pytest.mark.parametrize("bins", [2.5, float("nan"), "3"])
@pytest.mark.parametrize("key", [0, 7])
def test_level_for_rejects_a_bins_value_that_is_not_an_integer(
    odd_bundle, key, bins
):
    with pytest.raises(TypeError, match="bins must be an integer"):
        _channel(odd_bundle, key).level_for(bins=bins)


def test_level_for_accepts_numpy_integer_bins(odd_bundle):
    assert _channel(odd_bundle, 0).level_for(bins=np.int64(63)) == 2


def test_level_for_accepts_numpy_integer_bounds(odd_bundle):
    channel = _channel(odd_bundle, 0)
    assert channel.level_for(np.int64(0), np.uint32(_ODD_END_US), bins=63) == 2


# Spans


def test_spans_read_count_on_the_coarsest_level(odd_bundle):
    # Bins 6 and 8 of level 2 are NaN but count finite samples.
    assert _channel(odd_bundle, 0).spans() == [
        (0, 218_750),
        (250_000, _ODD_END_US),
    ]


def test_spans_read_minmax_on_a_level_without_count(odd_bundle):
    assert _channel(odd_bundle, 0).spans(level=1) == [
        (0, 195_313),
        (273_437, _ODD_END_US),
    ]


def test_spans_count_a_bin_with_one_finite_minmax_value_as_data():
    nodes = _hand_nodes(levels=1)
    minmax = np.zeros((16, 2), np.float32)
    minmax[3] = np.nan
    minmax[5, 0] = np.nan
    nodes[f"1/{MINMAX_KEY}"] = minmax
    assert _hand_parse(nodes).spans(level=1) == [
        (0, 23_438),
        (31_250, 125_000),
    ]


def test_spans_at_level_0_read_the_raw_samples(odd_bundle):
    assert _channel(odd_bundle, 0).spans(level=0) == [
        (0, 195_313),
        (273_437, _ODD_END_US),
    ]


def test_spans_of_a_channel_without_levels_read_the_raw_samples(odd_bundle):
    assert _channel(odd_bundle, 5).spans() == [(ODD_OFFSET_US, 4_000_000)]


def test_spans_clamp_to_the_window(odd_bundle):
    spans = _channel(odd_bundle, 0).spans(100_000, 300_000, level=1)
    assert spans == [(100_000, 195_313), (273_437, 300_000)]
    assert all(type(bound) is int for span in spans for bound in span)


def test_spans_clamp_numpy_integer_bounds_to_python_integers(odd_bundle):
    spans = _channel(odd_bundle, 0).spans(np.int64(100_000), np.int64(300_000))
    assert spans == [(100_000, 218_750), (250_000, 300_000)]
    assert all(type(bound) is int for span in spans for bound in span)


def test_spans_accept_a_numpy_integer_level_and_gap(odd_bundle):
    channel = _channel(odd_bundle, 0)
    spans = channel.spans(level=np.int64(1), gap_us=np.uint32(100_000))
    assert spans == channel.spans(level=1, gap_us=100_000)
    assert all(type(bound) is int for span in spans for bound in span)


def test_spans_clamp_to_the_channel_end(odd_bundle):
    spans = _channel(odd_bundle, 0).spans(1_900_000, 10**9, level=2)
    assert spans == [(1_900_000, _ODD_END_US)]


def test_spans_join_a_gap_no_wider_than_gap_us(odd_bundle):
    channel = _channel(odd_bundle, 0)
    assert channel.spans(gap_us=31_250) == [(0, _ODD_END_US)]
    assert len(channel.spans(gap_us=31_249)) == 2


def test_spans_outside_the_channel_are_empty(odd_bundle):
    channel = _channel(odd_bundle, 5)
    assert channel.spans(0, 1_000_000) == []
    assert channel.spans(5_000_000, 6_000_000) == []
    assert channel.spans(0, ODD_OFFSET_US) == []


def test_spans_of_a_channel_with_no_samples_are_empty(odd_bundle):
    assert _channel(odd_bundle, 7).spans() == []


def test_spans_reject_float_bounds(odd_bundle):
    with pytest.raises(TypeError, match="end_us must be an integer"):
        _channel(odd_bundle, 0).spans(0, 1.0)


def test_spans_reject_a_negative_gap(odd_bundle):
    with pytest.raises(ValueError, match="gap_us must be at least 0, got -1"):
        _channel(odd_bundle, 0).spans(gap_us=-1)


def test_spans_reject_a_float_gap(odd_bundle):
    with pytest.raises(TypeError, match="gap_us must be an integer count"):
        _channel(odd_bundle, 0).spans(gap_us=1.5)


def test_spans_reject_a_level_the_channel_does_not_hold(odd_bundle):
    with pytest.raises(ValueError, match="channel 0 has levels 1 through 2"):
        _channel(odd_bundle, 0).spans(level=3)


# Columns


def test_samples_columns_hold_time_us_and_values(odd_bundle):
    samples = _channel(odd_bundle, 0).read(0, 10_000)
    columns = samples.columns()
    assert list(columns) == ["time_us", "values"]
    np.testing.assert_array_equal(columns["time_us"], samples.time_us)
    assert columns["values"] is samples.values


def test_bins_columns_hold_min_and_max_then_every_statistic(odd_bundle):
    bins = _channel(odd_bundle, 2).read(300_000, 1_000_000, level=1)
    columns = bins.columns()
    assert list(columns) == ["time_us", "min", "max", "mean", "var", "count"]
    np.testing.assert_array_equal(columns["time_us"], bins.time_us)
    for name in ("min", "max", MEAN_KEY, VAR_KEY, COUNT_KEY):
        assert columns[name] is getattr(bins, name)
    assert columns[COUNT_KEY].dtype == np.uint32


def test_bins_columns_omit_the_statistics_a_level_lacks(odd_bundle):
    channel = _channel(odd_bundle, 0)
    assert list(channel.read(level=1).columns()) == ["time_us", "min", "max"]
    assert list(channel.read(level=2).columns()) == [
        "time_us",
        "min",
        "max",
        "count",
    ]


def test_columns_of_an_empty_read_are_empty_arrays(odd_bundle):
    channel = _channel(odd_bundle, 2)
    for result in (channel.read(0, 0), channel.read(0, 0, level=1)):
        columns = result.columns()
        assert all(values.shape == (0,) for values in columns.values())


@pytest.mark.parametrize(
    ("level", "method", "package"),
    [
        (None, "to_series", "pandas"),
        (None, "to_xarray", "xarray"),
        (1, "to_dataframe", "pandas"),
        (1, "to_xarray", "xarray"),
    ],
    ids=["samples-series", "samples-xarray", "bins-dataframe", "bins-xarray"],
)
def test_conversions_raise_import_error_naming_the_missing_package(
    odd_bundle, monkeypatch, level, method, package
):
    result = _channel(odd_bundle, 2).read(0, 10_000, level=level)
    monkeypatch.setitem(sys.modules, package, None)
    with pytest.raises(ImportError) as info:
        getattr(result, method)()
    assert info.value.name == package


# Validation


def test_parse_accepts_a_hand_built_channel():
    channel = _hand_parse(_hand_nodes())
    assert channel.name == "hand"
    assert channel.id == "hand-0"
    assert list(channel.levels) == [1, 2]
    assert channel.levels[2].bins == 4


def test_parse_ignores_unknown_nodes_and_keys():
    nodes = _hand_nodes()
    nodes["0"] = level_group_attrs(1.0)
    nodes["01"] = level_group_attrs(1.0)
    nodes["notes"] = {}
    nodes["1/extra"] = {}
    nodes["1/median"] = np.zeros(16, np.float32)
    nodes["2/0"] = np.zeros(3, np.float32)
    nodes["index"] = np.zeros(3, np.int64)
    attributes = _hand_attrs()
    attributes[TSZ_KEY]["future_key"] = "unknown"
    attributes["other_convention"] = {"key": 1}
    channel = _hand_parse(nodes, attributes)
    assert list(channel.levels) == [1, 2]


def test_parse_ignores_arrays_named_0_and_01():
    nodes = _hand_nodes()
    nodes["0"] = np.zeros(3, np.float32)
    nodes["01"] = np.zeros(3, np.float32)
    assert list(_hand_parse(nodes).levels) == [1, 2]


def test_parse_accepts_dtypes_of_the_right_kind_at_any_width():
    num_samples = _HAND_SAMPLES
    nodes = _hand_nodes(levels=1)
    nodes[RAW_KEY] = np.arange(num_samples, dtype=np.float64)
    nodes[f"1/{MINMAX_KEY}"] = np.zeros((16, 2), np.float64)
    nodes[f"1/{MEAN_KEY}"] = np.zeros(16, np.float16)
    nodes[f"1/{VAR_KEY}"] = np.zeros(16, np.float64)
    nodes[f"1/{COUNT_KEY}"] = np.zeros(16, np.int64)
    level = _hand_parse(nodes).levels[1]
    assert level.count.dtype == np.int64
    assert level.mean.dtype == np.float16


def test_parse_accepts_a_period_within_the_tolerance():
    nodes = _hand_nodes(levels=1)
    stated = level_period_us(_HAND_RATE_HZ, 4) * (1 + 5e-13)
    nodes["1"] = level_group_attrs(stated)
    assert _hand_parse(nodes).levels[1].period_us == stated


@pytest.mark.parametrize("excess", [2e-12, 1e-9, 1e-8])
def test_parse_rejects_a_period_outside_the_tolerance(excess):
    nodes = _hand_nodes(levels=1)
    nodes["1"] = level_group_attrs(
        level_period_us(_HAND_RATE_HZ, 4) * (1 + excess)
    )
    with pytest.raises(
        ValueError, match=r"channel 0 level 1: tsz\.period_us must be"
    ):
        _hand_parse(nodes)


def _hand_parse_30_khz(stated_us):
    nodes = _hand_nodes(levels=0)
    nodes["1"] = level_group_attrs(stated_us)
    attributes = continuous_channel_attrs(
        "hand", "hand-0", 30_000.0, 0, "uV", None
    )
    nodes["1/minmax"] = np.zeros((level_bins(_HAND_SAMPLES, 4), 2), np.float32)
    return _hand_parse(nodes, attributes)


def test_parse_accepts_a_30_khz_period_rounded_to_13_digits():
    channel = _hand_parse_30_khz(133.3333333333333)
    assert channel.levels[1].samples_per_bin == 4


def test_parse_rejects_a_30_khz_period_off_by_a_part_in_a_billion():
    with pytest.raises(ValueError, match=r"level 1: tsz\.period_us must be"):
        _hand_parse_30_khz(133.3333333333333 * (1 + 1e-9))


def test_parse_rejects_a_level_as_wide_as_2_to_the_32_samples():
    nodes = _hand_nodes(levels=0)
    nodes["1"] = level_group_attrs(level_period_us(_HAND_RATE_HZ, 2**32))
    nodes["1/minmax"] = np.zeros((1, 2), np.float32)
    with pytest.raises(ValueError, match="from 2 to 4294967295"):
        _hand_parse(nodes)


def test_parse_accepts_a_ladder_that_is_not_a_power_of_4():
    channel = _hand_parse(_hand_nodes(widths=[2, 10]))
    assert [level.samples_per_bin for level in channel.levels.values()] == [
        2,
        10,
    ]
    assert [level.bins for level in channel.levels.values()] == [32, 7]


def test_read_reports_the_period_and_bin_count_of_a_10x_level():
    channel = _hand_parse(_hand_nodes(widths=[10, 100]))
    bins = channel.read(level=2)
    assert bins.period_us == level_period_us(_HAND_RATE_HZ, 100)
    assert len(bins.min) == 1


def test_parse_rejects_a_period_that_is_not_a_whole_number_of_samples():
    nodes = _hand_nodes(levels=1)
    nodes["1"] = level_group_attrs(2.5e6 / _HAND_RATE_HZ)
    message = (
        "channel 0 level 1: tsz.period_us must be a whole number of "
        "samples, from 2 to 4294967295, at rate_hz 512.0, within 1e-12 "
        "relative, got 4882.8125"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _hand_parse(nodes)


def test_parse_rejects_a_level_one_sample_wide():
    nodes = _hand_nodes(levels=1)
    nodes["1"] = level_group_attrs(1e6 / _HAND_RATE_HZ)
    message = (
        "channel 0 level 1: tsz.period_us must be a whole number of "
        "samples, from 2 to 4294967295, at rate_hz 512.0, within 1e-12 "
        "relative, got 1953.125"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _hand_parse(nodes)


def test_parse_rejects_a_level_that_is_not_a_multiple_of_the_level_below():
    message = (
        "channel 0 level 2: tsz.period_us must be an integer multiple of "
        "level 1's 7812.5, at least 2 times it, got 11718.75"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _hand_parse(_hand_nodes(widths=[4, 6]))


def test_parse_rejects_a_level_as_wide_as_the_level_below():
    message = (
        "channel 0 level 2: tsz.period_us must be an integer multiple of "
        "level 1's 7812.5, at least 2 times it, got 7812.5"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _hand_parse(_hand_nodes(widths=[4, 4]))


def test_parse_lists_levels_past_9_in_level_order():
    root = _hand_build(_hand_nodes(num_samples=4**10, levels=11))
    zarr.consolidate_metadata(root.store)
    assert list(_channel(root.store, 0).levels) == list(range(1, 12))


def test_parse_counts_a_short_last_bin():
    channel = _hand_parse(_hand_nodes(num_samples=65, levels=3))
    assert [level.bins for level in channel.levels.values()] == [17, 5, 2]


def test_parse_rejects_channel_attributes_without_rate_hz():
    attributes = _hand_attrs()
    del attributes[TSZ_KEY]["rate_hz"]
    with pytest.raises(ValueError, match=r"channel 0: tsz\.rate_hz is missing"):
        _hand_parse(_hand_nodes(), attributes)


def test_parse_rejects_a_channel_without_raw():
    nodes = _hand_nodes()
    del nodes[RAW_KEY]
    message = "channel 0: raw is missing; it must be a rank-1 floating-point"
    with pytest.raises(ValueError, match=re.escape(message)):
        _hand_parse(nodes)


def test_parse_rejects_a_group_named_raw():
    nodes = _hand_nodes()
    nodes[RAW_KEY] = {}
    with pytest.raises(ValueError, match="channel 0: raw is missing"):
        _hand_parse(nodes)


@pytest.mark.parametrize(
    ("data", "found"),
    [
        (np.zeros((64, 2), np.float32), "shape (64, 2) and dtype float32"),
        (np.zeros(64, np.int16), "shape (64,) and dtype int16"),
    ],
)
def test_parse_rejects_raw_that_is_not_rank_1_floating_point(data, found):
    nodes = _hand_nodes()
    nodes[RAW_KEY] = data
    message = (
        f"channel 0: raw must be a rank-1 floating-point array, got {found}"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _hand_parse(nodes)


def test_parse_rejects_a_level_stored_as_an_array():
    nodes = _hand_nodes()
    nodes["3"] = np.zeros((1, 2), np.float32)
    message = (
        "channel 0: level 3 is an array; a level must be a group that holds "
        "minmax"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _hand_parse(nodes)


@pytest.mark.parametrize(
    ("missing", "found"), [(("1",), "2, 3, 4"), (("3",), "1, 2, 4")]
)
def test_parse_rejects_levels_with_a_gap(missing, found):
    nodes = _hand_nodes(levels=4)
    for level in missing:
        del nodes[level]
        del nodes[f"{level}/{MINMAX_KEY}"]
    message = (
        f"channel 0: levels must be numbered from 1 with no gap, got {found}"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _hand_parse(nodes)


def test_parse_rejects_a_level_without_period_us():
    nodes = _hand_nodes()
    nodes["2"] = {TSZ_KEY: {}}
    with pytest.raises(
        ValueError, match=r"channel 0 level 2: tsz\.period_us is missing"
    ):
        _hand_parse(nodes)


def test_parse_rejects_a_period_that_does_not_match_the_rate():
    nodes = _hand_nodes()
    nodes["2"] = level_group_attrs(31_281.25)
    message = (
        "channel 0 level 2: tsz.period_us must be a whole number of "
        "samples, from 2 to 4294967295, at rate_hz 512.0, within 1e-12 "
        "relative, got 31281.25"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _hand_parse(nodes)


def test_parse_rejects_a_level_without_minmax():
    nodes = _hand_nodes()
    del nodes[f"1/{MINMAX_KEY}"]
    message = (
        "channel 0 level 1: minmax is missing; it must be a floating-point "
        "array of shape (16, 2)"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _hand_parse(nodes)


@pytest.mark.parametrize(
    ("name", "data", "expected"),
    [
        (
            MINMAX_KEY,
            np.zeros((15, 2), np.float32),
            "a floating-point array of shape (16, 2)",
        ),
        (
            MINMAX_KEY,
            np.zeros((16, 3), np.float32),
            "a floating-point array of shape (16, 2)",
        ),
        (
            MINMAX_KEY,
            np.zeros(16, np.float32),
            "a floating-point array of shape (16, 2)",
        ),
        (
            MINMAX_KEY,
            np.zeros((16, 2), np.int32),
            "a floating-point array of shape (16, 2)",
        ),
        (
            MEAN_KEY,
            np.zeros(15, np.float32),
            "a floating-point array of shape (16,)",
        ),
        (
            MEAN_KEY,
            np.zeros((16, 1), np.float32),
            "a floating-point array of shape (16,)",
        ),
        (
            VAR_KEY,
            np.zeros(16, np.int32),
            "a floating-point array of shape (16,)",
        ),
        (
            VAR_KEY,
            np.zeros(17, np.float32),
            "a floating-point array of shape (16,)",
        ),
        (
            COUNT_KEY,
            np.zeros(16, np.float32),
            "an integer array of shape (16,)",
        ),
        (COUNT_KEY, np.zeros(16, np.bool_), "an integer array of shape (16,)"),
        (COUNT_KEY, np.zeros(15, np.uint32), "an integer array of shape (16,)"),
    ],
)
def test_parse_rejects_a_member_of_the_wrong_shape_or_dtype_kind(
    name, data, expected
):
    nodes = _hand_nodes()
    nodes[f"1/{name}"] = data
    message = (
        f"channel 0 level 1: {name} must be {expected}, got shape "
        f"{data.shape} and dtype {data.dtype}"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _hand_parse(nodes)


# Runs

_GAP_RATE_HZ = 1000.0
_GAP_ROWS = ((0, 4, 0), (7, 3, 6700))
"""Samples 0 through 3 at 0 us, a gap of grid samples 4 through 6, and samples
7 through 9 at 6700 us, 300 us before their grid time."""
_GAP_NAN = np.nan


def _gap_nodes(rows=_GAP_ROWS, raw=None, runs=True):
    """Return the nodes of a 10-sample channel at 1000 Hz that holds runs."""
    if raw is None:
        raw = [0, 1, 2, 3, _GAP_NAN, _GAP_NAN, _GAP_NAN, 7, 8, 9]
    nodes = {
        RAW_KEY: np.array(raw, dtype=np.float32),
        "1": level_group_attrs(level_period_us(_GAP_RATE_HZ, 2)),
        f"1/{MINMAX_KEY}": np.array(
            [
                [0, 0],
                [0, 0],
                [_GAP_NAN, _GAP_NAN],
                [_GAP_NAN, _GAP_NAN],
                [0, 0],
            ],
            dtype=np.float32,
        ),
    }
    if runs:
        nodes[RUNS_KEY] = np.array(rows, dtype=np.int64)
    return nodes


def _gap_attrs(offset_us=0):
    return continuous_channel_attrs(
        "gap", "gap-0", _GAP_RATE_HZ, offset_us, "uV", None
    )


def _gap_channel(**options):
    return _hand_parse(_gap_nodes(**options), _gap_attrs())


def test_a_channel_without_a_runs_array_has_no_runs(odd_bundle):
    channel = _channel(odd_bundle, 0)
    assert channel.runs_array is None
    assert channel.runs is None
    assert channel.read().runs is None


def test_the_runs_property_returns_a_read_only_int64_table():
    table = _gap_channel().runs
    assert table.dtype == np.int64
    assert table.tolist() == [list(row) for row in _GAP_ROWS]
    assert not table.flags.writeable


def test_the_runs_property_reads_the_array_once():
    channel = _gap_channel()
    assert channel.runs is channel.runs


def test_a_channel_with_runs_reports_the_exact_end_of_its_last_run():
    channel = _gap_channel()
    assert channel.start_us == 0
    assert channel.end_us == 9700
    assert len(channel.read(channel.start_us, channel.end_us)) == 10


def test_a_channel_that_ends_on_a_late_resume_includes_its_last_sample():
    rows = ((0, 4, 0), (7, 3, 7300))
    channel = _gap_channel(rows=rows)
    assert channel.end_us == 10_300
    assert len(channel.read(channel.start_us, channel.end_us)) == 10


def test_sample_times_are_exact_across_a_resume():
    samples = _gap_channel().read()
    assert samples.runs is not None
    assert samples.time_us.tolist() == [
        0.0,
        1000.0,
        2000.0,
        3000.0,
        4000.0,
        5000.0,
        6000.0,
        6700.0,
        7700.0,
        8700.0,
    ]
    assert samples.time_s[7] == 0.0067


def test_samples_in_the_gap_take_their_grid_time_and_stay_in_order():
    rows = ((0, 4, 0), (7, 3, 7300))
    times = _gap_channel(rows=rows).read().time_us
    assert times[4:7].tolist() == [4000.0, 5000.0, 6000.0]
    assert np.all(np.diff(times) > 0)


def test_the_exact_time_reaches_the_columns():
    samples = _gap_channel().read(6000, 8000)
    assert samples.columns()["time_us"].tolist() == [6000.0, 6700.0, 7700.0]


def test_a_window_inside_a_gap_reads_the_nan_samples_of_the_grid():
    samples = _gap_channel().read(4200, 4800)
    assert (samples.first, len(samples)) == (4, 1)
    assert np.isnan(samples.values).all()
    assert samples.time_us.tolist() == [4000.0]


def test_a_window_across_a_seam_holds_both_runs_and_the_gap():
    samples = _gap_channel().read(3500, 7000)
    assert (samples.first, len(samples)) == (3, 5)
    np.testing.assert_array_equal(
        samples.values, [3, np.nan, np.nan, np.nan, 7]
    )
    assert samples.time_us.tolist() == [3000.0, 4000.0, 5000.0, 6000.0, 6700.0]


def test_a_window_that_starts_at_a_resume_starts_at_its_first_sample():
    assert _gap_channel().read(6700, 6701).first == 7
    assert _gap_channel().read(6699, 6701).first == 6


def test_a_window_clamps_inside_the_run_at_its_end():
    channel = _gap_channel()
    assert (channel.read(3990, 4000).first, len(channel.read(3990, 4000))) == (
        3,
        1,
    )
    assert len(channel.read(8700, 9700)) == 1
    assert len(channel.read(9700, 50_000)) == 0
    assert len(channel.read(-50_000, 0)) == 0


def test_a_window_before_the_first_run_reads_nothing():
    rows = ((0, 4, 2500), (7, 3, 9200))
    channel = _hand_parse(_gap_nodes(rows), _gap_attrs(2500))
    assert len(channel.read(0, 2500)) == 0
    assert channel.read(0, 2501).first == 0


def test_bins_stay_on_the_grid_when_the_channel_has_runs():
    channel = _gap_channel()
    bins = channel.read(level=1)
    assert bins.time_us.tolist() == [0.0, 2000.0, 4000.0, 6000.0, 8000.0]
    assert channel.read(6700, 7000, level=1).first == 3


def test_the_level_choice_ignores_the_runs():
    channel = _gap_channel()
    assert channel.level_for(0, 10_000, bins=5) == 1
    assert channel.level_for(0, 10_000, bins=6) == 0


def test_spans_at_the_raw_level_start_and_end_at_exact_times():
    channel = _gap_channel()
    assert channel.spans(level=0) == [(0, 4000), (6700, 9700)]
    assert channel.spans(6800, 8000, level=0) == [(6800, 8000)]
    assert channel.spans(level=0, gap_us=2700) == [(0, 9700)]
    assert channel.spans(level=0, gap_us=2699) == [(0, 4000), (6700, 9700)]


def test_spans_at_a_level_stay_on_the_grid():
    channel = _gap_channel(raw=[0, 1, 2, 3, np.nan, np.nan, np.nan, 7, 8, 9])
    assert channel.spans(level=1) == [(0, 4000), (8000, 9700)]


def test_a_nan_inside_a_run_splits_a_raw_span_at_the_sample():
    raw = [0, 1, np.nan, 3, np.nan, np.nan, np.nan, 7, 8, 9]
    spans = _gap_channel(raw=raw).spans(level=0)
    assert spans == [(0, 2000), (3000, 4000), (6700, 9700)]


def test_a_value_between_two_runs_fails_a_raw_read_that_covers_it():
    raw = [0, 1, 2, 3, np.nan, 5.0, np.nan, 7, 8, 9]
    channel = _gap_channel(raw=raw)
    with pytest.raises(
        ValueError,
        match="channel 0: raw sample 5 lies outside every run and holds "
        "5.0; it must be NaN",
    ):
        channel.read()
    with pytest.raises(ValueError, match="raw sample 5 lies outside"):
        channel.spans(level=0)
    assert len(channel.read(0, 4000)) == 4


def test_a_value_between_two_runs_fails_a_stack_read_that_covers_it():
    raw = [0, 1, 2, 3, np.nan, 5.0, np.nan, 7, 8, 9]
    channel = _gap_channel(raw=raw)
    with pytest.raises(ValueError, match="raw sample 5 lies outside"):
        read_continuous.read_entries(channel, None, None, 0, RAW_KEY)


def test_a_stack_read_of_raw_stays_on_the_grid_when_the_channel_has_runs():
    channel = _gap_channel()
    first, period_us, rows = read_continuous.read_entries(
        channel, 6700, 7700, 0, RAW_KEY
    )
    assert (first, period_us, len(rows)) == (6, 1000.0, 2)


def test_a_runs_array_of_the_wrong_dtype_fails_at_parse():
    nodes = _gap_nodes()
    nodes[RUNS_KEY] = np.array(_GAP_ROWS, dtype=np.int32)
    with pytest.raises(
        ValueError,
        match=r"channel 0: runs must be an int64 array of shape \(runs, 3\) "
        r"with at least 1 run, got shape \(2, 3\) and dtype int32",
    ):
        _hand_parse(nodes, _gap_attrs())


@pytest.mark.parametrize(
    "table",
    [
        np.zeros((0, 3), np.int64),
        np.zeros((2, 2), np.int64),
        np.zeros((2, 4), np.int64),
        np.zeros((6,), np.int64),
        np.zeros((1, 3, 1), np.int64),
    ],
)
def test_a_runs_array_of_the_wrong_shape_fails_at_parse(table):
    nodes = _gap_nodes()
    nodes[RUNS_KEY] = table
    with pytest.raises(ValueError, match="channel 0: runs must be an int64"):
        _hand_parse(nodes, _gap_attrs())


def test_row_0_that_misses_the_offset_fails_on_first_access():
    channel = _gap_channel(rows=((0, 4, 5), (7, 3, 6700)))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "channel 0: runs row 0 is [0, 4, 5]; it must be [0, n_0, 0]"
        ),
    ):
        channel.runs
    with pytest.raises(ValueError, match="runs row 0"):
        channel.read()


def test_a_run_of_no_samples_fails_on_first_access():
    channel = _gap_channel(rows=((0, 4, 0), (7, 0, 6700)))
    with pytest.raises(ValueError, match="runs row 1 holds 0 samples"):
        channel.runs


def test_first_samples_that_do_not_ascend_fail_on_first_access():
    channel = _gap_channel(rows=((0, 3, 0), (8, 1, 8000), (5, 1, 9000)))
    with pytest.raises(ValueError, match="runs row 2 has first sample 5"):
        channel.runs


def test_times_that_do_not_ascend_fail_on_first_access():
    channel = _gap_channel(rows=((0, 3, 0), (5, 2, 5000), (8, 2, 4000)))
    with pytest.raises(ValueError, match="runs row 2 has time 4000"):
        channel.runs


def test_overlapping_runs_fail_on_first_access():
    channel = _gap_channel(rows=((0, 5, 0), (3, 7, 3000)))
    with pytest.raises(
        ValueError, match="runs row 1 starts at sample 3, inside run 0"
    ):
        channel.runs


def test_a_run_that_starts_before_the_previous_run_ends_in_time_fails_on_first_access():
    nodes = _gap_nodes(
        rows=((0, 10, 0), (20, 10, 20_900), (30, 5, 29_100)),
        raw=[np.nan] * 35,
    )
    nodes[f"1/{MINMAX_KEY}"] = np.full((18, 2), np.nan, dtype=np.float32)
    channel = _hand_parse(nodes, _gap_attrs())
    with pytest.raises(
        ValueError, match="runs row 2 has time 29100 us, before the samples"
    ):
        channel.runs


def test_a_last_run_that_misses_the_end_of_raw_fails_on_first_access():
    channel = _gap_channel(rows=((0, 4, 0), (7, 2, 6700)))
    with pytest.raises(
        ValueError,
        match="the last run, row 1, ends at sample 9, but raw holds 10",
    ):
        channel.runs


def test_a_time_a_period_from_its_grid_time_fails_on_first_access():
    channel = _gap_channel(rows=((0, 4, 0), (7, 3, 8000)))
    with pytest.raises(
        ValueError,
        match="runs row 1 has time 8000 us, .* under one sample period of "
        "1000.0 us",
    ):
        channel.runs


def test_a_channel_with_one_run_holds_a_table_that_reads_like_the_grid():
    nodes = _gap_nodes(rows=((0, 10, 0),), raw=list(range(10)))
    channel = _hand_parse(nodes, _gap_attrs())
    plain = _hand_parse(
        _gap_nodes(runs=False, raw=list(range(10))), _gap_attrs()
    )
    for start, end in [(None, None), (2500, 7500), (0, 1), (9000, 12_000)]:
        exact = channel.read(start, end)
        grid = plain.read(start, end)
        assert exact.first == grid.first
        np.testing.assert_array_equal(exact.values, grid.values)
        np.testing.assert_array_equal(exact.time_us, grid.time_us)
    assert channel.end_us == plain.end_us


# Runs bundle


def test_the_runs_bundle_channel_reads_its_table(runs_bundle):
    channel = _channel(runs_bundle, 0)
    assert channel.runs.tolist() == [list(row) for row in RUNS_TABLE]
    assert _channel(runs_bundle, 1).runs is None


def test_the_runs_bundle_gives_the_exact_time_of_each_run_start(runs_bundle):
    samples = _channel(runs_bundle, 0).read()
    assert len(samples) == RUNS_GRID_LENGTH
    time_us = samples.time_us
    starts = [first for first, _, _ in RUNS_TABLE]
    assert time_us[starts[0]] == 0.0
    assert time_us[starts[1]] == RUNS_RESUME_US
    source_starts = [RUNS_START_US, RUNS_START_US + RUNS_RESUME_US]
    # The reference time of the bundle is the first sample of channel 0.
    epoch_us = RUNS_START_US
    assert (epoch_us + time_us[starts]).tolist() == source_starts


def test_the_runs_bundle_times_follow_the_period_inside_each_run(runs_bundle):
    samples = _channel(runs_bundle, 0).read()
    period = samples.period_us
    first, second = runs_samples()
    time_us = samples.time_us
    inside_first = time_us[: RUNS_LENGTHS[0]]
    inside_second = time_us[RUNS_TABLE[1][0] :]
    np.testing.assert_allclose(
        inside_first, np.arange(RUNS_LENGTHS[0]) * period, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        inside_second,
        RUNS_RESUME_US + np.arange(RUNS_LENGTHS[1]) * period,
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(samples.values[: RUNS_LENGTHS[0]], first)
    np.testing.assert_array_equal(samples.values[RUNS_TABLE[1][0] :], second)


def test_the_runs_bundle_window_at_the_resume_starts_at_the_first_sample(
    runs_bundle,
):
    channel = _channel(runs_bundle, 0)
    samples = channel.read(RUNS_RESUME_US, RUNS_RESUME_US + 4000)
    assert samples.first == RUNS_TABLE[1][0]
    assert samples.time_us[0] == RUNS_RESUME_US
    assert (
        channel.read(RUNS_RESUME_US - 1, RUNS_RESUME_US + 1).first
        == (RUNS_TABLE[1][0])
    )
    assert channel.read(RUNS_RESUME_US - 1500, RUNS_RESUME_US).first == (
        RUNS_TABLE[1][0] - 1
    )


def test_the_runs_bundle_spans_hold_the_two_recorded_spans(runs_bundle):
    channel = _channel(runs_bundle, 0)
    period = channel.period_us
    ends_first = -(-RUNS_LENGTHS[0] * 1_000_000 // 512)
    ends_second = -(
        -(RUNS_RESUME_US * 512 + RUNS_LENGTHS[1] * 1_000_000) // 512
    )
    assert channel.spans(level=0) == [
        (0, ends_first),
        (RUNS_RESUME_US, ends_second),
    ]
    assert period == 1953.125


def test_the_runs_bundle_end_is_the_exact_end_of_the_last_run(runs_bundle):
    channel = _channel(runs_bundle, 0)
    assert channel.end_us == RUNS_RESUME_US + 4_000_000
    assert len(channel.read(channel.start_us, channel.end_us)) == (
        RUNS_GRID_LENGTH
    )
