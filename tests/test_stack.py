import dataclasses
import re
import sys
import threading

import numpy as np
import pytest

from tests.conftest import ODD_OFFSET_US, write_pair_bundle
from tests.test_read_continuous import _hand_build, _hand_nodes
from timeseries_zarr import read_continuous
from timeseries_zarr.convert import CHANNEL_DIMENSION, TIME_COLUMN
from timeseries_zarr.extras import missing_extra_message
from timeseries_zarr.read_bundle import open_bundle
from timeseries_zarr.read_continuous import parse_continuous_channel
from timeseries_zarr.stack import Stack, check_shared_grid, stack_channels
from timeseries_zarr.zarr_io import descendants

_RAW_PERIOD_US = 1953.125
_LEVEL_1_PERIOD_US = 7812.5
_PAIR_LEVEL_1_BINS = (1024, 750)


@pytest.fixture(scope="module")
def odd(odd_bundle):
    return open_bundle(odd_bundle)


@pytest.fixture(scope="module")
def mixed(mixed_bundle):
    return open_bundle(mixed_bundle)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """Open the pair bundle of tests/conftest.py."""
    path = tmp_path_factory.mktemp("pair") / "bundle"
    return open_bundle(write_pair_bundle(path))


@pytest.fixture
def pd():
    return pytest.importorskip("pandas")


@pytest.fixture
def xr():
    return pytest.importorskip("xarray")


def _padded(values, length, fill):
    """Return values followed by fill up to length entries."""
    row = np.full(length, fill, dtype=values.dtype)
    row[: values.shape[0]] = values
    return row


def _stack(units=("uV", "uV")):
    return Stack(
        keys=(0, 3),
        names=("LA1", "LA2"),
        units=units,
        level=2,
        member="mean",
        offset_us=500,
        first=4,
        period_us=31250.0,
        values=np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]], np.float32),
    )


def test_raw_stack_pads_the_shorter_channel_with_nan(odd):
    stack = odd.stack([0, 2])
    short = odd.continuous[0].raw[:]
    full = odd.continuous[2].raw[:]
    assert stack.values.shape == (2, 3000)
    assert stack.values.dtype == np.float32
    np.testing.assert_array_equal(stack.values[0], _padded(short, 3000, np.nan))
    np.testing.assert_array_equal(stack.values[1], full)
    assert stack.first == 0
    assert stack.offset_us == 0
    assert stack.period_us == _RAW_PERIOD_US


def test_raw_stack_records_level_0_and_member_none(odd):
    stack = odd.stack([0, 2], member="count")
    assert (stack.level, stack.member) == (0, None)
    assert np.isnan(stack.values[0, 1001:]).all()


@pytest.mark.parametrize("member", [None, "median"])
def test_raw_stack_ignores_member(odd, member):
    channels = [odd.continuous[0], odd.continuous[2]]
    stack = stack_channels(channels, None, None, member=member)
    _assert_same_stack(stack, odd.stack([0, 2]))
    assert stack.member is None


def test_stack_holds_the_keys_names_and_units_in_the_order_given(odd):
    stack = odd.stack([2, 0])
    assert stack.keys == (2, 0)
    assert stack.names == ("full", "short")
    assert stack.units == ("uV", "uV")
    np.testing.assert_array_equal(stack.values[0], odd.continuous[2].raw[:])


def test_stack_starts_every_row_at_the_window_on_the_shared_grid(odd):
    stack = odd.stack([0, 2], 1_900_000, 2_100_000)
    assert stack.first == 972
    assert len(stack) == 104
    short = odd.continuous[0].raw[972:1001]
    np.testing.assert_array_equal(stack.values[0], _padded(short, 104, np.nan))
    np.testing.assert_array_equal(
        stack.values[1], odd.continuous[2].raw[972:1076]
    )


def test_stack_fills_the_row_of_a_channel_that_ends_before_the_window(odd):
    stack = odd.stack([0, 2], 2_000_000, 3_000_000)
    assert stack.first == 1024
    assert stack.values.shape == (2, 512)
    assert np.isnan(stack.values[0]).all()
    np.testing.assert_array_equal(
        stack.values[1], odd.continuous[2].raw[1024:1536]
    )


def test_stack_after_every_channel_is_empty_at_the_longest_end(odd):
    stack = odd.stack([0, 2], 10_000_000, 11_000_000)
    assert stack.first == 3000
    assert stack.values.shape == (2, 0)
    assert stack.time_us.shape == (0,)


def test_stack_before_a_channel_is_empty(odd):
    stack = odd.stack([5], 0, ODD_OFFSET_US)
    assert (stack.first, len(stack)) == (0, 0)
    assert stack.offset_us == ODD_OFFSET_US


def test_stack_of_a_channel_without_samples_is_empty(odd):
    stack = odd.stack([7])
    assert stack.values.shape == (1, 0)
    assert stack.values.dtype == np.float32


def test_level_1_mean_stack_holds_the_mean_of_each_bin(odd):
    stack = odd.stack([2], level=1)
    bins = odd.continuous[2].read(level=1)
    assert (stack.level, stack.member) == (1, "mean")
    assert stack.period_us == _LEVEL_1_PERIOD_US
    np.testing.assert_array_equal(stack.values[0], bins.mean)


@pytest.mark.parametrize(("member", "column"), [("min", 0), ("max", 1)])
def test_level_1_envelope_stack_pads_the_shorter_channel_with_nan(
    odd, member, column
):
    stack = odd.stack([0, 2], level=1, member=member)
    short = odd.continuous[0].levels[1].minmax[:, column]
    full = odd.continuous[2].levels[1].minmax[:, column]
    assert stack.values.shape == (2, 750)
    np.testing.assert_array_equal(stack.values[0], _padded(short, 750, np.nan))
    np.testing.assert_array_equal(stack.values[1], full)


def test_level_2_count_stack_holds_the_count_of_each_bin(odd):
    stack = odd.stack([0], level=2, member="count")
    np.testing.assert_array_equal(
        stack.values[0], odd.continuous[0].levels[2].count[:]
    )


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_count_stack_pads_the_shorter_channel_with_zero(pair):
    stack = pair.stack([0, 1], level=1, member="count")
    longer, shorter = _PAIR_LEVEL_1_BINS
    assert stack.values.shape == (2, longer)
    assert stack.values.dtype == np.uint32
    np.testing.assert_array_equal(
        stack.values[0], pair.continuous[0].levels[1].count[:]
    )
    np.testing.assert_array_equal(
        stack.values[1],
        _padded(pair.continuous[1].levels[1].count[:], longer, 0),
    )
    assert (stack.values[1, shorter:] == 0).all()


def test_mean_stack_pads_the_shorter_channel_with_nan(pair):
    stack = pair.stack([0, 1], level=1)
    longer, shorter = _PAIR_LEVEL_1_BINS
    np.testing.assert_array_equal(
        stack.values[1, :shorter], pair.continuous[1].levels[1].mean[:]
    )
    assert np.isnan(stack.values[1, shorter:]).all()
    assert stack.values.shape == (2, longer)


def test_time_us_at_512_hz_is_exactly_each_index_times_1953_125(odd):
    stack = odd.stack([0, 2])
    np.testing.assert_array_equal(
        stack.time_us, np.arange(3000, dtype=np.float64) * _RAW_PERIOD_US
    )
    np.testing.assert_array_equal(stack.time_s, stack.time_us / 1e6)


def test_time_us_counts_from_first_and_offset_us(odd):
    stack = odd.stack([5], 3_000_000, 3_100_000)
    expected = ODD_OFFSET_US + (stack.first + np.arange(len(stack))) * (
        _RAW_PERIOD_US
    )
    assert stack.first == 512
    np.testing.assert_array_equal(stack.time_us, expected)


def test_time_us_at_30_khz_derives_each_time_from_its_own_index():
    period_us = 1e6 / 30000
    stack = Stack(
        keys=(0,),
        names=("LA1",),
        units=("uV",),
        level=0,
        member=None,
        offset_us=123,
        first=7,
        period_us=period_us,
        values=np.zeros((1, 1000), np.float32),
    )
    np.testing.assert_array_equal(
        stack.time_us, 123 + (7 + np.arange(1000)) * period_us
    )
    np.testing.assert_array_equal(stack.time_s, stack.time_us / 1e6)


def test_columns_hold_time_us_then_one_row_per_decimal_key(odd):
    stack = odd.stack([2, 0], 0, 10_000)
    columns = stack.columns()
    assert list(columns) == [TIME_COLUMN, "2", "0"]
    np.testing.assert_array_equal(columns[TIME_COLUMN], stack.time_us)
    np.testing.assert_array_equal(columns["2"], stack.values[0])
    np.testing.assert_array_equal(columns["0"], stack.values[1])


def test_stack_rejects_channels_of_different_rates(mixed):
    message = (
        "channel 1 has rate_hz 512.0 and channel 0 has 256.0; stack needs "
        "channels that share rate_hz, offset_us, and level 2"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        mixed.stack([0, 1], level=2)


def _hand_channel(key, widths):
    """Return a 512 Hz channel whose levels fold widths samples per bin."""
    channel = _hand_build(_hand_nodes(widths=widths))["0"]
    groups, arrays = descendants(channel)
    return parse_continuous_channel(key, channel.attrs, groups, arrays)


def test_stack_rejects_channels_whose_level_widths_differ():
    channels = [_hand_channel(0, [4, 16]), _hand_channel(1, [16, 64])]
    message = (
        "level 1 of channel 1 holds 16 samples per bin and channel 0 holds "
        "4; stack needs channels whose level has the same width in samples"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        check_shared_grid(channels, 1, "min")
    with pytest.raises(ValueError, match=re.escape(message)):
        stack_channels(channels, None, None, level=1, member="min")


def test_stack_accepts_channels_whose_level_widths_match():
    channels = [_hand_channel(0, [4, 16]), _hand_channel(1, [4, 16])]
    assert check_shared_grid(channels, 1, "min") is None
    assert check_shared_grid(channels, 2, "min") is None


def test_stack_compares_the_widths_of_the_level_it_reads():
    channels = [_hand_channel(0, [4, 16]), _hand_channel(1, [4, 64])]
    assert check_shared_grid(channels, 1, "min") is None
    with pytest.raises(ValueError, match="level 2 of channel 1 holds 64"):
        check_shared_grid(channels, 2, "min")


def test_stack_rejects_channels_of_different_offsets(odd):
    message = (
        "channel 5 has offset_us 2000000 and channel 0 has 0; stack needs "
        "channels that share rate_hz and offset_us"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        odd.stack([0, 5])


def test_stack_rejects_a_channel_without_the_level(odd):
    message = (
        "channel 2 has no level 2; stack needs a level that every channel holds"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        odd.stack([0, 2], level=2, member="count")


def test_stack_rejects_a_level_without_the_member(odd):
    message = (
        "level 1 of channel 0 has no var; stack needs a member that every "
        "channel holds at level 1"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        odd.stack([2, 0], level=1, member="var")


def test_stack_rejects_an_unknown_member_at_a_level(odd):
    message = (
        "member must be one of 'min', 'max', 'mean', 'var', 'count' at "
        "level 1, got 'median'"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        odd.stack([2], level=1, member="median")


def test_stack_rejects_a_level_that_is_not_an_integer(odd):
    with pytest.raises(TypeError, match="level must be an integer, got 1.0"):
        odd.stack([2], level=1.0)


def test_stack_rejects_a_window_that_ends_before_it_starts(odd):
    with pytest.raises(ValueError, match="end_us must be at or after"):
        odd.stack([0, 2], 2_000_000, 1_000_000)


def test_stack_rejects_a_key_given_twice(odd):
    message = "stack needs distinct channels, got channel 0 more than once"
    with pytest.raises(ValueError, match=re.escape(message)):
        odd.stack([0, 2, 0])


def test_stack_rejects_an_empty_key_list(odd):
    with pytest.raises(ValueError, match="stack needs at least one channel"):
        odd.stack([])


def test_stack_raises_key_error_for_an_unknown_key(odd):
    message = (
        "the bundle has no channel 3; its continuous channel keys are "
        "0, 2, 5, 7"
    )
    with pytest.raises(KeyError, match=re.escape(message)):
        odd.stack([0, 3])


def test_stack_rejects_the_key_of_an_event_channel(mixed):
    message = (
        "channel 2 is an event channel; the continuous channel keys are 0, 1"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        mixed.stack([0, 2])


def test_stack_channels_rejects_an_empty_channel_list():
    with pytest.raises(ValueError, match="stack needs at least one channel"):
        stack_channels([], None, None)


def test_stack_accepts_numpy_integer_bounds_and_level(odd):
    stack = odd.stack(
        [0, 2], np.int64(0), np.uint32(50_000), level=np.int64(1), member="max"
    )
    expected = odd.stack([0, 2], 0, 50_000, level=1, member="max")
    np.testing.assert_array_equal(stack.values, expected.values)
    assert (stack.first, stack.level) == (expected.first, 1)
    assert type(stack.level) is int


def test_stack_channels_matches_bundle_stack(odd):
    channels = [odd.continuous[0], odd.continuous[2]]
    direct = stack_channels(channels, 0, 50_000, level=1, member="max")
    through = odd.stack([0, 2], 0, 50_000, level=1, member="max")
    np.testing.assert_array_equal(direct.values, through.values)
    assert (direct.first, direct.keys) == (through.first, through.keys)


def _wrap_reads(monkeypatch, before):
    """Call before with each array that read_continuous.read_rows reads."""
    read_rows = read_continuous.read_rows

    def wrapped(array, start, stop):
        before(array)
        return read_rows(array, start, stop)

    monkeypatch.setattr(read_continuous, "read_rows", wrapped)


def _record_paths(monkeypatch):
    """Record the path of every array that read_continuous.read_rows reads."""
    paths = []
    _wrap_reads(monkeypatch, lambda array: paths.append(array.path))
    return paths


def _assert_same_stack(stack, expected):
    np.testing.assert_array_equal(stack.values, expected.values)
    assert stack.values.dtype == expected.values.dtype
    assert (stack.first, stack.period_us) == (
        expected.first,
        expected.period_us,
    )


@pytest.mark.parametrize("max_workers", [2, 8])
@pytest.mark.parametrize(
    ("fixture", "options"),
    [
        ("odd", {}),
        ("odd", {"level": 1, "member": "max"}),
        ("pair", {"level": 1, "member": "count"}),
    ],
)
def test_stack_does_not_depend_on_max_workers(
    request, fixture, options, max_workers
):
    bundle = request.getfixturevalue(fixture)
    keys = [0, 2] if fixture == "odd" else [0, 1]
    serial = bundle.stack(keys, **options)
    threaded = bundle.stack(keys, **options, max_workers=max_workers)
    _assert_same_stack(threaded, serial)


def test_stack_rejects_max_workers_below_1(odd):
    message = "max_workers must be at least 1, got 0"
    with pytest.raises(ValueError, match=re.escape(message)):
        odd.stack([0, 2], max_workers=0)


@pytest.mark.parametrize("max_workers", [2.5, "2"])
def test_stack_rejects_max_workers_that_is_not_an_integer(odd, max_workers):
    with pytest.raises(TypeError, match="max_workers must be an integer"):
        odd.stack([0, 2], max_workers=max_workers)


def test_stack_accepts_a_numpy_integer_max_workers(odd):
    stack = odd.stack([0, 2], max_workers=np.int64(2))
    _assert_same_stack(stack, odd.stack([0, 2]))


@pytest.mark.parametrize(
    ("fixture", "keys", "member", "array"),
    [("odd", [0, 2], "max", "minmax"), ("pair", [0, 1], "count", "count")],
)
def test_stack_reads_one_array_per_channel(
    request, monkeypatch, fixture, keys, member, array
):
    bundle = request.getfixturevalue(fixture)
    paths = _record_paths(monkeypatch)
    bundle.stack(keys, level=1, member=member, max_workers=2)
    assert sorted(paths) == [f"{key}/1/{array}" for key in keys]


def test_stack_rejects_channels_before_it_reads(odd, monkeypatch):
    paths = _record_paths(monkeypatch)
    with pytest.raises(ValueError, match="has no mean"):
        odd.stack([2, 0], level=1, member="mean", max_workers=2)
    assert paths == []


def _fail_reads(monkeypatch, keys):
    """Raise RuntimeError naming the channel for each read of a key in keys."""

    def fail(array):
        key = int(array.path.partition("/")[0])
        if key in keys:
            raise RuntimeError(f"read of channel {key} failed")

    _wrap_reads(monkeypatch, fail)


@pytest.mark.parametrize(
    ("keys", "failing", "raised"),
    [([2, 0], {0, 2}, 2), ([0, 2], {0, 2}, 0), ([0, 2], {2}, 2)],
)
def test_threaded_stack_raises_the_first_failure_in_key_order(
    odd, monkeypatch, keys, failing, raised
):
    _fail_reads(monkeypatch, failing)
    message = f"read of channel {raised} failed"
    with pytest.raises(RuntimeError, match=message):
        odd.stack(keys, max_workers=2)


def test_threaded_stack_reads_two_channels_at_once(odd, monkeypatch):
    barrier = threading.Barrier(2, timeout=5)
    _wrap_reads(monkeypatch, lambda array: barrier.wait())
    stack = odd.stack([0, 2], max_workers=2)
    assert stack.values.shape == (2, 3000)


def test_stack_with_one_worker_reads_one_channel_at_a_time(odd, monkeypatch):
    barrier = threading.Barrier(2, timeout=0.2)
    _wrap_reads(monkeypatch, lambda array: barrier.wait())
    with pytest.raises(threading.BrokenBarrierError):
        odd.stack([0, 2], max_workers=1)


@pytest.mark.parametrize(("keys", "max_workers"), [([0, 2], 1), ([2], 4)])
def test_stack_with_one_worker_reads_on_the_calling_thread(
    odd, monkeypatch, keys, max_workers
):
    threads = []
    _wrap_reads(
        monkeypatch, lambda array: threads.append(threading.get_ident())
    )
    odd.stack(keys, max_workers=max_workers)
    assert threads
    assert set(threads) == {threading.get_ident()}


def test_stack_channels_matches_bundle_stack_with_max_workers(odd):
    channels = [odd.continuous[0], odd.continuous[2]]
    direct = stack_channels(channels, None, None, max_workers=2)
    _assert_same_stack(odd.stack([0, 2], max_workers=2), direct)


def test_check_shared_grid_accepts_channels_that_share_a_level(odd):
    channels = [odd.continuous[0], odd.continuous[2]]
    assert check_shared_grid(channels, 1, "min") is None
    assert check_shared_grid(channels, 0, None) is None


def test_check_shared_grid_rejects_a_member_at_level_0(odd):
    message = "level 0 holds raw samples and takes member None, got 'mean'"
    with pytest.raises(ValueError, match=re.escape(message)):
        check_shared_grid([odd.continuous[0]], 0, "mean")


def test_check_shared_grid_rejects_member_none_at_a_level(odd):
    with pytest.raises(ValueError, match="at level 1, got None"):
        check_shared_grid([odd.continuous[2]], 1, None)


def test_check_shared_grid_rejects_a_negative_level(odd):
    with pytest.raises(ValueError, match="channel 2 has no level -1"):
        check_shared_grid([odd.continuous[2]], -1, "mean")


def test_stack_fields_are_frozen(odd):
    stack = odd.stack([0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        stack.first = 1


def test_stack_to_dataframe_holds_one_column_per_key(pd, odd):
    stack = odd.stack([0, 2], 0, 100_000)
    frame = stack.to_dataframe()
    assert list(frame.columns) == ["0", "2"]
    assert frame.index.name == TIME_COLUMN
    np.testing.assert_array_equal(frame.index, stack.time_us)
    np.testing.assert_array_equal(frame.to_numpy().T, stack.values)
    assert all(dtype == np.float32 for dtype in frame.dtypes)
    assert frame.attrs == {
        "names": ("short", "full"),
        "units": ("uV", "uV"),
        "level": 0,
        "member": None,
    }


def test_stack_to_dataframe_matches_a_frame_of_its_columns(pd):
    stack = _stack()
    frame = pd.DataFrame(stack.columns()).set_index(TIME_COLUMN)
    pd.testing.assert_frame_equal(frame, stack.to_dataframe())
    assert stack.to_dataframe().attrs["member"] == "mean"


def test_count_stack_to_dataframe_keeps_the_count_dtype(pd, pair):
    frame = pair.stack([0, 1], level=1, member="count").to_dataframe()
    assert all(dtype == np.uint32 for dtype in frame.dtypes)
    assert frame.attrs["level"] == 1


def test_stack_to_xarray_has_channel_and_time_us_dimensions(xr, odd):
    stack = odd.stack([2, 0], 0, 100_000)
    array = stack.to_xarray()
    assert array.dims == (CHANNEL_DIMENSION, TIME_COLUMN)
    assert array.dtype == np.float32
    np.testing.assert_array_equal(array.values, stack.values)
    np.testing.assert_array_equal(array[TIME_COLUMN], stack.time_us)
    assert array[CHANNEL_DIMENSION].values.tolist() == [2, 0]
    assert array.coords["name"].dims == (CHANNEL_DIMENSION,)
    assert array.coords["name"].values.tolist() == ["full", "short"]
    assert array.coords["unit"].values.tolist() == ["uV", "uV"]
    assert array.attrs == {
        "level": 0,
        "period_us": _RAW_PERIOD_US,
        "offset_us": 0,
        "units": "uV",
    }


def test_stack_to_xarray_names_the_member_of_a_level(xr):
    assert _stack().to_xarray().attrs == {
        "level": 2,
        "member": "mean",
        "period_us": 31250.0,
        "offset_us": 500,
        "units": "uV",
    }


def test_stack_to_xarray_omits_units_when_the_channels_differ(xr):
    array = _stack(units=("uV", "mV")).to_xarray()
    assert "units" not in array.attrs
    assert array.coords["unit"].values.tolist() == ["uV", "mV"]


@pytest.mark.parametrize(
    ("package", "method"),
    [("pandas", "to_dataframe"), ("xarray", "to_xarray")],
)
def test_stack_conversion_without_its_package_names_the_extra(
    monkeypatch, package, method
):
    monkeypatch.setitem(sys.modules, package, None)
    message = missing_extra_message(package, package)
    with pytest.raises(ImportError, match=re.escape(message)) as info:
        getattr(_stack(), method)()
    assert info.value.name == package
