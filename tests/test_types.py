import dataclasses
import os
from datetime import UTC, datetime
from typing import get_args

import pytest

from timeseries_zarr.constants import (
    DEFAULT_DECIMATION_FACTOR,
    INNER_CHUNK_SAMPLES,
    TARGET_SHARD_BYTES,
)
from timeseries_zarr.types import (
    ChannelKind,
    ChunkShard,
    EventColumn,
    LevelMembers,
    LevelPlan,
    RecordingMeta,
    WriteOpts,
)


def test_levelplan_construction_stores_fields():
    plan = LevelPlan(level=1, bins=250, period_us=125.0)
    assert plan.level == 1
    assert plan.bins == 250
    assert plan.period_us == 125.0


@pytest.mark.parametrize("level", [1, 7, 15])
def test_levelplan_name_is_the_decimal_level(level):
    assert LevelPlan(level=level, bins=250, period_us=125.0).name == str(level)


def test_levelplan_frozen_rejects_mutation():
    plan = LevelPlan(level=1, bins=250, period_us=125.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.level = 5


def test_levelmembers_holds_one_handle_per_member():
    members = LevelMembers(minmax="mm", mean="m", var="v", count="c")
    assert (members.minmax, members.mean, members.var, members.count) == (
        "mm",
        "m",
        "v",
        "c",
    )


def test_levelmembers_var_may_be_absent():
    assert LevelMembers(minmax="mm", mean="m", var=None, count="c").var is None


def test_levelmembers_frozen_rejects_mutation():
    members = LevelMembers(minmax="mm", mean="m", var=None, count="c")
    with pytest.raises(dataclasses.FrozenInstanceError):
        members.var = "v"


def test_recordingmeta_construction_stores_fields():
    start = datetime(2024, 3, 14, 9, 15, tzinfo=UTC)
    meta = RecordingMeta(
        session_start_time=start,
        subject={"id": "sub-01"},
        session={"id": "ses-01"},
        source={"converter": "timeseries-zarr-py"},
    )
    assert meta.session_start_time is start
    assert meta.subject == {"id": "sub-01"}
    assert meta.session == {"id": "ses-01"}
    assert meta.source == {"converter": "timeseries-zarr-py"}


def test_recordingmeta_frozen_rejects_mutation():
    meta = RecordingMeta(
        session_start_time=datetime(2024, 3, 14, tzinfo=UTC),
        subject={},
        session={},
        source={},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.subject = {"id": "x"}


def test_recordingmeta_defaults_to_no_session_start_and_empty_objects():
    meta = RecordingMeta()
    assert meta.session_start_time is None
    assert meta.reference_time is None
    assert meta.wall_clock is True
    assert meta.subject == {}
    assert meta.session == {}
    assert meta.source == {}


def test_recordingmeta_default_objects_are_distinct_per_instance():
    first, second = RecordingMeta(), RecordingMeta()
    assert first.subject is not second.subject
    assert first.session is not second.session
    assert first.source is not second.source


def test_chunkshard_construction_stores_fields():
    cs = ChunkShard(chunk_shape=(262144,), shard_shape=(1048576,))
    assert cs.chunk_shape == (262144,)
    assert cs.shard_shape == (1048576,)


def test_chunkshard_construction_rank2():
    cs = ChunkShard(chunk_shape=(131072, 2), shard_shape=(524288, 2))
    assert cs.chunk_shape == (131072, 2)
    assert cs.shard_shape == (524288, 2)


def test_chunkshard_frozen_rejects_mutation():
    cs = ChunkShard(chunk_shape=(262144,), shard_shape=(1048576,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cs.chunk_shape = (1,)


def test_channelkind_literal_values():
    assert get_args(ChannelKind.__value__) == ("continuous", "event")


def test_eventcolumn_literal_values():
    assert get_args(EventColumn.__value__) == (
        "durations",
        "labels",
        "text",
        "channels",
        "waveform_mean",
    )


def test_writeopts_defaults():
    opts = WriteOpts()
    assert opts.zstd_level == 5
    assert opts.decimation_factor == DEFAULT_DECIMATION_FACTOR
    assert opts.continuous_decimation_factor is None
    assert opts.event_decimation_factor is None
    assert opts.max_levels is None
    assert opts.level_cap == 15
    assert opts.event_level_cap == 15
    assert opts.min_bins == 1024
    assert opts.inner_len == INNER_CHUNK_SAMPLES
    assert opts.target_shard_bytes == TARGET_SHARD_BYTES
    assert opts.write_var is False
    assert opts.voltage_policy == "convert"


def test_writeopts_max_workers_defaults_to_the_available_cpus():
    opts = WriteOpts()
    assert opts.max_workers is None
    affinity = getattr(os, "sched_getaffinity", None)
    expected = len(affinity(0)) if affinity else os.cpu_count()
    assert opts.worker_count == (expected or 1)


def test_writeopts_worker_count_follows_the_cpu_affinity(monkeypatch):
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: {0, 1}, raising=False
    )
    assert WriteOpts().worker_count == 2


def test_writeopts_worker_count_falls_back_to_the_cpu_count(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert WriteOpts().worker_count == 6
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert WriteOpts().worker_count == 1


def test_writeopts_max_workers_sets_the_worker_count():
    assert WriteOpts(max_workers=3).worker_count == 3


@pytest.mark.parametrize("max_workers", [0, -2])
def test_writeopts_rejects_max_workers_below_1(max_workers):
    with pytest.raises(ValueError, match="max_workers must be 1 or more"):
        WriteOpts(max_workers=max_workers)


def test_writeopts_overrides():
    opts = WriteOpts(
        zstd_level=9,
        max_levels=3,
        min_bins=256,
        inner_len=4096,
        target_shard_bytes=2048,
        write_var=True,
    )
    assert opts.zstd_level == 9
    assert opts.max_levels == 3
    assert opts.min_bins == 256
    assert opts.inner_len == 4096
    assert opts.target_shard_bytes == 2048
    assert opts.write_var is True


def test_writeopts_frozen_rejects_mutation():
    opts = WriteOpts()
    with pytest.raises(dataclasses.FrozenInstanceError):
        opts.zstd_level = 1


@pytest.mark.parametrize("max_levels", [0, 1, 15])
def test_writeopts_accepts_max_levels_from_0_through_the_cap(max_levels):
    opts = WriteOpts(max_levels=max_levels)
    assert opts.max_levels == max_levels
    assert opts.level_cap == max_levels


@pytest.mark.parametrize("max_levels", [-1, 16])
def test_writeopts_rejects_max_levels_outside_its_range(max_levels):
    with pytest.raises(ValueError, match="0 through 15 at decimation_factor 4"):
        WriteOpts(max_levels=max_levels)


@pytest.mark.parametrize("decimation_factor", [2, 10])
def test_writeopts_accepts_a_decimation_factor_of_2_or_more(decimation_factor):
    opts = WriteOpts(decimation_factor=decimation_factor)
    assert opts.decimation_factor == decimation_factor


@pytest.mark.parametrize("decimation_factor", [1, 0, -4])
def test_writeopts_rejects_a_decimation_factor_below_2(decimation_factor):
    with pytest.raises(ValueError, match="decimation_factor must be 2 or more"):
        WriteOpts(decimation_factor=decimation_factor)


@pytest.mark.parametrize(
    ("decimation_factor", "cap"), [(2, 31), (3, 20), (4, 15), (10, 9)]
)
def test_writeopts_level_cap_follows_the_decimation_factor(
    decimation_factor, cap
):
    opts = WriteOpts(decimation_factor=decimation_factor)
    assert opts.max_levels is None
    assert opts.level_cap == cap


@pytest.mark.parametrize(("decimation_factor", "cap"), [(2, 31), (10, 9)])
def test_writeopts_checks_max_levels_against_the_cap_of_the_factor(
    decimation_factor, cap
):
    opts = WriteOpts(decimation_factor=decimation_factor, max_levels=cap)
    assert opts.level_cap == cap
    with pytest.raises(ValueError, match=f"0 through {cap} at decimation"):
        WriteOpts(decimation_factor=decimation_factor, max_levels=cap + 1)


def test_writeopts_rejects_write_var_with_no_levels():
    with pytest.raises(ValueError, match="write_var"):
        WriteOpts(max_levels=0, write_var=True)


@pytest.mark.parametrize(
    "field", ["min_bins", "inner_len", "target_shard_bytes"]
)
@pytest.mark.parametrize("value", [0, -1])
def test_writeopts_rejects_a_size_below_1(field, value):
    with pytest.raises(
        ValueError, match=f"^{field} must be 1 or more, got {value}$"
    ):
        WriteOpts(**{field: value})


@pytest.mark.parametrize(
    "field", ["min_bins", "inner_len", "target_shard_bytes"]
)
def test_writeopts_accepts_a_size_of_1(field):
    assert getattr(WriteOpts(**{field: 1}), field) == 1


def test_writeopts_continuous_factors_default_to_the_base_factor():
    assert WriteOpts().continuous_factors == (DEFAULT_DECIMATION_FACTOR,)
    assert WriteOpts(decimation_factor=3).continuous_factors == (3,)


def test_writeopts_continuous_factors_follow_the_override():
    assert WriteOpts(continuous_decimation_factor=8).continuous_factors == (8,)
    opts = WriteOpts(decimation_factor=3, continuous_decimation_factor=[8, 4])
    assert opts.continuous_factors == (8, 4)


def test_writeopts_event_factor_defaults_to_the_base_factor():
    assert WriteOpts().event_factor == DEFAULT_DECIMATION_FACTOR
    assert WriteOpts(decimation_factor=3).event_factor == 3


def test_writeopts_event_factor_follows_the_override():
    opts = WriteOpts(decimation_factor=3, event_decimation_factor=10)
    assert opts.event_factor == 10
    assert opts.continuous_factors == (3,)


def test_writeopts_overrides_leave_each_other_and_the_base_alone():
    opts = WriteOpts(
        decimation_factor=3,
        continuous_decimation_factor=(8, 4),
        event_decimation_factor=2,
    )
    assert opts.decimation_factor == 3
    assert opts.continuous_factors == (8, 4)
    assert opts.event_factor == 2


def test_writeopts_stores_a_ladder_as_a_tuple():
    opts = WriteOpts(continuous_decimation_factor=[8, 4])
    assert opts.continuous_decimation_factor == (8, 4)
    assert isinstance(opts.continuous_decimation_factor, tuple)


def test_writeopts_keeps_an_int_override_an_int_and_none_none():
    assert (
        WriteOpts(continuous_decimation_factor=8).continuous_decimation_factor
        == 8
    )
    assert WriteOpts().continuous_decimation_factor is None


def test_writeopts_with_a_ladder_survives_a_replace_round_trip():
    opts = WriteOpts(
        continuous_decimation_factor=[8, 4], event_decimation_factor=2
    )
    again = dataclasses.replace(opts, min_bins=8)
    assert again.continuous_decimation_factor == (8, 4)
    assert again.event_decimation_factor == 2
    assert dataclasses.replace(again, min_bins=1024) == opts


def test_writeopts_is_hashable_with_a_ladder():
    first = WriteOpts(continuous_decimation_factor=[8, 4])
    second = WriteOpts(continuous_decimation_factor=(8, 4))
    assert hash(first) == hash(second)
    assert {first, second} == {first}


@pytest.mark.parametrize(
    "value", [[8, 4], (8, 4), range(8, 3, -4), "48", 4.0, True, [4.0], [[4]]]
)
def test_writeopts_rejects_a_decimation_factor_that_is_not_an_int(value):
    with pytest.raises(TypeError, match="decimation_factor must be an int"):
        WriteOpts(decimation_factor=value)


@pytest.mark.parametrize("value", [[8, 4], (8, 4), "48", 4.0, True])
def test_writeopts_rejects_an_event_decimation_factor_that_is_not_an_int(value):
    with pytest.raises(
        TypeError, match="event_decimation_factor must be an int"
    ):
        WriteOpts(event_decimation_factor=value)


def test_writeopts_type_errors_point_to_the_continuous_ladder():
    with pytest.raises(TypeError, match="continuous_decimation_factor"):
        WriteOpts(decimation_factor=[8, 4])
    with pytest.raises(TypeError, match="continuous_decimation_factor"):
        WriteOpts(event_decimation_factor=(8, 4))


@pytest.mark.parametrize(
    "value",
    ["48", 4.0, True, [4.0], [True], [[8, 4]], [(8, 4)]],
)
def test_writeopts_rejects_a_continuous_ladder_of_the_wrong_type(value):
    with pytest.raises(TypeError, match="continuous_decimation_factor must"):
        WriteOpts(continuous_decimation_factor=value)


@pytest.mark.parametrize("value", [1, 0, -4])
def test_writeopts_rejects_an_event_decimation_factor_below_2(value):
    with pytest.raises(
        ValueError,
        match=f"event_decimation_factor must be 2 or more, got {value}",
    ):
        WriteOpts(event_decimation_factor=value)


@pytest.mark.parametrize("value", [1, 0, [8, 1], (1, 4), [4, 4, 0]])
def test_writeopts_rejects_a_continuous_ratio_below_2(value):
    with pytest.raises(
        ValueError, match="continuous_decimation_factor must .*2 or more"
    ):
        WriteOpts(continuous_decimation_factor=value)


def test_writeopts_names_the_ladder_it_got_in_the_value_error():
    with pytest.raises(ValueError, match=r"got \(8, 1\)"):
        WriteOpts(continuous_decimation_factor=[8, 1])


@pytest.mark.parametrize("value", [[], ()])
def test_writeopts_rejects_an_empty_continuous_ladder(value):
    with pytest.raises(
        ValueError, match="continuous_decimation_factor must hold at least one"
    ):
        WriteOpts(continuous_decimation_factor=value)


def test_writeopts_level_caps_follow_each_factor():
    opts = WriteOpts(continuous_decimation_factor=2, event_decimation_factor=10)
    assert opts.level_cap == 31
    assert opts.event_level_cap == 9
    ladder = WriteOpts(continuous_decimation_factor=(2**16, 2**15))
    assert ladder.level_cap == 2
    assert ladder.event_level_cap == 15


def test_writeopts_checks_max_levels_against_the_larger_cap():
    opts = WriteOpts(
        continuous_decimation_factor=2,
        event_decimation_factor=10,
        max_levels=31,
    )
    assert opts.level_cap == 31
    assert opts.event_level_cap == 9
    with pytest.raises(ValueError, match="max_levels must be 0 through 31"):
        WriteOpts(
            continuous_decimation_factor=2,
            event_decimation_factor=10,
            max_levels=32,
        )


def test_writeopts_clamps_max_levels_to_the_cap_of_each_kind():
    opts = WriteOpts(
        continuous_decimation_factor=(2**16, 2**15),
        event_decimation_factor=2,
        max_levels=20,
    )
    assert opts.level_cap == 2
    assert opts.event_level_cap == 20
    opts = WriteOpts(
        continuous_decimation_factor=2,
        event_decimation_factor=10,
        max_levels=20,
    )
    assert opts.level_cap == 20
    assert opts.event_level_cap == 9


def test_writeopts_max_levels_error_names_the_factors_of_an_override():
    with pytest.raises(ValueError, match=r"\(8, 4\).*event factor 4"):
        WriteOpts(continuous_decimation_factor=(8, 4), max_levels=16)


def test_writeopts_ladder_level_cap_counts_the_levels_that_fit():
    assert WriteOpts(continuous_decimation_factor=(8, 4)).level_cap == 15
    assert WriteOpts(continuous_decimation_factor=(3, 5)).level_cap == 14


def test_writeopts_write_var_requires_a_continuous_level():
    with pytest.raises(ValueError, match="write_var"):
        WriteOpts(write_var=True, max_levels=0)
    assert WriteOpts(write_var=True, max_levels=1).level_cap == 1
