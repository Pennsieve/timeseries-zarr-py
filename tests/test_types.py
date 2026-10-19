import dataclasses
from datetime import UTC, datetime
from typing import get_args

import pytest

from timeseries_zarr.constants import (
    INNER_CHUNK_SAMPLES,
    MAX_LEVELS,
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
    assert opts.max_levels == MAX_LEVELS
    assert opts.min_bins == 1024
    assert opts.inner_len == INNER_CHUNK_SAMPLES
    assert opts.target_shard_bytes == TARGET_SHARD_BYTES
    assert opts.write_var is False
    assert opts.voltage_policy == "convert"


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


@pytest.mark.parametrize("max_levels", [0, 1, MAX_LEVELS])
def test_writeopts_accepts_max_levels_from_0_through_max_levels(max_levels):
    assert WriteOpts(max_levels=max_levels).max_levels == max_levels


@pytest.mark.parametrize("max_levels", [-1, MAX_LEVELS + 1])
def test_writeopts_rejects_max_levels_outside_its_range(max_levels):
    with pytest.raises(ValueError, match=f"0 through {MAX_LEVELS}"):
        WriteOpts(max_levels=max_levels)


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
