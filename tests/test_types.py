import dataclasses
from typing import get_args

import pytest

from timeseries_zarr.constants import INNER_CHUNK_SAMPLES, TARGET_SHARD_BYTES
from timeseries_zarr.types import ChannelKind, ChunkShard, LevelPlan, WriteOpts


def test_construction_stores_fields():
    plan = LevelPlan(level=1, shape=(250, 2), period_us=125.0)
    assert plan.level == 1
    assert plan.shape == (250, 2)
    assert plan.period_us == 125.0


def test_frozen_rejects_mutation():
    plan = LevelPlan(level=1, shape=(250, 2), period_us=125.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.level = 5


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


def test_writeopts_defaults():
    opts = WriteOpts()
    assert opts.zstd_level == 5
    assert opts.max_levels == 7
    assert opts.min_bins == 1024
    assert opts.inner_len == INNER_CHUNK_SAMPLES
    assert opts.target_shard_bytes == TARGET_SHARD_BYTES


def test_writeopts_overrides():
    opts = WriteOpts(
        zstd_level=9,
        max_levels=3,
        min_bins=256,
        inner_len=4096,
        target_shard_bytes=2048,
    )
    assert opts.zstd_level == 9
    assert opts.max_levels == 3
    assert opts.min_bins == 256
    assert opts.inner_len == 4096
    assert opts.target_shard_bytes == 2048


def test_writeopts_frozen_rejects_mutation():
    opts = WriteOpts()
    with pytest.raises(dataclasses.FrozenInstanceError):
        opts.zstd_level = 1
