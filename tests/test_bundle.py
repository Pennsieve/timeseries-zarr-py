from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from timeseries_zarr.attrs import root_attrs
from timeseries_zarr.bundle import (
    assign_indices,
    channel_keys,
    clear_output_path,
    indexed_channels,
    max_key,
    reference_us,
    write_all_channels,
    write_bundle,
)
from timeseries_zarr.recording import RECORDING_FILE, epoch_us
from timeseries_zarr.types import RecordingMeta, WriteOpts
from timeseries_zarr.zarr_io import open_group

if TYPE_CHECKING:
    from tests.conftest import (
        ArrayContinuousSource,
        ArrayEventSource,
        ArrayMultiChannelSource,
    )


def _channels_of(indexed):
    return [channel for _, channel in indexed_channels(indexed)]


def _write_all(root, indexed, opts=None):
    reference = reference_us(_channels_of(indexed))
    write_all_channels(
        root,
        indexed,
        0 if reference is None else reference,
        channel_keys(indexed),
        opts or _OPTS,
    )


_OPTS = WriteOpts(
    min_bins=2, max_levels=8, inner_len=16, target_shard_bytes=256
)


def test_assign_indices_empty():
    assert assign_indices([], []) == []


def test_assign_indices_continuous_then_event(continuous_source):
    c0 = continuous_source(np.zeros(4, dtype=np.float32), id="c0")
    c1 = continuous_source(np.zeros(4, dtype=np.float32), id="c1")
    # An event source is never inspected.
    u0, u1 = object(), object()
    assert assign_indices([c0, c1], [u0, u1]) == [
        (0, c0),
        (1, c1),
        (2, u0),
        (3, u1),
    ]


def test_assign_indices_only_continuous(continuous_source):
    c0 = continuous_source(np.zeros(1, dtype=np.float32))
    assert assign_indices([c0], []) == [(0, c0)]


def test_assign_indices_only_events():
    u0 = object()
    assert assign_indices([], [u0]) == [(0, u0)]


def _zeros(
    continuous_source: type[ArrayContinuousSource],
    *ids: str,
    samples: int = 4,
) -> list[ArrayContinuousSource]:
    return [
        continuous_source(np.zeros(samples, np.float32), id=channel_id)
        for channel_id in ids
    ]


def test_assign_indices_gives_a_multi_channel_source_one_key_per_channel(
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
    event_source: type[ArrayEventSource],
) -> None:
    before, after = _zeros(continuous_source, "before", "after")
    many = multi_channel_source(_zeros(continuous_source, "m0", "m1", "m2"))
    marks = event_source([1], id="marks")
    indexed = assign_indices([before, many, after], [marks])
    assert [key for key, _ in indexed] == [0, 1, 4, 5]
    assert [source for _, source in indexed][::3] == [before, marks]


def test_assign_indices_gives_a_multi_channel_source_without_channels_no_key(
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    nothing = multi_channel_source([])
    (after,) = _zeros(continuous_source, "after")
    indexed = assign_indices([nothing, after], [])
    assert [key for key, _ in indexed] == [0, 0]
    assert indexed[1][1] == after


def test_indexed_channels_lists_each_channel_at_its_key(
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
    event_source: type[ArrayEventSource],
) -> None:
    before, after = _zeros(continuous_source, "before", "after")
    members = _zeros(continuous_source, "m0", "m1", "m2")
    marks = event_source([1], id="marks")
    indexed = assign_indices(
        [before, multi_channel_source(members), after], [marks]
    )
    assert indexed_channels(indexed) == [
        (0, before),
        (1, members[0]),
        (2, members[1]),
        (3, members[2]),
        (4, after),
        (5, marks),
    ]


def test_indexed_channels_leaves_a_flat_list_unchanged(
    continuous_source: type[ArrayContinuousSource],
    event_source: type[ArrayEventSource],
) -> None:
    (single,) = _zeros(continuous_source, "single")
    flat = [(0, single), (3, event_source([1], id="marks")), (9, single)]
    assert indexed_channels(flat) == flat


def test_write_all_channels_dispatches_by_type(
    tmp_path, continuous_source, event_source
):
    root = open_group(tmp_path / "bundle")
    cont = continuous_source(np.arange(64, dtype=np.float32), id="c0")
    event = event_source([5, 6, 7], id="e0")
    _write_all(root, assign_indices([cont], [event]))
    g = open_group(tmp_path / "bundle")
    assert dict(g["0"].attrs)["tsz"]["kind"] == "continuous"
    assert "raw" in list(g["0"].array_keys())
    assert dict(g["1"].attrs)["tsz"]["kind"] == "event"
    assert list(g["1"].array_keys()) == ["events"]


def test_write_all_channels_empty(tmp_path):
    root = open_group(tmp_path / "bundle")
    write_all_channels(root, [], 0, {}, _OPTS)
    assert list(open_group(tmp_path / "bundle").group_keys()) == []


def test_write_all_channels_returns_none(tmp_path, continuous_source):
    root = open_group(tmp_path / "bundle")
    indexed = assign_indices(
        [continuous_source(np.arange(8, dtype=np.float32))], []
    )
    result = write_all_channels(root, indexed, 0, channel_keys(indexed), _OPTS)
    assert result is None


def test_write_all_channels_resolves_event_channel_ids_through_the_keys(
    tmp_path, continuous_source, event_source
):
    root = open_group(tmp_path / "bundle")
    first = continuous_source(np.zeros(8, np.float32), id="c0")
    second = continuous_source(np.zeros(8, np.float32), id="c1")
    event = event_source([5, 6], id="e0", channel_ids=[["c1"], ["c0", "c1"]])
    _write_all(root, assign_indices([first, second], [event]))
    channels = open_group(tmp_path / "bundle")["2"]["channels"][:]
    assert channels.tolist() == [[2], [3]]


def test_write_all_channels_writes_a_multi_channel_source_at_consecutive_keys(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    solo = np.arange(70, dtype=np.float32)
    first = np.arange(150, dtype=np.float32) + 1000
    second = np.arange(90, dtype=np.float32) + 2000
    many = multi_channel_source(
        [
            continuous_source(first, id="m0", start_us=100),
            continuous_source(second, id="m1", start_us=300),
        ],
        window_rows=[7, 13],
    )
    root = open_group(tmp_path / "bundle")
    _write_all(
        root,
        assign_indices(
            [continuous_source(solo, id="solo", start_us=500), many], []
        ),
    )
    group = open_group(tmp_path / "bundle")
    assert sorted(group.group_keys()) == ["0", "1", "2"]
    attrs = [dict(group[key].attrs)["tsz"] for key in "012"]
    assert [item["id"] for item in attrs] == ["solo", "m0", "m1"]
    assert [item["offset_us"] for item in attrs] == [400, 0, 200]
    for key, samples in zip("012", (solo, first, second), strict=True):
        assert np.array_equal(group[key]["raw"][:], samples)


def test_write_all_channels_resolves_event_channel_ids_of_multi_channel_source_channels(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
    event_source: type[ArrayEventSource],
) -> None:
    (solo,) = _zeros(continuous_source, "solo", samples=8)
    many = multi_channel_source(
        _zeros(continuous_source, "m0", "m1", samples=8)
    )
    marks = event_source(
        [5, 6], id="marks", channel_ids=[["m1"], ["solo", "m0"]]
    )
    root = open_group(tmp_path / "bundle")
    _write_all(root, assign_indices([solo, many], [marks]))
    channels = open_group(tmp_path / "bundle")["3"]["channels"][:]
    assert channels.tolist() == [[4], [3]]


def test_write_bundle_writes_a_mixed_bundle(
    tmp_path, continuous_source, event_source
):
    output = tmp_path / "bundle"
    samples = np.arange(64, dtype=np.float32)
    cont = continuous_source(samples, id="c0")
    event = event_source([5, 6, 7], id="e0")
    write_bundle([cont], [event], output_path=output, opts=_OPTS)
    assert output.exists()
    g = open_group(output)
    assert dict(g["0"].attrs)["tsz"]["kind"] == "continuous"
    assert dict(g["1"].attrs)["tsz"]["kind"] == "event"
    assert np.array_equal(g["0"]["raw"][:], samples)
    assert g["1"]["events"].shape == (3,)


def test_write_bundle_counts_every_time_from_the_first_sample(
    tmp_path, continuous_source, event_source
):
    cont = continuous_source(np.zeros(8, np.float32), id="c0", start_us=1_000)
    event = event_source([500, 700], id="e0")
    output = tmp_path / "bundle"
    write_bundle([cont], [event], output_path=output, opts=_OPTS)
    g = open_group(output)
    assert dict(g["0"].attrs)["tsz"]["offset_us"] == 0
    assert g["1"]["events"][:].tolist() == [-500, -300]


def test_write_bundle_rejects_two_channels_with_one_id_before_writing(
    tmp_path, continuous_source, event_source
):
    output = tmp_path / "bundle"
    cont = continuous_source(np.zeros(8, np.float32), id="same")
    event = event_source([1], id="same")
    with pytest.raises(ValueError, match="share the id"):
        write_bundle([cont], [event], output_path=output, opts=_OPTS)
    assert not output.exists()


def test_write_bundle_consolidates_root_inline(tmp_path, continuous_source):
    output = tmp_path / "bundle"
    cont = continuous_source(np.arange(8, dtype=np.float32))
    write_bundle([cont], [], output_path=output, opts=_OPTS)
    root_json = json.loads((output / "zarr.json").read_text())
    assert "consolidated_metadata" in root_json
    assert not (output / ".zmetadata").exists()


def test_write_bundle_empty_inputs(tmp_path):
    output = tmp_path / "bundle"
    write_bundle([], [], output_path=output, opts=_OPTS)
    assert output.exists()
    assert list(open_group(output).group_keys()) == []


def test_write_bundle_returns_none(tmp_path, continuous_source):
    cont = continuous_source(np.arange(8, dtype=np.float32))
    result = write_bundle(
        [cont], [], output_path=tmp_path / "bundle", opts=_OPTS
    )
    assert result is None


def test_write_bundle_creates_missing_parent_directories(
    tmp_path, continuous_source
):
    output = tmp_path / "a" / "b" / "bundle"
    cont = continuous_source(np.arange(8, dtype=np.float32))
    write_bundle([cont], [], output_path=output, opts=_OPTS)
    assert open_group(output)["0"]["raw"].shape == (8,)


def test_write_bundle_writes_into_an_empty_directory_and_keeps_it(
    tmp_path, continuous_source
):
    output = tmp_path / "bundle"
    output.mkdir()
    inode = output.stat().st_ino
    cont = continuous_source(np.arange(8, dtype=np.float32))
    write_bundle([cont], [], output_path=output, opts=_OPTS)
    assert output.stat().st_ino == inode
    assert open_group(output)["0"]["raw"].shape == (8,)


def test_write_bundle_replaces_a_bundle_and_keeps_its_directory(
    tmp_path, continuous_source
):
    output = tmp_path / "bundle"
    first = [
        continuous_source(np.arange(64, dtype=np.float32), id="c0"),
        continuous_source(np.arange(64, dtype=np.float32), id="c1"),
    ]
    write_bundle(first, [], output_path=output, opts=_OPTS)
    inode = output.stat().st_ino
    second = continuous_source(np.ones(64, dtype=np.float32), id="c0")
    write_bundle([second], [], output_path=output, opts=_OPTS)
    replaced = open_group(output)
    assert output.stat().st_ino == inode
    assert replaced["0"]["raw"][:].tolist() == [1.0] * 64
    assert "1" not in replaced
    assert not (output / "1").exists()


def test_write_bundle_replaces_a_partial_bundle_an_interrupted_run_left(
    tmp_path, continuous_source
):
    output = tmp_path / "bundle"
    open_group(output, root_attrs()).create_group("7")
    cont = continuous_source(np.arange(8, dtype=np.float32))
    write_bundle([cont], [], output_path=output, opts=_OPTS)
    root_json = json.loads((output / "zarr.json").read_text())
    assert not (output / "7").exists()
    assert "consolidated_metadata" in root_json


def test_write_bundle_rejects_a_directory_holding_other_files(
    tmp_path, continuous_source
):
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "notes.txt").write_text("keep")
    cont = continuous_source(np.arange(8, dtype=np.float32))
    with pytest.raises(ValueError, match="holds files but no bundle"):
        write_bundle([cont], [], output_path=output, opts=_OPTS)
    assert (output / "notes.txt").read_text() == "keep"
    assert not (output / "zarr.json").exists()


@pytest.mark.parametrize(
    "root_json",
    [
        "{not json",
        "[]",
        '{"attributes": []}',
        '{"attributes": {}}',
        '{"attributes": {"tsz": {}}}',
    ],
)
def test_write_bundle_rejects_a_root_zarr_json_without_a_format_version(
    tmp_path, continuous_source, root_json
):
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "zarr.json").write_text(root_json)
    cont = continuous_source(np.arange(8, dtype=np.float32))
    with pytest.raises(ValueError, match="holds files but no bundle"):
        write_bundle([cont], [], output_path=output, opts=_OPTS)
    assert (output / "zarr.json").read_text() == root_json


def test_write_bundle_rejects_a_file_at_the_output_path(
    tmp_path, continuous_source
):
    output = tmp_path / "bundle"
    output.write_text("keep")
    cont = continuous_source(np.arange(8, dtype=np.float32))
    with pytest.raises(ValueError, match="is not a directory"):
        write_bundle([cont], [], output_path=output, opts=_OPTS)
    assert output.read_text() == "keep"


def test_write_bundle_leaves_an_existing_bundle_when_validation_fails(
    tmp_path, continuous_source
):
    output = tmp_path / "bundle"
    first = continuous_source(np.arange(8, dtype=np.float32), id="c0")
    write_bundle([first], [], output_path=output, opts=_OPTS)
    before = (output / "zarr.json").read_text()
    twins = [
        continuous_source(np.zeros(8, np.float32), id="same"),
        continuous_source(np.zeros(8, np.float32), id="same"),
    ]
    with pytest.raises(ValueError, match="share the id"):
        write_bundle(twins, [], output_path=output, opts=_OPTS)
    assert (output / "zarr.json").read_text() == before
    assert open_group(output)["0"]["raw"][:].tolist() == list(range(8))


def test_clear_output_path_removes_the_root_metadata_last(
    tmp_path, continuous_source, monkeypatch
):
    output = tmp_path / "bundle"
    cont = continuous_source(np.arange(8, dtype=np.float32))
    write_bundle([cont], [], output_path=output, opts=_OPTS)

    def fail(path):
        raise OSError(f"cannot remove {path}")

    monkeypatch.setattr("timeseries_zarr.bundle.shutil.rmtree", fail)
    with pytest.raises(OSError, match="cannot remove"):
        clear_output_path(output)
    assert (output / "zarr.json").is_file()


def test_clear_output_path_unlinks_a_symlink_and_keeps_its_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "data.txt").write_text("keep")
    output = tmp_path / "bundle"
    open_group(output, root_attrs())
    (output / "link").symlink_to(target)
    clear_output_path(output)
    assert list(output.iterdir()) == []
    assert (target / "data.txt").read_text() == "keep"


def test_write_all_channels_places_continuous_channels_relative_to_the_earliest_start(
    tmp_path, continuous_source
):
    root = open_group(tmp_path / "bundle")
    early = continuous_source(
        np.arange(8, dtype=np.float32), id="a", start_us=1_000
    )
    late = continuous_source(
        np.arange(8, dtype=np.float32), id="b", start_us=5_000
    )
    _write_all(root, assign_indices([early, late], []))
    g = open_group(tmp_path / "bundle")
    assert dict(g["0"].attrs)["tsz"]["offset_us"] == 0
    assert dict(g["1"].attrs)["tsz"]["offset_us"] == 4_000


def _samples(continuous_source, start_us):
    return continuous_source(np.zeros(4, np.float32), start_us=start_us)


def test_reference_us_is_the_first_sample_without_a_session_start(
    continuous_source, event_source
):
    sources = [
        _samples(continuous_source, 5_000),
        event_source([2_000, 9_000]),
        _samples(continuous_source, 7_000),
    ]
    assert reference_us(sources) == 5_000


def test_reference_us_is_the_session_start_when_it_precedes_every_sample(
    continuous_source,
):
    sources = [_samples(continuous_source, 5_000)]
    assert reference_us(sources, session_start_us=1_000) == 1_000


def test_reference_us_is_the_first_sample_when_the_session_starts_later(
    continuous_source,
):
    sources = [_samples(continuous_source, 5_000)]
    assert reference_us(sources, session_start_us=9_000) == 5_000


def test_reference_us_ignores_an_event_before_the_first_sample(
    continuous_source, event_source
):
    sources = [_samples(continuous_source, 5_000), event_source([40, 50])]
    assert reference_us(sources, session_start_us=5_000) == 5_000


def test_reference_us_ignores_a_continuous_channel_without_samples(
    continuous_source,
):
    empty = continuous_source(np.zeros(0, np.float32), start_us=1_000)
    sources = [empty, _samples(continuous_source, 5_000)]
    assert reference_us(sources) == 5_000
    assert reference_us(sources, session_start_us=9_000) == 5_000


def test_reference_us_falls_back_to_events_when_every_channel_is_empty(
    continuous_source, event_source
):
    empty = continuous_source(np.zeros(0, np.float32), start_us=1_000)
    assert reference_us([empty, event_source([40, 50])]) == 40
    assert reference_us([empty, event_source([40])], session_start_us=90) == 90
    assert reference_us([empty]) is None


def test_reference_us_is_the_first_sample_of_a_multi_channel_source_channel(
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    many = multi_channel_source(
        [
            continuous_source(
                np.zeros(4, np.float32), id="late", start_us=7_000
            ),
            continuous_source(
                np.zeros(4, np.float32), id="first", start_us=3_000
            ),
            continuous_source(
                np.zeros(0, np.float32), id="empty", start_us=1_000
            ),
        ]
    )
    channels = _channels_of(assign_indices([many], []))
    assert reference_us(channels) == 3_000
    assert reference_us(channels, session_start_us=2_000) == 2_000


def test_reference_us_is_the_session_start_without_samples(event_source):
    assert reference_us([event_source([40, 50])], session_start_us=90) == 90


def test_reference_us_is_the_first_event_without_samples_or_a_session_start(
    event_source,
):
    sources = [event_source([], start_us=7), event_source([40, 50])]
    assert reference_us(sources) == 40


def test_reference_us_of_nothing_is_none(event_source):
    assert reference_us([]) is None
    assert reference_us([event_source([], start_us=77)]) is None


def test_reference_us_is_the_declared_value_over_everything(
    continuous_source, event_source
):
    sources = [_samples(continuous_source, 5_000), event_source([2_000])]
    assert reference_us(sources, 1_000, declared_us=8_000) == 8_000
    assert reference_us([], declared_us=8_000) == 8_000


def test_reference_us_may_be_negative_for_a_start_before_the_epoch(
    continuous_source,
):
    assert reference_us([_samples(continuous_source, -5)]) == -5


def test_channel_keys_maps_every_id_to_its_index(
    continuous_source, event_source
):
    indexed = assign_indices(
        [
            continuous_source(np.zeros(4, np.float32), id="LA1"),
            continuous_source(np.zeros(4, np.float32), id="LA2"),
        ],
        [event_source([1], id="units"), event_source([1], id="marks")],
    )
    assert channel_keys(indexed) == {"LA1": 0, "LA2": 1, "units": 2, "marks": 3}


def test_channel_keys_maps_each_channel_of_a_multi_channel_source(
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
    event_source: type[ArrayEventSource],
) -> None:
    (solo,) = _zeros(continuous_source, "solo")
    many = multi_channel_source(_zeros(continuous_source, "m0", "m1"))
    indexed = assign_indices([solo, many], [event_source([1], id="marks")])
    assert channel_keys(indexed) == {"solo": 0, "m0": 1, "m1": 2, "marks": 3}


def test_channel_keys_of_no_channels_is_empty():
    assert channel_keys([]) == {}


def test_channel_keys_rejects_two_channels_with_one_id(continuous_source):
    indexed = assign_indices(
        [
            continuous_source(np.zeros(4, np.float32), id="same"),
            continuous_source(np.zeros(4, np.float32), id="other"),
            continuous_source(np.zeros(4, np.float32), id="same"),
        ],
        [],
    )
    with pytest.raises(
        ValueError, match="channels 0 and 2 share the id 'same'"
    ):
        channel_keys(indexed)


def test_channel_keys_rejects_a_continuous_and_an_event_channel_sharing_an_id(
    continuous_source, event_source
):
    indexed = [
        (0, continuous_source(np.zeros(4, np.float32), id="x")),
        (1, event_source([1], id="x")),
    ]
    with pytest.raises(ValueError, match="share the id 'x'"):
        channel_keys(indexed)


def test_channel_keys_rejects_two_channels_of_one_multi_channel_source_sharing_an_id(
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    many = multi_channel_source(_zeros(continuous_source, "x", "y", "x"))
    with pytest.raises(ValueError, match="channels 0 and 2 share the id 'x'"):
        channel_keys(assign_indices([many], []))


def test_channel_keys_rejects_a_multi_channel_source_channel_sharing_an_id_with_a_single_source(
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    many = multi_channel_source(_zeros(continuous_source, "a", "b"))
    (twin,) = _zeros(continuous_source, "b")
    with pytest.raises(ValueError, match="channels 1 and 2 share the id 'b'"):
        channel_keys(assign_indices([many, twin], []))


def test_max_key_is_the_largest_index(continuous_source):
    sources = [continuous_source(np.zeros(4, np.float32)) for _ in range(5)]
    assert max_key(assign_indices(sources, [])) == 4


def test_max_key_does_not_assume_contiguous_keys(continuous_source):
    src = continuous_source(np.zeros(4, np.float32))
    assert max_key([(2, src), (9, src), (5, src)]) == 9


def test_max_key_counts_every_channel_of_a_multi_channel_source(
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    many = multi_channel_source(_zeros(continuous_source, "a", "b", "c", "d"))
    (solo,) = _zeros(continuous_source, "solo")
    assert max_key(assign_indices([many], [])) == 3
    assert max_key(assign_indices([solo, many], [])) == 4


def test_max_key_of_no_channels_is_zero():
    assert max_key([]) == 0


_SPEC_OPTS = WriteOpts(
    min_bins=256, max_levels=8, inner_len=256, target_shard_bytes=4096
)
_SESSION = dt.datetime(
    2024, 3, 14, 9, 15, tzinfo=dt.timezone(dt.timedelta(hours=-5))
)
_SESSION_US = epoch_us(_SESSION)
_RECORDING = RecordingMeta(
    session_start_time=_SESSION,
    subject={"id": "sub-0931", "species": "Homo sapiens"},
    session={"id": "ses-01"},
    source={"converter": "timeseries-zarr-py"},
)


_ACQUIRE = {
    "software": "acquire 4.1",
    "settings": {"gain": 2, "bands_hz": [1, 70], "notch": True},
}
_SORTER = {
    "software": "sorter 2.1",
    "parameters": {"threshold": 4.5, "detect_sign": -1},
    "notes": None,
}
_REVIEW = {"software": "review-tool 3.2"}
_MIXED_PROVENANCE = {
    "0": _ACQUIRE,
    "1": None,
    "2": _SORTER,
    "3": _REVIEW,
    "4": None,
}


def _signal(n, seed):
    rng = np.random.default_rng(seed)
    samples = (5000.0 + 8.0 * rng.standard_normal(n)).astype(np.float32)
    samples[300:340] = np.nan
    return samples


def _mixed_sources(
    continuous_source,
    event_source,
    base=_SESSION_US,
    with_provenance=True,
):
    def given(provenance):
        return provenance if with_provenance else None

    rng = np.random.default_rng(3)
    lead = continuous_source(
        _signal(8192, 0),
        id="LA1",
        name="LA1",
        rate_hz=256.0,
        start_us=base,
        provenance=given(_ACQUIRE),
    )
    lag = continuous_source(
        _signal(16384, 1),
        id="LA2",
        name="LA2",
        rate_hz=512.0,
        start_us=base + 500_000,
    )
    spike_times = np.sort(rng.integers(0, 30_000_000, 400)) + base
    spikes = event_source(
        spike_times,
        labels=rng.integers(0, 3, 400),
        label_names=["unit-0", "unit-1", "unit-2"],
        waveform_mean=rng.standard_normal((3, 16)),
        waveform_period_us=122.0703125,
        id="units",
        name="units",
        provenance=given(_SORTER),
    )
    marks = event_source(
        base + np.array([1, 4, 9, 15, 22, 28]) * 1_000_000,
        durations=[0, 0, 5_000_000, 0, 12_000_000, 0],
        labels=[1, 1, 0, 2, 0, 2],
        label_names=["seizure", "spike", "note"],
        text=["spike", "spike", "seizure onset", "note", "seizure", "x"],
        channel_ids=[["LA1"], [], ["LA1", "LA2"], [], ["LA2"], []],
        id="marks",
        name="curated marks",
        provenance=given(_REVIEW),
    )
    dense_times = np.sort(rng.integers(0, 700_000_000, 70_000))
    dense_times[-1] = 700_000_000
    dense = event_source(
        dense_times + base,
        labels=rng.integers(0, 2, 70_000),
        label_names=["ripple", "fast-ripple"],
        id="hfo",
        name="hfo detector",
    )
    return [lead, lag], [spikes, marks, dense]


def _read_nodes(path):
    return {
        "/".join(file.parent.relative_to(path).parts): json.loads(
            file.read_text()
        )
        for file in path.rglob("zarr.json")
    }


def _array_ok(node, shape, dtype, names):
    fill = "NaN" if dtype.startswith("float") else 0
    return (
        node["node_type"] == "array"
        and tuple(node["shape"]) == tuple(shape)
        and node["data_type"] == dtype
        and node["fill_value"] == fill
        and node.get("dimension_names") == names
    )


def _children(nodes, key):
    prefix = key + "/"
    return {
        p[len(prefix) :]
        for p in nodes
        if p.startswith(prefix) and "/" not in p[len(prefix) :]
    }


def _check_root(path, nodes):
    root = nodes[""]
    assert root["attributes"] == {"tsz": {"version": "1.0"}}
    assert not any(
        "version" in n["attributes"].get("tsz", {})
        for p, n in nodes.items()
        if p
    )
    consolidated = root["consolidated_metadata"]
    assert consolidated["kind"] == "inline"
    assert consolidated["must_understand"] is False
    assert set(consolidated["metadata"]) == set(nodes) - {""}
    for p, entry in consolidated["metadata"].items():
        stored = {
            k: v for k, v in nodes[p].items() if k != "consolidated_metadata"
        }
        listed = {
            k: v for k, v in entry.items() if k != "consolidated_metadata"
        }
        assert listed == stored, p
    assert all(
        isinstance(n["attributes"].get("tsz"), dict)
        for n in nodes.values()
        if n.get("attributes")
    )
    texts = [(path / p / "zarr.json").read_text() for p in nodes]
    assert not any(re.search(r"\d{4}-\d{2}-\d{2}", text) for text in texts)
    assert not any("sub-0931" in text for text in texts)


def _check_continuous(nodes, key, attrs):
    rate = attrs["rate_hz"]
    assert rate > 0
    assert isinstance(attrs["offset_us"], int)
    samples = nodes[f"{key}/raw"]["shape"][0]
    assert _array_ok(nodes[f"{key}/raw"], (samples,), "float32", ["time"])
    levels = sorted(int(c) for c in _children(nodes, key) if c.isdigit())
    assert levels == list(range(1, len(levels) + 1))
    for k in levels:
        level_attrs = nodes[f"{key}/{k}"]["attributes"]["tsz"]
        period = level_attrs["period_us"]
        assert "offset_us" not in level_attrs
        assert math.isclose(period, 4**k * 1e6 / rate, rel_tol=1e-12)
        members = _children(nodes, f"{key}/{k}")
        assert (
            {"minmax", "mean", "count"}
            <= members
            <= {
                "minmax",
                "mean",
                "var",
                "count",
            }
        )
        bins = math.ceil(samples / 4**k)
        expected = {
            "minmax": ((bins, 2), "float32", ["time", None]),
            "mean": ((bins,), "float32", ["time"]),
            "var": ((bins,), "float32", ["time"]),
            "count": ((bins,), "uint32", ["time"]),
        }
        for member in members:
            assert _array_ok(nodes[f"{key}/{k}/{member}"], *expected[member])


def _check_count_levels(nodes, key, events, labels, categories):
    levels = sorted(int(c) for c in _children(nodes, key) if c.isdigit())
    assert levels == list(range(1, len(levels) + 1))
    data = open_group(PATH_HOLDER["path"])[key]
    previous = None
    for k in levels:
        level_attrs = nodes[f"{key}/{k}"]["attributes"]["tsz"]
        period = level_attrs["period_us"]
        offset = level_attrs["offset_us"]
        assert isinstance(offset, int) and offset % period == 0
        assert _children(nodes, f"{key}/{k}") == {"count"}
        node = nodes[f"{key}/{k}/count"]
        bins = node["shape"][0]
        assert _array_ok(node, (bins, categories), "uint32", ["time", None])
        counts = data[str(k)]["count"][:]
        assert offset <= events[0] and offset + bins * period > events[-1]
        assert (
            counts.sum(axis=0).tolist()
            == np.bincount(labels, minlength=categories).tolist()
        )
        if previous is None:
            rung = int(period)
            while rung % 4 == 0 and rung > 1000:
                rung //= 4
            assert float(period).is_integer() and rung == 1000
        else:
            assert period == 4 * previous
        previous = period


PATH_HOLDER = {}


def _check_text_and_channels(nodes, key, data, n, group_keys):
    text = nodes.get(f"{key}/text")
    if text is not None:
        width = text["shape"][1]
        assert _array_ok(text, (n, width), "uint8", ["event", None])
        rows = [bytes(row).rstrip(b"\0") for row in data["text"][:]]
        assert all(b"\0" not in row for row in rows)
        assert max(len(row.decode("utf-8")) for row in rows) <= width
        assert max(len(row) for row in rows) == width
    channels = nodes.get(f"{key}/channels")
    if channels is not None:
        width = (max(group_keys) + 1 + 7) // 8
        assert _array_ok(channels, (n, width), "uint8", ["event", None])
        bits = np.unpackbits(data["channels"][:], axis=1, bitorder="little")
        assert set(np.flatnonzero(bits.any(axis=0)).tolist()) <= set(group_keys)


def _check_event(path, nodes, key, attrs, group_keys):
    n = nodes[f"{key}/events"]["shape"][0]
    assert _array_ok(nodes[f"{key}/events"], (n,), "int64", ["event"])
    data = open_group(path)[key]
    events = data["events"][:]
    assert (np.diff(events) >= 0).all()
    arrays = {c for c in _children(nodes, key) if not c.isdigit()}
    assert arrays <= {
        "events",
        "durations",
        "labels",
        "text",
        "channels",
        "waveform_mean",
    }
    assert all(
        nodes[f"{key}/{a}"]["shape"][0] == n for a in arrays - {"waveform_mean"}
    )
    labels = data["labels"][:] if "labels" in arrays else None
    names = attrs.get("label_names")
    categories = (
        len(names)
        if names
        else (int(labels.max()) + 1 if labels is not None else 1)
    )
    if labels is not None:
        assert _array_ok(nodes[f"{key}/labels"], (n,), "uint16", ["event"])
        assert int(labels.max()) < categories
    if "durations" in arrays:
        durations = data["durations"][:]
        assert _array_ok(nodes[f"{key}/durations"], (n,), "int64", ["event"])
        assert (
            durations.min() >= 0 and durations.max() <= attrs["max_duration_us"]
        )
    if "waveform_mean" in arrays:
        node = nodes[f"{key}/waveform_mean"]
        assert _array_ok(node, tuple(node["shape"]), "float32", ["label", None])
        assert node["shape"][0] == categories and labels is not None
        assert node["attributes"]["tsz"]["period_us"] > 0
    _check_text_and_channels(nodes, key, data, n, group_keys)
    if any(c.isdigit() for c in _children(nodes, key)):
        PATH_HOLDER["path"] = path
        _check_count_levels(
            nodes,
            key,
            events,
            labels if labels is not None else np.zeros(n, int),
            categories,
        )


def _check_recording(path, nodes):
    """Return the reference time and the session start of the bundle."""
    document = json.loads((path / RECORDING_FILE).read_text("utf-8"))
    times = [
        dt.datetime.fromisoformat(document["reference_time"]),
        dt.datetime.fromisoformat(document["session"]["start_time"]),
    ]
    assert all(time.utcoffset() is not None for time in times)
    assert all(
        isinstance(document.get(k, {}), dict)
        for k in ("subject", "session", "source")
    )
    assert RECORDING_FILE not in nodes[""]["consolidated_metadata"]["metadata"]
    return times


def _check_provenance(nodes):
    holders = [
        path
        for path, node in nodes.items()
        if "provenance" in node.get("attributes", {}).get("tsz", {})
    ]
    assert all(path and "/" not in path for path in holders)
    for path in holders:
        assert isinstance(nodes[path]["attributes"]["tsz"]["provenance"], dict)


def _validate_bundle(path):
    """Check a bundle against the specification's validation list."""
    nodes = _read_nodes(path)
    _check_root(path, nodes)
    _check_provenance(nodes)
    group_keys = sorted(int(p) for p in nodes if p and "/" not in p)
    for key in map(str, group_keys):
        assert key == str(int(key))
        attrs = nodes[key]["attributes"]["tsz"]
        assert isinstance(attrs["name"], str)
        if attrs["kind"] == "continuous":
            _check_continuous(nodes, key, attrs)
        else:
            assert attrs["kind"] == "event"
            _check_event(path, nodes, key, attrs, group_keys)
    return nodes


def _write_mixed(
    tmp_path,
    continuous_source,
    event_source,
    recording=_RECORDING,
    base=_SESSION_US,
    opts=_SPEC_OPTS,
    with_provenance=True,
    name="bundle",
):
    continuous, events = _mixed_sources(
        continuous_source, event_source, base, with_provenance
    )
    output = tmp_path / name
    write_bundle(
        continuous,
        events,
        output_path=output,
        opts=opts,
        recording=recording,
    )
    return output


def test_write_bundle_conforms_to_the_specification_validation_list(
    tmp_path, continuous_source, event_source
):
    output = _write_mixed(tmp_path, continuous_source, event_source)
    nodes = _validate_bundle(output)
    assert sorted(p for p in nodes if p and "/" not in p) == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]
    assert _check_recording(output, nodes) == [_SESSION, _SESSION]


def test_write_bundle_without_a_recording_has_no_recording_json_and_conforms(
    tmp_path, continuous_source, event_source
):
    output = _write_mixed(
        tmp_path, continuous_source, event_source, recording=None
    )
    _validate_bundle(output)
    assert not (output / RECORDING_FILE).exists()


def test_write_bundle_recording_without_a_session_start_dates_the_first_sample(
    tmp_path, continuous_source, event_source
):
    output = _write_mixed(
        tmp_path,
        continuous_source,
        event_source,
        recording=RecordingMeta(subject={"id": "sub-0931"}),
    )
    nodes = _validate_bundle(output)
    document = json.loads((output / RECORDING_FILE).read_text("utf-8"))
    assert document == {
        "reference_time": "2024-03-14T14:15:00.000000+00:00",
        "subject": {"id": "sub-0931"},
    }
    assert nodes["0"]["attributes"]["tsz"]["offset_us"] == 0


def test_write_bundle_reference_time_is_the_first_sample_before_the_session_start(
    tmp_path, continuous_source, event_source
):
    output = _write_mixed(
        tmp_path, continuous_source, event_source, base=_SESSION_US - 2_000_000
    )
    nodes = _validate_bundle(output)
    reference, session_start = _check_recording(output, nodes)
    assert reference == _SESSION - dt.timedelta(seconds=2)
    assert session_start == _SESSION
    assert nodes["0"]["attributes"]["tsz"]["offset_us"] == 0


def test_write_bundle_reference_time_is_a_session_start_before_every_sample(
    tmp_path, continuous_source, event_source
):
    output = _write_mixed(
        tmp_path, continuous_source, event_source, base=_SESSION_US + 5_000_000
    )
    nodes = _validate_bundle(output)
    reference, session_start = _check_recording(output, nodes)
    assert reference == _SESSION
    assert session_start == _SESSION
    assert nodes["0"]["attributes"]["tsz"]["offset_us"] == 5_000_000


def test_write_bundle_places_an_epoch_before_the_session_at_a_negative_time(
    tmp_path, continuous_source, event_source
):
    eeg = continuous_source(
        np.zeros(64, np.float32), id="eeg", start_us=_SESSION_US
    )
    epochs = event_source([_SESSION_US - 32_000], id="hypnogram")
    output = tmp_path / "bundle"
    write_bundle(
        [eeg], [epochs], output_path=output, opts=_OPTS, recording=_RECORDING
    )
    nodes = _validate_bundle(output)
    assert _check_recording(output, nodes) == [_SESSION, _SESSION]
    assert nodes["0"]["attributes"]["tsz"]["offset_us"] == 0
    assert open_group(output)["1"]["events"][:].tolist() == [-32_000]


def test_write_bundle_places_the_reference_at_the_first_sample_past_an_empty_channel(
    tmp_path, continuous_source
):
    hour_us = 3_600_000_000
    empty = continuous_source(
        np.zeros(0, np.float32), id="empty", start_us=_SESSION_US - hour_us
    )
    eeg = continuous_source(
        np.zeros(64, np.float32), id="eeg", start_us=_SESSION_US
    )
    output = tmp_path / "bundle"
    write_bundle(
        [empty, eeg],
        [],
        output_path=output,
        opts=_OPTS,
        recording=RecordingMeta(subject={"id": "sub-0931"}),
    )
    document = json.loads((output / RECORDING_FILE).read_text("utf-8"))
    assert document["reference_time"] == "2024-03-14T14:15:00.000000+00:00"
    group = open_group(output)
    assert dict(group["0"].attrs)["tsz"]["offset_us"] == -hour_us
    assert dict(group["1"].attrs)["tsz"]["offset_us"] == 0


def test_write_bundle_annotations_alone_share_the_reference_of_their_recording(
    tmp_path, event_source
):
    epochs = event_source([_SESSION_US - 32_000, _SESSION_US], id="hypnogram")
    output = tmp_path / "bundle"
    write_bundle(
        [], [epochs], output_path=output, opts=_OPTS, recording=_RECORDING
    )
    nodes = _validate_bundle(output)
    assert _check_recording(output, nodes) == [_SESSION, _SESSION]
    assert open_group(output)["0"]["events"][:].tolist() == [-32_000, 0]


def test_write_bundle_keeps_a_declared_reference_time_and_offsets_from_it(
    tmp_path, continuous_source, event_source
):
    declared = _SESSION + dt.timedelta(seconds=10)
    recording = RecordingMeta(
        session_start_time=_SESSION, reference_time=declared, subject={}
    )
    output = _write_mixed(
        tmp_path, continuous_source, event_source, recording=recording
    )
    nodes = _validate_bundle(output)
    reference, session_start = _check_recording(output, nodes)
    assert reference == declared
    assert session_start == _SESSION
    assert nodes["0"]["attributes"]["tsz"]["offset_us"] == -10_000_000
    assert open_group(output)["3"]["events"][0] == 1_000_000 - 10_000_000


def test_write_bundle_rejects_a_declared_reference_time_without_an_offset(
    tmp_path, continuous_source
):
    naive = RecordingMeta(reference_time=dt.datetime(2024, 3, 14, 9, 15))
    cont = continuous_source(np.zeros(8, np.float32), id="c0")
    with pytest.raises(ValueError, match="reference_time must carry a UTC"):
        write_bundle(
            [cont],
            [],
            output_path=tmp_path / "bundle",
            opts=_OPTS,
            recording=naive,
        )
    assert not (tmp_path / "bundle").exists()


def test_write_bundle_without_samples_or_events_dates_the_session_start(
    tmp_path,
):
    output = tmp_path / "bundle"
    write_bundle([], [], output_path=output, opts=_OPTS, recording=_RECORDING)
    document = json.loads((output / RECORDING_FILE).read_text("utf-8"))
    session_start = _SESSION.isoformat(timespec="microseconds")
    assert document["reference_time"] == session_start
    assert document["session"]["start_time"] == session_start


def test_write_bundle_without_samples_events_or_a_session_start_has_no_reference(
    tmp_path,
):
    output = tmp_path / "bundle"
    subject_only = RecordingMeta(subject={"id": "sub-0931"})
    write_bundle([], [], output_path=output, opts=_OPTS, recording=subject_only)
    document = json.loads((output / RECORDING_FILE).read_text("utf-8"))
    assert document == {"subject": {"id": "sub-0931"}}


def test_write_bundle_without_wall_clock_times_writes_no_reference_time(
    tmp_path, continuous_source, event_source
):
    undated = RecordingMeta(
        session_start_time=_SESSION,
        subject={},
        session={},
        source={},
        wall_clock=False,
    )
    output = _write_mixed(
        tmp_path, continuous_source, event_source, recording=undated
    )
    nodes = _validate_bundle(output)
    document = json.loads((output / RECORDING_FILE).read_text("utf-8"))
    assert list(document) == ["session"]
    assert nodes["0"]["attributes"]["tsz"]["offset_us"] == 0


def test_write_bundle_without_wall_clock_times_ignores_the_session_start(
    tmp_path, event_source
):
    undated = RecordingMeta(session_start_time=_SESSION, wall_clock=False)
    output = tmp_path / "bundle"
    write_bundle(
        [],
        [event_source([40, 50])],
        output_path=output,
        opts=_OPTS,
        recording=undated,
    )
    document = json.loads((output / RECORDING_FILE).read_text("utf-8"))
    assert "reference_time" not in document
    assert open_group(output)["0"]["events"][:].tolist() == [0, 10]


def test_write_bundle_rejects_a_reference_time_without_wall_clock_times(
    tmp_path, continuous_source
):
    mixed_up = RecordingMeta(reference_time=_SESSION, wall_clock=False)
    cont = continuous_source(np.zeros(8, np.float32), id="c0")
    with pytest.raises(
        ValueError, match="reference_time needs wall_clock=True"
    ):
        write_bundle(
            [cont],
            [],
            output_path=tmp_path / "bundle",
            opts=_OPTS,
            recording=mixed_up,
        )
    assert not (tmp_path / "bundle").exists()


def test_write_bundle_keeps_identity_and_the_date_out_of_every_zarr_node(
    tmp_path, continuous_source, event_source
):
    output = _write_mixed(tmp_path, continuous_source, event_source)
    document = json.loads((output / RECORDING_FILE).read_text("utf-8"))
    assert document["subject"] == {"id": "sub-0931", "species": "Homo sapiens"}
    assert "sub-0931" not in (output / "zarr.json").read_text()
    assert (
        document["reference_time"][:10]
        not in (output / "zarr.json").read_text()
    )


def test_write_bundle_rejects_a_session_start_without_an_offset_before_writing(
    tmp_path, continuous_source
):
    naive = RecordingMeta(
        session_start_time=dt.datetime(2024, 3, 14, 9, 15),
        subject={},
        session={},
        source={},
    )
    cont = continuous_source(np.zeros(8, np.float32), id="c0")
    with pytest.raises(ValueError, match="UTC offset"):
        write_bundle(
            [cont],
            [],
            output_path=tmp_path / "bundle",
            opts=_OPTS,
            recording=naive,
        )
    assert not (tmp_path / "bundle").exists()


def test_write_bundle_with_a_channel_too_short_for_a_level_still_conforms(
    tmp_path, continuous_source
):
    short = continuous_source(_signal(100, 5), id="short", name="short")
    output = tmp_path / "bundle"
    write_bundle([short], [], output_path=output, opts=_SPEC_OPTS)
    nodes = _validate_bundle(output)
    assert sorted(p for p in nodes if p.startswith("0/")) == ["0/raw"]


@pytest.mark.parametrize("factor", [2, 10])
def test_write_bundle_writes_the_levels_of_the_decimation_factor(
    tmp_path, continuous_source, factor
):
    samples = _signal(6000, 5)
    cont = continuous_source(samples, id="c0", name="c0", rate_hz=512.0)
    output = tmp_path / "bundle"
    opts = dataclasses.replace(_SPEC_OPTS, decimation_factor=factor, min_bins=4)
    write_bundle([cont], [], output_path=output, opts=opts)
    group = open_group(output)["0"]
    levels = sorted(group.group_keys(), key=int)
    assert len(levels) >= 2
    for name in levels:
        width = factor ** int(name)
        level = group[name]
        assert level.attrs["tsz"]["period_us"] == width * 1e6 / 512.0
        assert level["count"].shape == (-(-6000 // width),)
        assert int(level["count"][:].sum()) == 6000 - 40


def test_write_bundle_with_max_levels_0_writes_no_level_and_conforms(
    tmp_path, continuous_source, event_source
):
    output = _write_mixed(
        tmp_path,
        continuous_source,
        event_source,
        opts=dataclasses.replace(_SPEC_OPTS, max_levels=0),
    )
    nodes = _validate_bundle(output)
    for key in ("0", "1", "2", "3", "4"):
        assert not any(child.isdigit() for child in _children(nodes, key))
    assert _children(nodes, "0") == {"raw"}
    assert _children(nodes, "4") == {"events", "labels"}
    consolidated = nodes[""]["consolidated_metadata"]["metadata"]
    assert not any(
        part.isdigit() for path in consolidated for part in path.split("/")[1:]
    )


def test_write_bundle_require_uv_rejects_a_volts_source_before_writing(
    tmp_path, continuous_source
):
    output = tmp_path / "bundle"
    cont = continuous_source(np.zeros(8, np.float32), id="ch-7", unit="volts")
    opts = WriteOpts(voltage_policy="require_uv")
    with pytest.raises(ValueError, match=r"channel 0 \('ch-7'\).*'volts'"):
        write_bundle([cont], [], output_path=output, opts=opts)
    assert not output.exists()


def test_write_bundle_convert_leaves_a_microvolt_source_bit_identical(
    tmp_path, continuous_source
):
    samples = np.linspace(-3000.0, 3000.0, 64, dtype=np.float32)
    source = continuous_source(samples, unit="uV")
    output = tmp_path / "bundle"
    write_bundle([source], [], output_path=output, opts=_OPTS)
    stored = open_group(output)["0"]["raw"][:]
    assert stored.tobytes() == samples.tobytes()


def _channel_attrs(path):
    """Return each channel's tsz object, read from its own zarr.json."""
    return {
        file.parent.name: json.loads(file.read_text())["attributes"]["tsz"]
        for file in sorted(path.glob("*/zarr.json"))
    }


def _consolidated_attrs(path):
    """Return each channel's tsz object, read from the root's listing."""
    root = json.loads((path / "zarr.json").read_text())
    listing = root["consolidated_metadata"]["metadata"]
    return {
        key: entry["attributes"]["tsz"]
        for key, entry in listing.items()
        if "/" not in key
    }


def _assert_provenance_matches(attrs, expected):
    assert set(attrs) == set(expected)
    for key, provenance in expected.items():
        if provenance is None:
            assert "provenance" not in attrs[key], key
        else:
            assert attrs[key]["provenance"] == provenance, key


def test_write_bundle_writes_each_source_provenance_on_its_channel_group(
    tmp_path, continuous_source, event_source
):
    output = _write_mixed(tmp_path, continuous_source, event_source)
    attrs = _channel_attrs(output)
    _assert_provenance_matches(attrs, _MIXED_PROVENANCE)
    kinds = {key: attrs[key]["kind"] for key in attrs}
    assert kinds == {
        "0": "continuous",
        "1": "continuous",
        "2": "event",
        "3": "event",
        "4": "event",
    }


def test_write_bundle_lists_provenance_in_the_consolidated_root_metadata(
    tmp_path, continuous_source, event_source
):
    output = _write_mixed(tmp_path, continuous_source, event_source)
    _assert_provenance_matches(_consolidated_attrs(output), _MIXED_PROVENANCE)
    root = json.loads((output / "zarr.json").read_text())
    assert root["attributes"] == {"tsz": {"version": "1.0"}}


def test_write_bundle_reads_provenance_back_through_the_zarr_library(
    tmp_path, continuous_source, event_source
):
    output = _write_mixed(tmp_path, continuous_source, event_source)
    group = open_group(output)
    attrs = {key: dict(group[key].attrs)["tsz"] for key in group.group_keys()}
    _assert_provenance_matches(attrs, _MIXED_PROVENANCE)


_JSON_VALUES = {
    "text": 'caf\N{LATIN SMALL LETTER E WITH ACUTE} "quoted" \\ path\nline',
    "integer": 7,
    "large_integer": 10**12,
    "negative": -3,
    "fraction": 0.1,
    "small": 1e-9,
    "large": 2.5e9,
    "flags": [True, False],
    "missing": None,
    "empty_list": [],
    "empty_object": {},
    "nested": [{"a": [1, 2, {"b": "c"}]}, [[]], "x"],
}


def test_write_bundle_round_trips_every_json_value_type_in_provenance(
    tmp_path, continuous_source, event_source
):
    output = tmp_path / "bundle"
    cont = continuous_source(
        np.arange(64, dtype=np.float32), id="c", provenance=_JSON_VALUES
    )
    event = event_source([1, 2, 3], id="e", provenance=_JSON_VALUES)
    write_bundle([cont], [event], output_path=output, opts=_OPTS)
    expected = {"0": _JSON_VALUES, "1": _JSON_VALUES}
    _assert_provenance_matches(_channel_attrs(output), expected)
    _assert_provenance_matches(_consolidated_attrs(output), expected)


def test_write_bundle_keeps_each_channels_provenance_separate(
    tmp_path, continuous_source, event_source
):
    shared = {"software": "shared 1.0"}
    continuous = [
        continuous_source(np.zeros(8, np.float32), id="c0", provenance=shared),
        continuous_source(np.zeros(8, np.float32), id="c1", provenance=shared),
        continuous_source(
            np.zeros(8, np.float32),
            id="c2",
            provenance={"software": "shared 1.0"},
        ),
        continuous_source(np.zeros(8, np.float32), id="c3"),
        continuous_source(
            np.zeros(8, np.float32), id="c4", provenance={"software": "own 2"}
        ),
    ]
    events = [
        event_source([1], id="e0", provenance={"software": "own 3"}),
        event_source([1], id="e1"),
        event_source([1], id="e2", provenance=shared),
        event_source([1], id="e3", provenance={}),
    ]
    output = tmp_path / "bundle"
    write_bundle(continuous, events, output_path=output, opts=_OPTS)
    expected = {
        "0": shared,
        "1": shared,
        "2": {"software": "shared 1.0"},
        "3": None,
        "4": {"software": "own 2"},
        "5": {"software": "own 3"},
        "6": None,
        "7": shared,
        "8": {},
    }
    _assert_provenance_matches(_channel_attrs(output), expected)
    _assert_provenance_matches(_consolidated_attrs(output), expected)


def _strip_provenance(document):
    """Return document without any provenance key under a tsz object."""
    document = json.loads(json.dumps(document))
    nodes = [document]
    nodes += (
        document.get("consolidated_metadata", {}).get("metadata", {}).values()
    )
    for node in nodes:
        node.get("attributes", {}).get("tsz", {}).pop("provenance", None)
    return document


def test_write_bundle_provenance_changes_no_other_stored_object(
    tmp_path, continuous_source, event_source
):
    plain = _write_mixed(
        tmp_path,
        continuous_source,
        event_source,
        with_provenance=False,
        name="plain",
    )
    marked = _write_mixed(
        tmp_path,
        continuous_source,
        event_source,
        with_provenance=True,
        name="marked",
    )
    files = sorted(
        p.relative_to(plain) for p in plain.rglob("*") if p.is_file()
    )
    assert files == sorted(
        p.relative_to(marked) for p in marked.rglob("*") if p.is_file()
    )
    assert len(files) > 20
    for file in files:
        if file.name == "zarr.json":
            before = json.loads((plain / file).read_text())
            after = json.loads((marked / file).read_text())
            assert _strip_provenance(after) == before, file
        else:
            assert (plain / file).read_bytes() == (marked / file).read_bytes()
    assert (plain / "zarr.json").read_text() != (
        marked / "zarr.json"
    ).read_text()


@pytest.mark.parametrize(
    "opts",
    [
        dataclasses.replace(_SPEC_OPTS, write_var=True),
        dataclasses.replace(_SPEC_OPTS, max_levels=0),
        dataclasses.replace(_SPEC_OPTS, voltage_policy="require_uv"),
        dataclasses.replace(_SPEC_OPTS, zstd_level=1, max_levels=2),
    ],
    ids=["write_var", "no_levels", "require_uv", "two_levels"],
)
def test_write_bundle_carries_provenance_under_every_write_option(
    tmp_path, continuous_source, event_source, opts
):
    output = _write_mixed(tmp_path, continuous_source, event_source, opts=opts)
    _validate_bundle(output)
    _assert_provenance_matches(_channel_attrs(output), _MIXED_PROVENANCE)
    _assert_provenance_matches(_consolidated_attrs(output), _MIXED_PROVENANCE)


def test_write_bundle_carries_provenance_on_a_channel_converted_to_microvolts(
    tmp_path, continuous_source
):
    source = continuous_source(
        np.linspace(-3.0, 3.0, 64, dtype=np.float32),
        unit="mV",
        provenance=_ACQUIRE,
    )
    output = tmp_path / "bundle"
    write_bundle([source], [], output_path=output, opts=_OPTS)
    attrs = _channel_attrs(output)["0"]
    assert attrs["unit"] == "uV"
    assert attrs["provenance"] == _ACQUIRE


def test_write_bundle_carries_provenance_on_a_channel_too_short_for_a_level(
    tmp_path, continuous_source
):
    short = continuous_source(_signal(100, 5), id="short", provenance=_ACQUIRE)
    output = tmp_path / "bundle"
    write_bundle([short], [], output_path=output, opts=_SPEC_OPTS)
    nodes = _validate_bundle(output)
    assert sorted(p for p in nodes if p.startswith("0/")) == ["0/raw"]
    assert nodes["0"]["attributes"]["tsz"]["provenance"] == _ACQUIRE


def test_write_bundle_carries_provenance_on_an_event_channel_with_no_events(
    tmp_path, event_source
):
    empty = event_source([], start_us=0, id="none", provenance=_REVIEW)
    output = tmp_path / "bundle"
    write_bundle([], [empty], output_path=output, opts=_OPTS)
    assert _channel_attrs(output)["0"]["provenance"] == _REVIEW


def test_write_bundle_replacing_a_bundle_drops_the_provenance_it_held(
    tmp_path, continuous_source, event_source
):
    _write_mixed(tmp_path, continuous_source, event_source)
    output = _write_mixed(
        tmp_path, continuous_source, event_source, with_provenance=False
    )
    _assert_provenance_matches(
        _channel_attrs(output), dict.fromkeys(_MIXED_PROVENANCE)
    )
    for file in output.rglob("zarr.json"):
        assert "provenance" not in file.read_text(), file
    assert not any(
        b"acquire" in f.read_bytes() for f in output.rglob("*") if f.is_file()
    )


_BAD_PROVENANCE = [
    pytest.param({"threshold": float("nan")}, "finite", id="nan"),
    pytest.param({"bands": [1.0, float("inf")]}, "finite", id="infinity"),
    pytest.param({"when": dt.datetime(2024, 1, 1)}, "JSON", id="datetime"),
    pytest.param({"threshold": np.float32(3.5)}, "JSON", id="numpy-float"),
    pytest.param(["acquire"], "must be a dict", id="list"),
]


@pytest.mark.parametrize(("provenance", "reason"), _BAD_PROVENANCE)
def test_write_bundle_rejects_a_bad_continuous_provenance_before_writing(
    tmp_path, continuous_source, event_source, provenance, reason
):
    output = tmp_path / "bundle"
    good = event_source([1], id="e0", provenance=_REVIEW)
    bad = continuous_source(
        np.zeros(8, np.float32), id="c9", provenance=provenance
    )
    with pytest.raises(ValueError, match=rf"channel 0 \('c9'\).*{reason}"):
        write_bundle([bad], [good], output_path=output, opts=_OPTS)
    assert not output.exists()


@pytest.mark.parametrize(("provenance", "reason"), _BAD_PROVENANCE)
def test_write_bundle_rejects_a_bad_event_provenance_before_writing(
    tmp_path, continuous_source, event_source, provenance, reason
):
    output = tmp_path / "bundle"
    good = continuous_source(np.zeros(8, np.float32), id="c0")
    bad = event_source([1], id="e9", provenance=provenance)
    with pytest.raises(ValueError, match=rf"channel 1 \('e9'\).*{reason}"):
        write_bundle([good], [bad], output_path=output, opts=_OPTS)
    assert not output.exists()


def test_write_bundle_leaves_an_existing_bundle_when_provenance_is_invalid(
    tmp_path, continuous_source, event_source
):
    output = _write_mixed(tmp_path, continuous_source, event_source)
    before = (output / "zarr.json").read_text()
    bad = event_source([1], id="e9", provenance={"threshold": float("nan")})
    with pytest.raises(ValueError, match="provenance"):
        write_bundle([], [bad], output_path=output, opts=_OPTS)
    assert (output / "zarr.json").read_text() == before
    _assert_provenance_matches(_channel_attrs(output), _MIXED_PROVENANCE)


def test_write_bundle_accepts_sources_that_have_no_provenance_method(
    tmp_path,
):
    class Bare:
        """A continuous source written before provenance existed."""

        id, name, unit = "bare-c", "bare", "uV"

        def rate_hz(self):
            return 512.0

        def start_us(self):
            return 0

        def num_samples(self):
            return 64

        def read_samples(self, start, stop):
            return np.zeros(stop - start, dtype=np.float32)

    class BareEvents:
        id, name, unit = "bare-e", "bare events", None

        def start_us(self):
            return 0

        def num_events(self):
            return 2

        def columns(self):
            return frozenset()

        def label_names(self):
            return None

        def num_labels(self):
            return 0

        def max_duration_us(self):
            return None

        def text_width(self):
            return 0

        def read_events(self, start, stop):
            return np.array([1, 2], dtype=np.int64)[start:stop]

        def read_durations(self, start, stop):
            raise NotImplementedError

        def read_labels(self, start, stop):
            raise NotImplementedError

        def read_text(self, start, stop):
            raise NotImplementedError

        def read_channel_ids(self, start, stop):
            raise NotImplementedError

        def waveform_mean(self):
            raise NotImplementedError

        def waveform_period_us(self):
            return 1.0

    output = tmp_path / "bundle"
    write_bundle([Bare()], [BareEvents()], output_path=output, opts=_OPTS)
    _assert_provenance_matches(_channel_attrs(output), {"0": None, "1": None})
    assert _channel_attrs(output)["1"]["kind"] == "event"


def test_write_bundle_keeps_list_order_across_single_and_multi_channel_sources(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
    event_source: type[ArrayEventSource],
) -> None:
    first, last = _zeros(continuous_source, "first", "last", samples=8)
    many = multi_channel_source(
        _zeros(continuous_source, "m0", "m1", samples=8)
    )
    output = tmp_path / "bundle"
    write_bundle(
        [first, many, last],
        [event_source([1], id="marks")],
        output_path=output,
        opts=_OPTS,
    )
    attrs = _channel_attrs(output)
    assert {key: item["id"] for key, item in attrs.items()} == {
        "0": "first",
        "1": "m0",
        "2": "m1",
        "3": "last",
        "4": "marks",
    }


def test_write_bundle_names_a_multi_channel_source_channel_in_a_unit_error(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    (solo,) = _zeros(continuous_source, "solo", samples=8)
    many = multi_channel_source(
        [
            continuous_source(np.zeros(8, np.float32), id="m0"),
            continuous_source(np.zeros(8, np.float32), id="m1", unit="volts"),
        ]
    )
    output = tmp_path / "bundle"
    opts = WriteOpts(voltage_policy="require_uv")
    with pytest.raises(ValueError, match=r"channel 2 \('m1'\).*'volts'"):
        write_bundle([solo, many], [], output_path=output, opts=opts)
    assert not output.exists()


@pytest.mark.parametrize(("provenance", "reason"), _BAD_PROVENANCE)
def test_write_bundle_rejects_a_bad_multi_channel_source_provenance_before_writing(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    multi_channel_source: type[ArrayMultiChannelSource],
    provenance: object,
    reason: str,
) -> None:
    many = multi_channel_source(
        [
            continuous_source(np.zeros(8, np.float32), id="m0"),
            continuous_source(
                np.zeros(8, np.float32), id="m1", provenance=provenance
            ),
        ]
    )
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match=rf"channel 1 \('m1'\).*{reason}"):
        write_bundle([many], [], output_path=output, opts=_OPTS)
    assert not output.exists()


def test_write_bundle_with_a_multi_channel_source_conforms_to_the_specification_validation_list(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    event_source: type[ArrayEventSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    continuous, events = _mixed_sources(continuous_source, event_source)
    many = multi_channel_source(continuous, window_rows=[700, 1000])
    output = tmp_path / "bundle"
    write_bundle(
        [many],
        events,
        output_path=output,
        opts=_SPEC_OPTS,
        recording=_RECORDING,
    )
    nodes = _validate_bundle(output)
    assert sorted(p for p in nodes if p and "/" not in p) == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]
    assert _check_recording(output, nodes) == [_SESSION, _SESSION]
    _assert_provenance_matches(_channel_attrs(output), _MIXED_PROVENANCE)


def _bundle_files(path: Path) -> dict[str, bytes]:
    return {
        file.relative_to(path).as_posix(): file.read_bytes()
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def test_write_bundle_writes_the_same_files_from_a_multi_channel_source_as_from_its_channels(
    tmp_path: Path,
    continuous_source: type[ArrayContinuousSource],
    event_source: type[ArrayEventSource],
    multi_channel_source: type[ArrayMultiChannelSource],
) -> None:
    channels = [
        continuous_source(
            _signal(1000, 1),
            id="a",
            rate_hz=256.0,
            start_us=_SESSION_US + 1_000,
            provenance=_ACQUIRE,
        ),
        continuous_source(
            _signal(517, 2),
            id="b",
            rate_hz=1000.0,
            start_us=_SESSION_US + 50_000,
            unit="mV",
        ),
        continuous_source(
            _signal(130, 3),
            id="c",
            rate_hz=64.0,
            start_us=_SESSION_US - 20_000,
        ),
    ]
    marks = event_source(
        _SESSION_US + np.array([2_000, 1_500_000, 3_000_000]),
        labels=[0, 1, 0],
        label_names=["spike", "seizure"],
        channel_ids=[["a"], ["b", "c"], []],
        id="marks",
        provenance=_REVIEW,
    )
    per_channel = tmp_path / "per-channel"
    sweep = tmp_path / "sweep"
    write_bundle(
        channels,
        [marks],
        output_path=per_channel,
        opts=_OPTS,
        recording=_RECORDING,
    )
    write_bundle(
        [multi_channel_source(channels, window_rows=(7, 5, 13))],
        [marks],
        output_path=sweep,
        opts=_OPTS,
        recording=_RECORDING,
    )
    left, right = _bundle_files(per_channel), _bundle_files(sweep)
    assert len(left) > 30
    assert left.keys() == right.keys()
    assert [name for name in left if left[name] != right[name]] == []


def _without(source, *names):
    """Return a copy of source that lacks the members called names."""
    cls = type(source)
    members = {
        name: getattr(cls, name)
        for name in dir(cls)
        if not name.startswith("__") and name not in names
    }
    partial = type("Partial", (), members)()
    partial.__dict__.update(
        {k: v for k, v in vars(source).items() if k not in names}
    )
    return partial


def test_write_bundle_names_the_member_an_event_source_lacks(
    tmp_path, event_source
):
    output = tmp_path / "bundle"
    partial = _without(event_source([1], id="e9"), "waveform_mean")
    with pytest.raises(
        ValueError,
        match=r"events\[0\] \('e9'\) does not implement EventChannelSource; "
        "missing waveform_mean$",
    ):
        write_bundle([], [partial], output_path=output, opts=_OPTS)
    assert not output.exists()


def test_write_bundle_lists_every_member_an_event_source_lacks_in_order(
    tmp_path, event_source
):
    partial = _without(event_source([1]), "waveform_period_us", "read_text")
    with pytest.raises(
        ValueError, match="missing read_text, waveform_period_us"
    ):
        write_bundle([], [partial], output_path=tmp_path / "bundle", opts=_OPTS)


def test_write_bundle_names_the_member_a_continuous_source_lacks(
    tmp_path, continuous_source
):
    output = tmp_path / "bundle"
    partial = _without(
        continuous_source(np.zeros(8, np.float32), id="c3"), "read_samples"
    )
    with pytest.raises(
        ValueError,
        match=r"continuous\[0\] \('c3'\) does not implement "
        "ContinuousChannelSource; missing read_samples$",
    ):
        write_bundle([partial], [], output_path=output, opts=_OPTS)
    assert not output.exists()


def test_write_bundle_names_the_member_a_multi_channel_source_lacks(
    tmp_path, continuous_source, multi_channel_source
):
    output = tmp_path / "bundle"
    multi = multi_channel_source([continuous_source(np.zeros(8, np.float32))])
    partial = _without(multi, "iter_windows")
    with pytest.raises(
        ValueError,
        match=r"continuous\[0\] does not implement MultiChannelSource; "
        "missing iter_windows$",
    ):
        write_bundle([partial], [], output_path=output, opts=_OPTS)
    assert not output.exists()


def test_write_bundle_holds_a_source_with_channels_to_the_multi_channel_protocol(
    tmp_path, continuous_source, multi_channel_source
):
    multi = multi_channel_source([continuous_source(np.zeros(8, np.float32))])
    partial = _without(multi, "channels")
    with pytest.raises(
        ValueError, match="MultiChannelSource; missing channels"
    ):
        write_bundle([partial], [], output_path=tmp_path / "bundle", opts=_OPTS)


def test_write_bundle_accepts_a_continuous_source_with_an_unrelated_channels_attribute(
    tmp_path, continuous_source
):
    source = continuous_source(np.zeros(8, np.float32))
    source.channels = ("a", "b")
    write_bundle([source], [], output_path=tmp_path / "bundle", opts=_OPTS)
    assert (tmp_path / "bundle" / "0").is_dir()


def test_write_bundle_names_a_member_set_to_none_as_missing(
    tmp_path, continuous_source
):
    source = continuous_source(np.zeros(8, np.float32))
    source.rate_hz = None
    with pytest.raises(
        ValueError,
        match=r"continuous\[0\] .*does not implement "
        "ContinuousChannelSource; missing rate_hz$",
    ):
        write_bundle([source], [], output_path=tmp_path / "bundle", opts=_OPTS)


def test_write_bundle_names_the_member_a_listed_channel_lacks(
    tmp_path, continuous_source, multi_channel_source
):
    output = tmp_path / "bundle"
    good = continuous_source(np.zeros(8, np.float32), id="c0")
    bad = _without(continuous_source(np.zeros(8, np.float32), id="c1"), "unit")
    multi = multi_channel_source([good, bad])
    with pytest.raises(
        ValueError,
        match=r"continuous\[0\]\.channels\(\)\[1\] \('c1'\) does not "
        "implement ContinuousChannelMetadata; missing unit$",
    ):
        write_bundle([multi], [], output_path=output, opts=_OPTS)
    assert not output.exists()


def test_write_bundle_rejects_an_event_source_listed_as_continuous(
    tmp_path, event_source
):
    output = tmp_path / "bundle"
    with pytest.raises(
        ValueError,
        match=r"continuous\[0\] \('ch-0'\) does not implement "
        "ContinuousChannelSource; missing .*rate_hz",
    ):
        write_bundle(
            [event_source([1], id="ch-0")],
            [],
            output_path=output,
            opts=_OPTS,
        )
    assert not output.exists()


def test_write_bundle_leaves_an_existing_bundle_when_a_source_is_incomplete(
    tmp_path, continuous_source, event_source
):
    output = _write_mixed(tmp_path, continuous_source, event_source)
    before = (output / "zarr.json").read_text()
    partial = _without(event_source([1], id="e9"), "waveform_mean")
    with pytest.raises(ValueError, match="missing waveform_mean"):
        write_bundle([], [partial], output_path=output, opts=_OPTS)
    assert (output / "zarr.json").read_text() == before


def _assert_existing_bundle_kept(output, before):
    assert (output / "zarr.json").read_text() == before
    assert open_group(output)["0"]["raw"][:].tolist() == list(range(8))


@pytest.mark.parametrize(
    "rate_hz", [float("nan"), float("inf"), 0.0], ids=["nan", "inf", "zero"]
)
@pytest.mark.parametrize("samples", [0, 8], ids=["empty", "samples"])
def test_write_bundle_rejects_a_rate_that_is_not_finite_and_positive_before_writing(
    tmp_path, continuous_source, samples, rate_hz
):
    output = tmp_path / "bundle"
    bad = continuous_source(
        np.zeros(samples, np.float32), id="bad", rate_hz=rate_hz
    )
    with pytest.raises(ValueError, match=r"channel 0 \('bad'\).*rate_hz"):
        write_bundle([bad], [], output_path=output, opts=_OPTS)
    assert not output.exists()


@pytest.mark.parametrize("start_us", [2**53, -(2**53)])
@pytest.mark.parametrize("samples", [0, 8], ids=["empty", "samples"])
def test_write_bundle_rejects_a_channel_offset_at_the_time_limit_before_writing(
    tmp_path, continuous_source, samples, start_us
):
    output = tmp_path / "bundle"
    first = continuous_source(np.arange(8, dtype=np.float32), id="c0")
    write_bundle([first], [], output_path=output, opts=_OPTS)
    before = (output / "zarr.json").read_text()
    bad = continuous_source(
        np.zeros(samples, np.float32), id="bad", start_us=start_us
    )
    with pytest.raises(ValueError, match=r"offset_us.*\(-2\^53, 2\^53\)"):
        write_bundle([first, bad], [], output_path=output, opts=_OPTS)
    _assert_existing_bundle_kept(output, before)


def test_write_bundle_rejects_a_grid_that_ends_past_the_time_limit_before_writing(
    tmp_path, continuous_source
):
    output = tmp_path / "bundle"
    first = continuous_source(np.zeros(8, np.float32), id="c0")
    bad = continuous_source(
        np.zeros(100, np.float32), id="bad", rate_hz=1.0, start_us=2**53 - 10
    )
    with pytest.raises(ValueError, match="grid time of the last sample"):
        write_bundle([first, bad], [], output_path=output, opts=_OPTS)
    assert not output.exists()


def test_write_bundle_rejects_an_event_time_at_the_time_limit_before_writing(
    tmp_path, continuous_source, event_source
):
    output = tmp_path / "bundle"
    first = continuous_source(np.arange(8, dtype=np.float32), id="c0")
    write_bundle([first], [], output_path=output, opts=_OPTS)
    before = (output / "zarr.json").read_text()
    bad = event_source([0, 2**53], id="e9")
    with pytest.raises(ValueError, match=r"channel 0 \('e9'\).*an event time"):
        write_bundle([], [bad], output_path=output, opts=_OPTS)
    _assert_existing_bundle_kept(output, before)


def test_write_bundle_rejects_a_duration_bound_at_the_time_limit_before_writing(
    tmp_path, event_source
):
    output = tmp_path / "bundle"
    bad = event_source([1], durations=[0], max_duration_us=2**53, id="e9")
    with pytest.raises(ValueError, match="max_duration_us"):
        write_bundle([], [bad], output_path=output, opts=_OPTS)
    assert not output.exists()


def test_write_bundle_rejects_2_to_the_32_events_before_writing(
    tmp_path, event_source
):
    class Huge(type(event_source([0]))):
        def num_events(self):
            return 2**32

    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="fewer than 2.32 events"):
        write_bundle([], [Huge([0])], output_path=output, opts=_OPTS)
    assert not output.exists()


def test_write_bundle_rejects_a_non_finite_recording_value_before_writing(
    tmp_path, continuous_source
):
    output = tmp_path / "bundle"
    first = continuous_source(np.arange(8, dtype=np.float32), id="c0")
    write_bundle([first], [], output_path=output, opts=_OPTS)
    before = (output / "zarr.json").read_text()
    recording = RecordingMeta(subject={"weight_kg": float("nan")})
    with pytest.raises(ValueError, match=r"recording\.json.*finite"):
        write_bundle(
            [first], [], output_path=output, opts=_OPTS, recording=recording
        )
    _assert_existing_bundle_kept(output, before)
    assert not (output / RECORDING_FILE).exists()


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_write_bundle_rejects_an_infinite_recording_value_before_writing(
    tmp_path, continuous_source, value
):
    output = tmp_path / "bundle"
    first = continuous_source(np.arange(8, dtype=np.float32), id="c0")
    recording = RecordingMeta(source={"gain": [1.0, value]})
    with pytest.raises(ValueError, match=r"recording\.json.*finite"):
        write_bundle(
            [first], [], output_path=output, opts=_OPTS, recording=recording
        )
    assert not output.exists()


def test_write_bundle_rejects_a_recording_value_that_is_not_json_before_writing(
    tmp_path, continuous_source
):
    output = tmp_path / "bundle"
    first = continuous_source(np.arange(8, dtype=np.float32), id="c0")
    recording = RecordingMeta(subject={"tags": {1, 2}})
    with pytest.raises(ValueError, match=r"recording\.json.*JSON"):
        write_bundle(
            [first], [], output_path=output, opts=_OPTS, recording=recording
        )
    assert not output.exists()
