"""The onset-relative timeline and the meta/ group that holds the one date."""

import json

import numpy as np
import pytest

from timeseries_zarr.bundle import bundle_onset_us, write_bundle
from timeseries_zarr.types import RecordingMeta, WriteOpts
from timeseries_zarr.zarr_io import open_group

_OPTS = WriteOpts(
    min_bins=2, max_levels=8, inner_len=16, target_shard_bytes=256
)

# A plausible wall clock: 2026-09-18T14:50:34Z in microseconds. Large enough
# that it cannot collide with a sample value, an index, or a period.
_EPOCH_US = 1_789_743_034_000_000


def _meta():
    return RecordingMeta(
        subject={"subject_id": "sub-01", "species": "Homo sapiens"},
        session={"session_id": "ses-01"},
        source={"converter": "timeseries-zarr-py"},
    )


def _write(tmp_path, continuous=(), units=(), meta=None):
    final = tmp_path / "bundle"
    write_bundle(
        list(continuous),
        list(units),
        staging_dir=tmp_path / "staging",
        final_dir=final,
        opts=_OPTS,
        meta=meta,
    )
    return final


def _read_meta(bundle):
    return json.loads((bundle / "meta" / "zarr.json").read_text())["attributes"]


def test_bundle_onset_is_the_earliest_start(continuous_source):
    late = continuous_source(np.zeros(4, dtype=np.float32), start_us=900)
    early = continuous_source(np.zeros(4, dtype=np.float32), start_us=100)
    assert bundle_onset_us([late, early]) == 100


def test_bundle_onset_of_no_channels_is_zero():
    assert bundle_onset_us([]) == 0


def test_channel_offsets_are_relative_to_the_onset(tmp_path, continuous_source):
    samples = np.arange(8, dtype=np.float32)
    bundle = _write(
        tmp_path,
        continuous=[
            continuous_source(samples, id="c0", start_us=_EPOCH_US),
            continuous_source(
                samples, id="c1", start_us=_EPOCH_US + 30_000_000
            ),
        ],
        meta=_meta(),
    )
    root = open_group(bundle)
    # The earliest channel is the onset itself, so it sits at zero; the other
    # records its 30 s distance and neither carries the wall clock.
    assert dict(root["0"].attrs)["offset_us"] == 0
    assert dict(root["1"].attrs)["offset_us"] == 30_000_000


def test_channel_offsets_are_never_negative(tmp_path, continuous_source):
    samples = np.arange(8, dtype=np.float32)
    bundle = _write(
        tmp_path,
        continuous=[
            continuous_source(samples, id="c0", start_us=_EPOCH_US + 5),
            continuous_source(samples, id="c1", start_us=_EPOCH_US),
        ],
        meta=_meta(),
    )
    root = open_group(bundle)
    offsets = [dict(root[key].attrs)["offset_us"] for key in ("0", "1")]
    assert min(offsets) == 0
    assert all(offset >= 0 for offset in offsets)


def test_event_timestamps_are_onset_relative(
    tmp_path, continuous_source, unit_source
):
    events = np.array([0, 10, 25], dtype=np.int64) + _EPOCH_US
    bundle = _write(
        tmp_path,
        continuous=[
            continuous_source(
                np.arange(8, dtype=np.float32), id="c0", start_us=_EPOCH_US
            )
        ],
        units=[unit_source(events, id="u0", start_us=_EPOCH_US)],
        meta=_meta(),
    )
    stored = open_group(bundle)["1"]["events"][:]
    assert np.array_equal(stored, np.array([0, 10, 25], dtype=np.int64))


def test_meta_holds_the_onset_as_the_only_wall_clock(
    tmp_path, continuous_source
):
    bundle = _write(
        tmp_path,
        continuous=[
            continuous_source(
                np.arange(8, dtype=np.float32), start_us=_EPOCH_US
            )
        ],
        meta=_meta(),
    )
    attributes = _read_meta(bundle)
    assert attributes["session"]["start_us"] == _EPOCH_US
    assert attributes["subject"]["subject_id"] == "sub-01"
    assert attributes["source"]["converter"] == "timeseries-zarr-py"


def test_supplied_session_start_us_does_not_win(tmp_path, continuous_source):
    meta = RecordingMeta(subject={}, session={"start_us": 1}, source={})
    bundle = _write(
        tmp_path,
        continuous=[
            continuous_source(
                np.arange(8, dtype=np.float32), start_us=_EPOCH_US
            )
        ],
        meta=meta,
    )
    # The onset is the bundle's to compute; a caller cannot contradict it.
    assert _read_meta(bundle)["session"]["start_us"] == _EPOCH_US


def test_no_wall_clock_appears_outside_meta(
    tmp_path, continuous_source, unit_source
):
    """The guarantee the whole step exists for, asserted over the written bytes.

    Scanning every metadata object rather than named attributes means a new
    attribute carrying a date cannot slip past this test later.
    """
    bundle = _write(
        tmp_path,
        continuous=[
            continuous_source(
                np.arange(8, dtype=np.float32), id="c0", start_us=_EPOCH_US
            )
        ],
        units=[
            unit_source(
                np.array([0, 5], dtype=np.int64) + _EPOCH_US,
                id="u0",
                start_us=_EPOCH_US,
            )
        ],
        meta=_meta(),
    )
    needle = str(_EPOCH_US)
    outside = [
        path
        for path in bundle.rglob("zarr.json")
        if path.parent.name != "meta" and needle in path.read_text()
    ]
    assert outside == []
    assert needle in (bundle / "meta" / "zarr.json").read_text()


def test_meta_is_excluded_from_consolidated_metadata(
    tmp_path, continuous_source
):
    bundle = _write(
        tmp_path,
        continuous=[
            continuous_source(
                np.arange(8, dtype=np.float32), start_us=_EPOCH_US
            )
        ],
        meta=_meta(),
    )
    root = json.loads((bundle / "zarr.json").read_text())
    consolidated = root["consolidated_metadata"]["metadata"]
    # Were meta/ inlined here, deleting the directory would leave a copy of
    # the subject behind in the one object every cache holds.
    assert "meta" not in consolidated
    assert "0" in consolidated


def test_deleting_meta_leaves_a_readable_bundle(tmp_path, continuous_source):
    samples = np.arange(8, dtype=np.float32)
    bundle = _write(
        tmp_path,
        continuous=[continuous_source(samples, id="c0", start_us=_EPOCH_US)],
        meta=_meta(),
    )
    for path in sorted(
        (bundle / "meta").rglob("*"), key=lambda p: -len(p.parts)
    ):
        path.unlink()
    (bundle / "meta").rmdir()

    root = open_group(bundle)
    assert np.array_equal(root["0"]["raw"][:], samples)
    assert dict(root["0"].attrs)["offset_us"] == 0


def test_no_meta_writes_no_directory(tmp_path, continuous_source):
    bundle = _write(
        tmp_path,
        continuous=[
            continuous_source(
                np.arange(8, dtype=np.float32), start_us=_EPOCH_US
            )
        ],
        meta=None,
    )
    assert not (bundle / "meta").exists()


def test_meta_zarr_json_is_a_v3_group_node(tmp_path, continuous_source):
    bundle = _write(
        tmp_path,
        continuous=[
            continuous_source(
                np.arange(8, dtype=np.float32), start_us=_EPOCH_US
            )
        ],
        meta=_meta(),
    )
    node = json.loads((bundle / "meta" / "zarr.json").read_text())
    assert node["zarr_format"] == 3
    assert node["node_type"] == "group"


@pytest.mark.parametrize("key", ["subject", "session", "source"])
def test_meta_carries_all_three_objects(tmp_path, continuous_source, key):
    bundle = _write(
        tmp_path,
        continuous=[
            continuous_source(
                np.arange(8, dtype=np.float32), start_us=_EPOCH_US
            )
        ],
        meta=_meta(),
    )
    assert key in _read_meta(bundle)
