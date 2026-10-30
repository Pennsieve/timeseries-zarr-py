import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from timeseries_zarr import open_bundle, validate_bundle

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_fixtures.py"
CONFORMANCE = ROOT / "conformance"
MANIFEST = json.loads((CONFORMANCE / "manifest.json").read_text("utf-8"))
FIXTURES = MANIFEST["fixtures"]
CONFORMING = [name for name, entry in FIXTURES.items() if not entry["findings"]]
NONCONFORMING = [name for name, entry in FIXTURES.items() if entry["findings"]]
VALUED = [name for name, entry in FIXTURES.items() if "channels" in entry]
MEAN_TOLERANCE = 2**-20


def _build(directory):
    subprocess.run(
        [sys.executable, str(SCRIPT), str(directory)],
        check=True,
        capture_output=True,
    )
    return directory


def _tree(directory):
    return {
        file.relative_to(directory).as_posix(): file.read_bytes()
        for file in sorted(directory.rglob("*"))
        if file.is_file()
    }


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("first") / "conformance")


def _nan_or(expected):
    return math.nan if expected is None else expected


def _close(actual, expected):
    if expected is None:
        return math.isnan(actual)
    return math.isclose(actual, expected, rel_tol=MEAN_TOLERANCE, abs_tol=1e-9)


def _same(actual, expected):
    if expected is None:
        return math.isnan(actual)
    return actual == expected


def test_the_script_writes_the_same_bytes_on_every_run(built, tmp_path):
    second = _build(tmp_path / "conformance")
    assert _tree(second) == _tree(built)


def test_the_script_refuses_to_replace_a_directory_without_a_manifest(tmp_path):
    (tmp_path / "keep.txt").write_text("data")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "manifest.json" in result.stderr
    assert (tmp_path / "keep.txt").read_text() == "data"


def test_the_committed_manifest_is_what_the_script_writes(built):
    committed = json.loads((CONFORMANCE / "manifest.json").read_text("utf-8"))
    assert json.loads((built / "manifest.json").read_text("utf-8")) == committed


def test_the_manifest_lists_every_built_bundle(built):
    on_disk = {
        file.parent.relative_to(built).as_posix()
        for pattern in ("*/zarr.json", "invalid/*/zarr.json")
        for file in built.glob(pattern)
    }
    assert {entry["path"] for entry in FIXTURES.values()} == on_disk


def test_the_total_size_stays_small(built):
    size = sum(
        file.stat().st_size for file in built.rglob("*") if file.is_file()
    )
    assert size < 4_000_000


@pytest.mark.parametrize("name", CONFORMING)
def test_a_conforming_fixture_has_no_finding(name, built):
    path = built / FIXTURES[name]["path"]
    assert validate_bundle(path, full=True) == []
    assert validate_bundle(path) == []


@pytest.mark.parametrize("name", NONCONFORMING)
@pytest.mark.parametrize("full", [False, True])
def test_a_nonconforming_fixture_yields_its_named_findings(name, full, built):
    path = built / FIXTURES[name]["path"]
    findings = validate_bundle(path, full=full)
    expected = FIXTURES[name]["findings"]
    assert len(findings) == len(expected), findings
    for finding, named in zip(findings, expected, strict=True):
        assert finding.path == named["path"]
        assert finding.section == named["section"]
        assert named["message_contains"] in finding.message


def test_a_reader_rejects_a_root_without_consolidated_metadata(built):
    rejected = [
        entry["path"]
        for entry in FIXTURES.values()
        if entry["reader_must_reject"]
    ]
    assert rejected == ["invalid/no_consolidated_metadata"]
    with pytest.raises(ValueError, match="consolidated metadata"):
        open_bundle(built / rejected[0])


@pytest.mark.parametrize("name", VALUED)
def test_the_manifest_channels_match_the_reader(name, built):
    entry = FIXTURES[name]
    with open_bundle(built / entry["path"]) as bundle:
        assert bundle.version == entry["version"]
        assert len(bundle.channels) == entry["channel_count"]
        for channel in entry["channels"]:
            if channel["kind"] == "continuous":
                _check_continuous(bundle.continuous[channel["index"]], channel)
            else:
                _check_event(bundle.events[channel["index"]], channel)


@pytest.mark.parametrize("name", VALUED)
def test_the_manifest_recording_matches_the_file_and_the_reader(name, built):
    entry = FIXTURES[name]
    path = built / entry["path"]
    expected = entry["recording_json"]
    with open_bundle(path) as bundle:
        if expected is None:
            assert not (path / "recording.json").exists()
            assert bundle.recording is None
            return
        assert (
            json.loads((path / "recording.json").read_text("utf-8")) == expected
        )
        assert bundle.recording is not None
        reference = dt.datetime.fromisoformat(expected["reference_time"])
        assert bundle.recording.reference_time == reference


def _check_continuous(channel, expected):
    for key in ("name", "id", "unit", "rate_hz", "offset_us"):
        assert getattr(channel, key) == expected[key], key
    assert channel.num_samples == expected["num_samples"]
    if expected["runs"] is None:
        assert channel.runs is None
    else:
        assert channel.runs.tolist() == expected["runs"]
    for sample in expected["samples"]:
        actual = float(channel.raw[sample["index"]])
        assert _same(actual, sample["value"]), sample
    assert list(channel.levels) == [
        level["level"] for level in expected["levels"]
    ]
    for described in expected["levels"]:
        level = channel.levels[described["level"]]
        assert level.period_us == described["period_us"]
        assert level.samples_per_bin == described["samples_per_bin"]
        assert level.bins == described["bins"]
        assert (level.var is None) == ("var" not in described["members"])
        for values in described["bin_values"]:
            _check_bin(level, values)


def _check_bin(level, expected):
    index = expected["index"]
    low, high = (float(value) for value in level.minmax[index])
    assert _same(low, expected["min"]), expected
    assert _same(high, expected["max"]), expected
    assert int(level.count[index]) == expected["count"], expected
    assert _close(float(level.mean[index]), expected["mean"]), expected
    if "var" in expected:
        assert _close(float(level.var[index]), expected["var"]), expected


def _check_event(channel, expected):
    assert channel.name == expected["name"]
    assert channel.id == expected["id"]
    assert channel.unit == expected["unit"]
    assert list(channel.label_names) == expected["label_names"]
    assert channel.num_labels() == expected["num_labels"]
    assert channel.max_duration_us == expected["max_duration_us"]
    assert channel.num_events == expected["num_events"]
    assert list(channel.columns) == expected["columns"]
    table = channel.read()
    assert table.time_us.tolist() == expected["events"]
    if "durations" in expected:
        assert table.duration_us.tolist() == expected["durations"]
    if "labels" in expected:
        assert table.label.tolist() == expected["labels"]
    if "values" in expected:
        assert table.value.tolist() == expected["values"]
    if "text" in expected:
        assert channel.text.shape[1] == expected["text_width"]
        assert table.text == expected["text"]
    if "channels" in expected:
        assert channel.channels.shape[1] == expected["channel_bytes"]
        assert table.channel_indices == expected["channels"]
    if "waveform" in expected:
        _check_waveforms(channel, table, expected["waveform"])
    assert list(channel.count_levels) == [
        level["level"] for level in expected["count_levels"]
    ]
    for described in expected["count_levels"]:
        level = channel.count_levels[described["level"]]
        assert level.period_us == described["period_us"]
        assert level.offset_us == described["offset_us"]
        assert level.bins == described["bins"]
        assert level.num_labels == expected["num_labels"]
        assert level.count[:].tolist() == described["count"]


def _check_waveforms(channel, table, expected):
    for snippets in (channel.waveforms, channel.waveform_mean):
        assert snippets.samples == expected["samples"]
        assert snippets.period_us == expected["period_us"]
        assert snippets.event_sample == expected["event_sample"]
    stored = np.asarray(table.waveform.values, dtype=np.float64)
    rows = np.array(
        [[_nan_or(value) for value in row] for row in expected["waveforms"]]
    ).reshape(stored.shape)
    np.testing.assert_array_equal(stored, rows)
    means = np.array(
        [[_nan_or(value) for value in row] for row in expected["waveform_mean"]]
    )
    stored_means = np.asarray(channel.waveform_mean.array[:], dtype=np.float64)
    np.testing.assert_allclose(
        stored_means, means, rtol=MEAN_TOLERANCE, atol=1e-9
    )
