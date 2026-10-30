import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
import zarr

from timeseries_zarr.validate import main


@pytest.fixture
def bundle(full_bundle, tmp_path):
    destination = tmp_path / "bundle"
    shutil.copytree(full_bundle, destination)
    return destination


def test_main_prints_nothing_and_returns_0_for_a_conforming_bundle(
    full_bundle, capsys
):
    assert main([str(full_bundle), "--full"]) == 0
    assert capsys.readouterr().out == ""


def test_main_prints_one_line_for_each_finding_and_returns_1(bundle, capsys):
    document = json.loads((bundle / "recording.json").read_text())
    document["reference_time"] = "yesterday"
    document["session"] = {"start_time": "later"}
    (bundle / "recording.json").write_text(json.dumps(document))
    assert main([str(bundle)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("recording.json: section 6.6: reference_time")
    assert "session.start_time" in lines[1]


def test_main_names_the_root_group_with_a_dot(bundle, capsys):
    root = json.loads((bundle / "zarr.json").read_text())
    root["attributes"]["tsz"]["version"] = "2.0"
    (bundle / "zarr.json").write_text(json.dumps(root))
    assert main([str(bundle)]) == 1
    assert capsys.readouterr().out.startswith(".: section 9: ")


def test_main_checks_statistics_only_with_full(bundle, capsys):
    zarr.open_array(bundle / "0" / "1" / "mean", mode="r+")[0] = np.float32(1)
    assert main([str(bundle)]) == 0
    assert main([str(bundle), "--full"]) == 1
    assert capsys.readouterr().out.startswith("0/1/mean: section 8.4: ")


def test_the_module_runs_without_a_warning(full_bundle):
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            "-m",
            "timeseries_zarr.validate",
            str(full_bundle),
            "--full",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
