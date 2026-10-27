import subprocess
import sys

import pytest

import timeseries_zarr

PUBLIC_NAMES = {
    "MAX_LABELS",
    "MAX_LEVELS",
    "UNIT_TO_UV",
    "BinMember",
    "Bins",
    "Bundle",
    "Channel",
    "ContinuousChannel",
    "ContinuousChannelMetadata",
    "ContinuousChannelSource",
    "CountLevel",
    "Counts",
    "EventChannel",
    "EventChannelSource",
    "EventColumn",
    "EventTable",
    "Level",
    "MultiChannelSource",
    "Recording",
    "RecordingMeta",
    "Samples",
    "Snippets",
    "Stack",
    "VoltagePolicy",
    "Waveform",
    "WriteOpts",
    "__version__",
    "bundle_sources",
    "channel_writer_bytes",
    "decode_channels_rows",
    "decode_text_rows",
    "open_bundle",
    "recording_meta",
    "text_width",
    "write_bundle",
}

EXTRA_MODULES = ("pandas", "xarray", "fsspec")


def _run_isolated(code: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_all_lists_exactly_the_public_names() -> None:
    assert set(timeseries_zarr.__all__) == PUBLIC_NAMES


@pytest.mark.parametrize("name", sorted(PUBLIC_NAMES))
def test_every_public_name_resolves(name: str) -> None:
    assert hasattr(timeseries_zarr, name)


def test_version_is_a_non_empty_string() -> None:
    assert isinstance(timeseries_zarr.__version__, str)
    assert timeseries_zarr.__version__


def test_import_adds_no_pandas_xarray_or_fsspec_beyond_zarr() -> None:
    # zarr imports fsspec when it is installed.
    code = (
        "import sys\n"
        "import zarr\n"
        "for name in list(sys.modules):\n"
        "    if name.partition('.')[0] == 'fsspec':\n"
        "        del sys.modules[name]\n"
        "before = set(sys.modules)\n"
        "import timeseries_zarr\n"
        "added = {name.partition('.')[0] for name in set(sys.modules) - before}\n"
        f"print(sorted(added & set({EXTRA_MODULES!r})))\n"
    )
    assert _run_isolated(code) == "[]\n"


def test_the_package_imports_without_pandas_xarray_or_fsspec() -> None:
    code = (
        "import sys\n"
        f"for name in {EXTRA_MODULES!r}:\n"
        "    sys.modules[name] = None\n"
        "import timeseries_zarr\n"
        "names = timeseries_zarr.__all__\n"
        "print([name for name in names if not hasattr(timeseries_zarr, name)])\n"
    )
    assert _run_isolated(code) == "[]\n"
