import json
import re
import sys

import pytest

from timeseries_zarr.extras import missing_extra_message, require

_PANDAS_MESSAGE = (
    "pandas is not installed. Install the pandas extra: "
    "pip install 'timeseries-zarr[pandas]'"
)


def test_missing_extra_message_names_the_package_and_the_pip_command():
    assert missing_extra_message("pandas", "pandas") == _PANDAS_MESSAGE


def test_missing_extra_message_names_the_extra_apart_from_the_package():
    assert missing_extra_message("fsspec", "remote") == (
        "fsspec is not installed. Install the remote extra: "
        "pip install 'timeseries-zarr[remote]'"
    )


def test_require_returns_the_imported_module():
    assert require("json", "json") is json


def test_require_raises_import_error_named_for_an_absent_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match=re.escape(_PANDAS_MESSAGE)) as info:
        require("pandas", "pandas")
    assert info.value.name == "pandas"
    assert isinstance(info.value.__cause__, ModuleNotFoundError)


def test_require_passes_on_the_error_of_a_module_the_package_imports(
    tmp_path, monkeypatch
):
    (tmp_path / "tsz_needs_absent.py").write_text("import tsz_absent\n")
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.setitem(sys.modules, "tsz_absent", None)
    with pytest.raises(ModuleNotFoundError) as info:
        require("tsz_needs_absent", "extra")
    assert info.value.name == "tsz_absent"
    assert "Install the" not in str(info.value)
