import pytest

import timeseries_zarr

PUBLIC_NAMES = {
    "UNIT_TO_UV",
    "ContinuousChannelSource",
    "EventChannelSource",
    "EventColumn",
    "RecordingMeta",
    "VoltagePolicy",
    "WriteOpts",
    "__version__",
    "write_bundle",
}


def test_all_lists_exactly_the_public_names() -> None:
    assert set(timeseries_zarr.__all__) == PUBLIC_NAMES


@pytest.mark.parametrize("name", sorted(PUBLIC_NAMES))
def test_every_public_name_resolves(name: str) -> None:
    assert hasattr(timeseries_zarr, name)


def test_version_is_a_non_empty_string() -> None:
    assert isinstance(timeseries_zarr.__version__, str)
    assert timeseries_zarr.__version__
