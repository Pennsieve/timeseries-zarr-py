import pytest

from timeseries_zarr.attrs import (
    channel_group_attrs,
    level_group_attrs,
    root_attrs,
    waveform_array_attrs,
)


def test_root_attrs_returns_empty():
    assert root_attrs() == {}


@pytest.mark.parametrize("kind", ["continuous", "unit"])
def test_channel_group_attrs(kind):
    attrs = channel_group_attrs(
        "N:channel:abc", 32000.0, 1000, kind, "channel one", "uV"
    )
    assert attrs == {
        "id": "N:channel:abc",
        "rate_hz": 32000.0,
        "offset_us": 1000,
        "kind": kind,
        "name": "channel one",
        "unit": "uV",
    }


@pytest.mark.parametrize("period_us", [31.25, 125.0, 512000.0])
def test_level_group_attrs(period_us):
    assert level_group_attrs(period_us) == {"period_us": period_us}


@pytest.mark.parametrize("period_us", [31.25, 125.0, 512000.0])
def test_waveform_array_attrs(period_us):
    assert waveform_array_attrs(period_us) == {"period_us": period_us}
