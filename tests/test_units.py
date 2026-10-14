import pytest

from timeseries_zarr.constants import UNIT_TO_UV
from timeseries_zarr.types import ResolvedUnit
from timeseries_zarr.units import resolve_unit


@pytest.mark.parametrize("unit", ["volts", "V", "mV", "Millivolts", "uv", "uV"])
def test_convert_maps_every_voltage_spelling_to_microvolts(unit):
    resolved = resolve_unit(unit, "convert")
    assert resolved == ResolvedUnit(UNIT_TO_UV[unit.lower()], "uV")


@pytest.mark.parametrize("unit", ["pA", "", "degC"])
def test_convert_passes_an_unknown_unit_through(unit):
    assert resolve_unit(unit, "convert") == ResolvedUnit(1.0, unit)


def test_require_accepts_microvolts_spelled_uv():
    assert resolve_unit("uV", "require_uv") == ResolvedUnit(1.0, "uV")


@pytest.mark.parametrize("unit", ["volts", "mV", "uv", "microvolts"])
def test_require_rejects_a_voltage_unit_not_spelled_uv(unit):
    with pytest.raises(ValueError, match="microvolts") as caught:
        resolve_unit(unit, "require_uv")
    assert repr(unit) in str(caught.value)
    assert "'uV'" in str(caught.value)


@pytest.mark.parametrize("unit", ["pA", "", "degC"])
def test_require_passes_an_unknown_unit_through(unit):
    assert resolve_unit(unit, "require_uv") == ResolvedUnit(1.0, unit)
