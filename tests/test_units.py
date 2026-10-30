import pytest

from timeseries_zarr.constants import UNIT_TO_UV
from timeseries_zarr.types import ResolvedUnit
from timeseries_zarr.units import resolve_unit


@pytest.mark.parametrize(
    ("unit", "factor"),
    [
        ("V", 1e6),
        ("mV", 1e3),
        ("uV", 1.0),
        ("nV", 1e-3),
        ("volts", 1e6),
        ("Volts", 1e6),
        ("Millivolts", 1e3),
        ("MICROVOLTS", 1.0),
        ("nanovolts", 1e-3),
    ],
)
def test_convert_maps_every_voltage_spelling_to_microvolts(unit, factor):
    assert resolve_unit(unit, "convert") == ResolvedUnit(factor, "uV")
    assert factor == UNIT_TO_UV.get(unit, UNIT_TO_UV.get(unit.lower()))


@pytest.mark.parametrize("unit", ["MV", "Mv", "v", "mv", "uv", "NV", "Nv"])
def test_convert_does_not_match_a_symbol_in_another_case(unit):
    assert resolve_unit(unit, "convert") == ResolvedUnit(1.0, unit)
    assert resolve_unit(unit, "require_uv") == ResolvedUnit(1.0, unit)


@pytest.mark.parametrize("unit", ["pA", "", "degC"])
def test_convert_passes_an_unknown_unit_through(unit):
    assert resolve_unit(unit, "convert") == ResolvedUnit(1.0, unit)


def test_require_accepts_microvolts_spelled_uv():
    assert resolve_unit("uV", "require_uv") == ResolvedUnit(1.0, "uV")


@pytest.mark.parametrize("unit", ["volts", "V", "mV", "nV", "microvolts"])
def test_require_rejects_a_voltage_unit_not_spelled_uv(unit):
    with pytest.raises(ValueError, match="microvolts") as caught:
        resolve_unit(unit, "require_uv")
    assert repr(unit) in str(caught.value)
    assert "'uV'" in str(caught.value)


@pytest.mark.parametrize("unit", ["pA", "", "degC"])
def test_require_passes_an_unknown_unit_through(unit):
    assert resolve_unit(unit, "require_uv") == ResolvedUnit(1.0, unit)
