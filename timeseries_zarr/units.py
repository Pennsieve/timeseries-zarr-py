"""Resolve the stored unit and scale factor of a continuous channel."""

from timeseries_zarr.constants import UNIT_TO_UV
from timeseries_zarr.types import ResolvedUnit, VoltagePolicy

MICROVOLT_UNIT = "uV"
"""Unit string the format fixes for a voltage channel."""


def resolve_unit(unit: str, policy: VoltagePolicy) -> ResolvedUnit:
    """Return the factor and stored unit for a channel that declares unit.

    A unit whose lowercase form is a key of UNIT_TO_UV is a voltage unit.
    Under convert it resolves to that key's factor and "uV". Under require_uv
    it must be exactly "uV" and resolves to factor 1.0. Any other unit
    resolves to factor 1.0 and itself. Raises ValueError under require_uv
    for a voltage unit spelled other than "uV".
    """
    factor = UNIT_TO_UV.get(unit.lower())
    if factor is None:
        return ResolvedUnit(1.0, unit)
    if policy == "require_uv":
        if unit != MICROVOLT_UNIT:
            raise ValueError(
                f"unit {unit!r} is a voltage unit: a voltage channel must "
                f"declare {MICROVOLT_UNIT!r} and hold microvolts"
            )
        return ResolvedUnit(1.0, unit)
    return ResolvedUnit(factor, MICROVOLT_UNIT)
