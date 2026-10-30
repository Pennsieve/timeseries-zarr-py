"""Resolve the stored unit and scale factor of a channel."""

from timeseries_zarr.constants import UNIT_TO_UV
from timeseries_zarr.types import ResolvedUnit, VoltagePolicy

MICROVOLT_UNIT = "uV"
"""Unit string the format fixes for a voltage channel."""


def resolve_unit(unit: str, policy: VoltagePolicy) -> ResolvedUnit:
    """Return the factor and stored unit for a channel that declares unit.

    A unit is a voltage unit when it is exactly a symbol key of UNIT_TO_UV
    (V, mV, uV, nV) or when its lowercase form is a spelled-out key (volts,
    millivolts, microvolts, nanovolts). Under convert a voltage unit resolves
    to its factor and "uV". Under require_uv it must be exactly "uV" and
    resolves to factor 1.0. Any other unit, including "MV", resolves to
    factor 1.0 and itself.

    Args:
        unit: Unit string the channel declares.
        policy: How to treat a voltage unit.

    Returns:
        The factor that scales samples to the stored unit, and the stored unit.

    Raises:
        ValueError: The policy is require_uv and unit is a voltage unit
            spelled other than "uV".
    """
    factor = UNIT_TO_UV.get(unit)
    if factor is None:
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
