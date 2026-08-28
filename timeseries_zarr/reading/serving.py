"""The serving rules: which level may serve which view, and how loudly.

Implements the rho rule of the TS-Zarr paper (S4): rho is the highest
filter transition frequency divided by the serving level's Nyquist.
Tiers: rho <= 0.15 serve silently; 0.15 < rho <= 0.5 serve with a
fidelity indicator; rho > 0.5 refuse (select a finer level, ultimately
raw). Thresholds are halved for broadband (flat-spectrum) signals and
quartered for low-edge-dominated filters (pure highpass), per the
measured ~4x sensitivity of low edges.
"""

from dataclasses import dataclass
from enum import Enum

SILENT_RHO = 0.15
MARKED_RHO = 0.5
HIGHPASS_FACTOR = 0.25
BROADBAND_FACTOR = 0.5

#: Default zero-input decay target for filter pre-roll (Theorem: warm-up).
PREROLL_EPSILON = 0.01

#: Fetch-free fallback pad at a data boundary: tiled reflection of a
#: block of roughly this many seconds (the best-measured fallback).
REFLECTION_BLOCK_S = 0.05


class Tier(Enum):
    """How a level may serve a filtered view."""

    SILENT = "silent"
    MARKED = "marked"
    REFUSE = "refuse"


class ServingError(RuntimeError):
    """A view was requested that no stored level may serve."""


@dataclass(frozen=True)
class FilterSpec:
    """A filter request: band edges in Hz; ``None`` edge = one-sided.

    ``(None, 70)`` is a lowpass, ``(0.5, None)`` a highpass,
    ``(0.5, 70)`` a bandpass. ``order`` is the Butterworth order used
    by the reference implementation (causal IIR redesigned at the
    serving rate -- the pessimistic implementation the paper's tier
    measurements assume).
    """

    low_hz: float | None
    high_hz: float | None
    order: int = 4

    def __post_init__(self) -> None:
        """Validate that at least one edge is present."""
        if self.low_hz is None and self.high_hz is None:
            raise ValueError("filter needs at least one band edge")

    @property
    def btype(self) -> str:
        """Return the scipy band type string."""
        if self.low_hz is None:
            return "lowpass"
        if self.high_hz is None:
            return "highpass"
        return "bandpass"

    @property
    def transition_hz(self) -> float:
        """Highest transition frequency: the quantity rho is built on."""
        if self.high_hz is not None:
            return float(self.high_hz)
        assert self.low_hz is not None
        return float(self.low_hz)

    @property
    def is_pure_highpass(self) -> bool:
        """True when the response is governed by a low edge alone."""
        return self.high_hz is None


@dataclass(frozen=True)
class ServingDecision:
    """The outcome of the rho rule for one (view, level) pairing."""

    level: int
    rate_hz: float
    rho: float
    tier: Tier

    @property
    def marked(self) -> bool:
        """True when the client must show a fidelity indicator."""
        return self.tier is Tier.MARKED


def rho_for(spec: FilterSpec, level_rate_hz: float) -> float:
    """Return rho = highest transition frequency / level Nyquist."""
    return spec.transition_hz / (level_rate_hz / 2.0)


def classify(
    spec: FilterSpec,
    level_rate_hz: float,
    *,
    broadband: bool = False,
) -> tuple[float, Tier]:
    """Classify one level against one filter request.

    Returns ``(rho, tier)``. ``broadband=True`` halves the thresholds
    (flat-spectrum signals); a pure highpass quarters them.
    """
    factor = 1.0
    if spec.is_pure_highpass:
        factor *= HIGHPASS_FACTOR
    if broadband:
        factor *= BROADBAND_FACTOR
    rho = rho_for(spec, level_rate_hz)
    if rho <= SILENT_RHO * factor:
        return rho, Tier.SILENT
    if rho <= MARKED_RHO * factor:
        return rho, Tier.MARKED
    return rho, Tier.REFUSE
