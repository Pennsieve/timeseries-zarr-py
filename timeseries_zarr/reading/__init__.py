"""Reading layer: open, query, montage, and filter TS-Zarr bundles.

Everything here is convenience over stock Zarr -- a bundle needs no
library to be read. What this package owns is the parts that are easy
to get wrong: level selection, the rho serving tiers, exact montage
from mean levels, filter redesign with pole-decay pre-roll, and event
queries bounded by ``max_duration_us``.
"""

from timeseries_zarr.reading.bundle import (
    Bundle,
    ContinuousChannel,
    EventChannel,
    open_bundle,
)
from timeseries_zarr.reading.serving import (
    FilterSpec,
    ServingDecision,
    ServingError,
)

__all__ = [
    "Bundle",
    "ContinuousChannel",
    "EventChannel",
    "FilterSpec",
    "ServingDecision",
    "ServingError",
    "open_bundle",
]
