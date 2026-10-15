"""Producer for pyramid Zarr v3 viewer bundles from neurophysiology time series."""

from timeseries_zarr.bundle import write_bundle
from timeseries_zarr.constants import UNIT_TO_UV
from timeseries_zarr.protocols import (
    ContinuousChannelSource,
    EventChannelSource,
)
from timeseries_zarr.types import (
    EventColumn,
    RecordingMeta,
    VoltagePolicy,
    WriteOpts,
)

try:
    from timeseries_zarr._version import version as __version__
except ImportError:
    __version__ = "0+unknown"

__all__ = [
    "UNIT_TO_UV",
    "ContinuousChannelSource",
    "EventChannelSource",
    "EventColumn",
    "RecordingMeta",
    "VoltagePolicy",
    "WriteOpts",
    "__version__",
    "write_bundle",
]
