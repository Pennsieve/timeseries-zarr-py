"""Read and write TS-Zarr bundles of neurophysiology time series."""

from timeseries_zarr.bundle import write_bundle
from timeseries_zarr.columns import (
    decode_channels_rows,
    decode_text_rows,
    text_width,
)
from timeseries_zarr.constants import MAX_LABELS, MAX_LEVELS, UNIT_TO_UV
from timeseries_zarr.protocols import (
    ContinuousChannelSource,
    EventChannelSource,
)
from timeseries_zarr.read_bundle import Bundle, Channel, open_bundle
from timeseries_zarr.read_continuous import (
    Bins,
    ContinuousChannel,
    Level,
    Samples,
)
from timeseries_zarr.read_event import (
    CountLevel,
    Counts,
    EventChannel,
    EventTable,
    Snippets,
    Waveform,
)
from timeseries_zarr.read_recording import Recording
from timeseries_zarr.sources import bundle_sources, recording_meta
from timeseries_zarr.stack import BinMember, Stack
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
    "MAX_LABELS",
    "MAX_LEVELS",
    "UNIT_TO_UV",
    "BinMember",
    "Bins",
    "Bundle",
    "Channel",
    "ContinuousChannel",
    "ContinuousChannelSource",
    "CountLevel",
    "Counts",
    "EventChannel",
    "EventChannelSource",
    "EventColumn",
    "EventTable",
    "Level",
    "Recording",
    "RecordingMeta",
    "Samples",
    "Snippets",
    "Stack",
    "VoltagePolicy",
    "Waveform",
    "WriteOpts",
    "__version__",
    "bundle_sources",
    "decode_channels_rows",
    "decode_text_rows",
    "open_bundle",
    "recording_meta",
    "text_width",
    "write_bundle",
]
