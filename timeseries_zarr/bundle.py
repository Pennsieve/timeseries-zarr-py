"""Top-level bundle orchestration."""

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from timeseries_zarr.attrs import root_attrs
from timeseries_zarr.constants import TSZ_KEY, ZARR_METADATA_FILE
from timeseries_zarr.protocols import (
    ContinuousChannelSource,
    EventChannelSource,
)
from timeseries_zarr.recording import (
    recording_document,
    session_start_us,
    write_recording_json,
)
from timeseries_zarr.types import RecordingMeta, WriteOpts
from timeseries_zarr.units import resolve_unit
from timeseries_zarr.write_continuous import write_continuous_channel
from timeseries_zarr.write_event import write_event_channel
from timeseries_zarr.zarr_io import ZarrGroup, consolidate, open_group

type AnySource = ContinuousChannelSource | EventChannelSource
"""Any channel source the bundle writes, whichever protocol it follows."""


def assign_indices(
    continuous: Sequence[ContinuousChannelSource],
    events: Sequence[EventChannelSource],
) -> list[tuple[int, AnySource]]:
    """Assign each channel a digit index, continuous first then event.

    Indices are contiguous from 0 and become the bundle's digit-named
    channel-group directories. Sources keep their input order within each kind.
    """
    return list(enumerate([*continuous, *events]))


def origin_us(
    sources: Sequence[AnySource],
    session_start_us: int | None = None,
) -> int:
    """Return the bundle origin: the earliest wall-clock instant it holds.

    Every channel and event is placed in microseconds from this instant, so it
    is the minimum of the session start, when given, and the start of every
    source. No offset or event time is then negative. The origin is the one
    wall-clock value the writer maps to the time zero of the bundle. A bundle
    with neither a session start nor a channel has no instant to count from and
    reports 0.
    """
    starts = [source.start_us() for source in sources]
    if session_start_us is not None:
        starts.append(session_start_us)
    return min(starts, default=0)


def channel_keys(indexed: Sequence[tuple[int, AnySource]]) -> dict[str, int]:
    """Return each channel's id mapped to its channel key.

    The key is the index assign_indices gave the channel, which names its group
    in the bundle. An event channel names the channels it applies to by id, so
    this is the table a channels column resolves through. Raises ValueError if
    two channels share an id.
    """
    keys: dict[str, int] = {}
    for index, source in indexed:
        if source.id in keys:
            raise ValueError(
                f"channels {keys[source.id]} and {index} share the id "
                f"{source.id!r}"
            )
        keys[source.id] = index
    return keys


def max_key(indexed: Sequence[tuple[int, AnySource]]) -> int:
    """Return the largest channel key, which sizes the channels bitmask.

    A bundle with no channels reports 0.
    """
    return max((index for index, _ in indexed), default=0)


def clear_output_path(output_path: Path) -> None:
    """Empty output_path of a bundle an earlier run left there.

    Does nothing when output_path is absent or empty. Keeps the directory
    itself and removes its root zarr.json last, so an interrupted clear still
    leaves a bundle the next run recognizes. Raises ValueError, and removes
    nothing, if output_path is not a directory or holds files but no bundle.
    """
    if not output_path.exists():
        return
    if not output_path.is_dir():
        raise ValueError(
            f"output_path {output_path} is not a directory; "
            "use an absent path, an empty directory, or an existing bundle"
        )
    entries = list(output_path.iterdir())
    if not entries:
        return
    root = output_path / ZARR_METADATA_FILE
    if not _is_bundle_root(root):
        raise ValueError(
            f"output_path {output_path} holds files but no bundle; "
            "use an absent path, an empty directory, or an existing bundle"
        )
    for entry in entries:
        if entry == root:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    root.unlink()


def _is_bundle_root(path: Path) -> bool:
    """Return whether path is a root zarr.json that carries tsz.version."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(document, dict):
        return False
    attributes = document.get("attributes")
    if not isinstance(attributes, dict):
        return False
    tsz = attributes.get(TSZ_KEY)
    return isinstance(tsz, dict) and isinstance(tsz.get("version"), str)


def write_all_channels(
    root: ZarrGroup,
    indexed: Sequence[tuple[int, AnySource]],
    origin_us: int,
    channel_keys: Mapping[str, int],
    opts: WriteOpts,
) -> None:
    """Write every indexed channel under root, dispatching by source type.

    Each (index, source) pair becomes its own channel group beneath root,
    governed by opts and placed relative to origin_us. channel_keys maps
    channel ids to keys, which the event channels use to resolve the channels
    they apply to.
    """
    largest_key = max_key(indexed)
    for index, source in indexed:
        if isinstance(source, EventChannelSource):
            write_event_channel(
                root,
                index,
                source,
                origin_us=origin_us,
                channel_keys=channel_keys,
                max_key=largest_key,
                opts=opts,
            )
        else:
            write_continuous_channel(
                root, index, source, origin_us=origin_us, opts=opts
            )


def write_bundle(
    continuous: Sequence[ContinuousChannelSource],
    events: Sequence[EventChannelSource],
    *,
    output_path: Path,
    opts: WriteOpts,
    recording: RecordingMeta | None = None,
) -> None:
    """Write the whole viewer bundle at output_path.

    The root group carries the format version. Every channel is written into
    a fresh root group at output_path, the metadata of every node is
    consolidated into the root, and then recording.json is written when
    recording is given, so identity and the wall-clock start never enter the
    root object. A bundle an earlier run left at output_path is replaced.
    Every time counts from the origin, the earliest of the session start in
    recording and every channel start. recording.json holds that instant as
    its start when recording has a start. Sizing and compression follow opts. Raises ValueError if
    two channels share an id, recording.start has no UTC offset, a voltage
    unit is not "uV" under voltage_policy require_uv, or output_path is not
    absent, empty, or a bundle, before anything is written. A failed write
    leaves a partial bundle with no consolidated metadata.
    """
    indexed = assign_indices(continuous, events)
    session_start = None if recording is None else session_start_us(recording)
    origin = origin_us([source for _, source in indexed], session_start)
    keys = channel_keys(indexed)
    for index, source in indexed:
        if not isinstance(source, EventChannelSource):
            try:
                resolve_unit(source.unit, opts.voltage_policy)
            except ValueError as error:
                raise ValueError(
                    f"channel {index} ({source.id!r}): {error}"
                ) from error
    clear_output_path(output_path)
    root = open_group(output_path, root_attrs())
    write_all_channels(root, indexed, origin, keys, opts)
    consolidate(root)
    if recording is not None:
        write_recording_json(output_path, recording_document(recording, origin))
