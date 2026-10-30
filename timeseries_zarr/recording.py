"""The bundle's recording.json, which holds identity and datetimes."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from timeseries_zarr.types import RecordingMeta

RECORDING_FILE = "recording.json"
"""Name of the plain JSON file at the root of a bundle."""

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _require_offset(instant: datetime, name: str) -> None:
    """Raise ValueError if instant, called name, lacks a valid UTC offset.

    RFC 3339 writes an offset as hours and minutes, so the offset must be a
    whole number of minutes.
    """
    offset = instant.utcoffset()
    if offset is None:
        raise ValueError(f"{name} must carry a UTC offset")
    if offset % timedelta(minutes=1):
        raise ValueError(
            f"{name} must carry a UTC offset of whole minutes, got {offset}"
        )


def epoch_us(instant: datetime) -> int:
    """Return instant in microseconds since the Unix epoch.

    The arithmetic is exact in integers.

    Args:
        instant: A datetime that carries a UTC offset.

    Returns:
        Whole microseconds since 1970-01-01T00:00:00Z.

    Raises:
        ValueError: If instant has no UTC offset or an offset that is not a
            whole number of minutes.
    """
    _require_offset(instant, "instant")
    return (instant - _EPOCH) // timedelta(microseconds=1)


def session_start_us(meta: RecordingMeta) -> int | None:
    """Return the session start in microseconds since the Unix epoch.

    Args:
        meta: The recording metadata.

    Returns:
        The microseconds, or None when meta.session_start_time is None.

    Raises:
        ValueError: If meta.session_start_time has no UTC offset or an offset
            that is not a whole number of minutes.
    """
    if meta.session_start_time is None:
        return None
    _require_offset(meta.session_start_time, "session_start_time")
    return epoch_us(meta.session_start_time)


def _require_wall_clock(meta: RecordingMeta) -> None:
    """Raise ValueError if meta sets reference_time with wall_clock False."""
    if meta.reference_time is not None and not meta.wall_clock:
        raise ValueError(
            "reference_time needs wall_clock=True; leave it unset when the "
            "source times do not count from the Unix epoch"
        )


def reference_time_us(meta: RecordingMeta) -> int | None:
    """Return the reference time in microseconds since the Unix epoch.

    Args:
        meta: The recording metadata.

    Returns:
        The microseconds, or None when meta.reference_time is None.

    Raises:
        ValueError: If meta.reference_time has no UTC offset, or if it is set
            while meta.wall_clock is False.
    """
    if meta.reference_time is None:
        return None
    _require_wall_clock(meta)
    _require_offset(meta.reference_time, "reference_time")
    return epoch_us(meta.reference_time)


def recording_document(
    meta: RecordingMeta, reference_us: int | None
) -> dict[str, object]:
    """Return the recording.json content of a bundle.

    The reference_time entry is an RFC 3339 string with microsecond precision.
    It takes the UTC offset of meta.reference_time, else of
    meta.session_start_time, else UTC. The session entry is meta.session with
    meta.session_start_time first, as given, under start_time. The entries
    subject, session, and source follow reference_time in that order, each
    only when it holds something.

    Args:
        meta: The recording metadata.
        reference_us: The reference time in microseconds since the Unix
            epoch. None means a bundle with no sample and no event.

    Returns:
        The document. It has no reference_time entry when reference_us is
        None or meta.wall_clock is False.

    Raises:
        ValueError: If a time of meta has no UTC offset, if
            meta.reference_time is set while meta.wall_clock is False, or if
            meta.session already holds a start_time key.
    """
    _require_wall_clock(meta)
    session_start = meta.session_start_time
    if session_start is not None:
        _require_offset(session_start, "session_start_time")
    if meta.reference_time is not None:
        _require_offset(meta.reference_time, "reference_time")
    if "start_time" in meta.session:
        raise ValueError(
            "session holds a start_time key; set session_start_time instead"
        )
    document: dict[str, object] = {}
    if reference_us is not None and meta.wall_clock:
        zoned = meta.reference_time or session_start
        zone = UTC if zoned is None else zoned.tzinfo
        reference = (_EPOCH + timedelta(microseconds=reference_us)).astimezone(
            zone
        )
        document["reference_time"] = reference.isoformat(
            timespec="microseconds"
        )
    session: dict[str, object] = {}
    if session_start is not None:
        session["start_time"] = session_start.isoformat(timespec="microseconds")
    session.update(meta.session)
    for key, value in (
        ("subject", dict(meta.subject)),
        ("session", session),
        ("source", dict(meta.source)),
    ):
        if value:
            document[key] = value
    return document


def _dumps(document: dict[str, object]) -> str:
    """Return document as JSON text indented by one space.

    Raises:
        TypeError: If a value in document is not JSON serializable.
        ValueError: If a number in document is NaN or infinite.
    """
    try:
        return json.dumps(
            document, indent=1, ensure_ascii=False, allow_nan=False
        )
    except ValueError as error:
        raise ValueError(
            f"{RECORDING_FILE} must hold only finite numbers: {error}"
        ) from error


def check_recording_document(document: dict[str, object]) -> None:
    """Raise ValueError unless document can be written as recording.json.

    Args:
        document: The content of recording.json.

    Raises:
        ValueError: If document holds a value JSON cannot store, or a NaN or
            infinite number, which is not a JSON value.
    """
    try:
        _dumps(document)
    except TypeError as error:
        raise ValueError(
            f"{RECORDING_FILE} must hold only JSON values: {error}"
        ) from error


def write_recording_json(bundle_dir: Path, document: dict[str, object]) -> None:
    """Write document as recording.json at the root of bundle_dir.

    The file is UTF-8 JSON indented by one space and ends with one newline, so
    an editor or a text tool can change it. It is not a Zarr node and never
    enters the consolidated metadata. An existing file is overwritten.

    Args:
        bundle_dir: An existing directory.
        document: The content to write.

    Raises:
        TypeError: If a value in document is not JSON serializable.
        ValueError: If a number in document is NaN or infinite.
    """
    text = _dumps(document)
    (bundle_dir / RECORDING_FILE).write_text(text + "\n", encoding="utf-8")
