"""The bundle's recording.json: identity and wall-clock time in one plain file."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from timeseries_zarr.types import RecordingMeta

RECORDING_FILE = "recording.json"
"""Name of the plain JSON file at the root of a bundle."""

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _require_offset(instant: datetime, name: str) -> None:
    """Raise ValueError if instant, called name, has no UTC offset."""
    if instant.utcoffset() is None:
        raise ValueError(f"{name} must carry a UTC offset")


def epoch_us(instant: datetime) -> int:
    """Return instant in microseconds since the Unix epoch.

    The arithmetic is exact in integers. Raises ValueError if instant has no
    UTC offset.
    """
    _require_offset(instant, "instant")
    return (instant - _EPOCH) // timedelta(microseconds=1)


def recording_document(
    meta: RecordingMeta, recording_start_us: int | None
) -> dict[str, object]:
    """Return the recording.json content of a bundle.

    start_time is recording_start_us, the recording start in wall-clock
    microseconds since the Unix epoch, as an RFC 3339 string with microsecond
    precision. It takes the UTC offset of meta.session_start_time, or UTC
    without one. start_time is left out when recording_start_us is None, a
    bundle with no sample and no event, or when meta.wall_clock is False.
    session is meta.session with meta.session_start_time first, as given,
    under start_time. subject, session, and source follow start_time in that
    order, each only when it holds something. Raises ValueError if
    meta.session_start_time has no UTC offset or meta.session already holds a
    start_time key.
    """
    session_start = meta.session_start_time
    if session_start is not None:
        _require_offset(session_start, "session_start_time")
    if "start_time" in meta.session:
        raise ValueError(
            "session holds a start_time key; set session_start_time instead"
        )
    document: dict[str, object] = {}
    if recording_start_us is not None and meta.wall_clock:
        zone = UTC if session_start is None else session_start.tzinfo
        start = (
            _EPOCH + timedelta(microseconds=recording_start_us)
        ).astimezone(zone)
        document["start_time"] = start.isoformat(timespec="microseconds")
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


def write_recording_json(bundle_dir: Path, document: dict[str, object]) -> None:
    """Write document as recording.json at the root of bundle_dir.

    The file is UTF-8 JSON indented by one space and ends with one newline, so
    an editor or a text tool can change it. It is not a Zarr node and never
    enters the consolidated metadata. An existing file is overwritten.
    bundle_dir must exist. Raises TypeError if a value in document is not JSON
    serializable.
    """
    text = json.dumps(document, indent=1, ensure_ascii=False)
    (bundle_dir / RECORDING_FILE).write_text(text + "\n", encoding="utf-8")
