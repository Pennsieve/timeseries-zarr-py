"""The bundle's recording.json: identity and wall-clock time in one plain file."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from timeseries_zarr.types import RecordingMeta

RECORDING_FILE = "recording.json"
"""Name of the plain JSON file at the root of a bundle."""

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _require_offset(start: datetime) -> None:
    """Raise ValueError if start has no UTC offset."""
    if start.utcoffset() is None:
        raise ValueError("session_start_time must carry a UTC offset")


def session_start_us(meta: RecordingMeta) -> int | None:
    """Return the session start in microseconds since the Unix epoch.

    The arithmetic is exact in integers. A meta without a session start returns
    None. Raises ValueError if meta.session_start_time has no UTC offset.
    """
    start = meta.session_start_time
    if start is None:
        return None
    _require_offset(start)
    return (start - _EPOCH) // timedelta(microseconds=1)


def recording_document(
    meta: RecordingMeta, origin_us: int
) -> dict[str, object]:
    """Return the recording.json content for a bundle with the given origin.

    session_start_time is meta.session_start_time as given.
    timestamps_reference_time is origin_us, the instant every time in the
    bundle counts from, in the UTC offset of the session start. Both are RFC
    3339 strings with microsecond precision. origin_us is wall-clock
    microseconds since the Unix epoch. A meta without a session start gives a
    document with neither time, and origin_us is then unused.
    subject, session, and source follow in that order, each only when it holds
    something. Raises ValueError if meta.session_start_time has no UTC offset.
    """
    document: dict[str, object] = {}
    start = meta.session_start_time
    if start is not None:
        _require_offset(start)
        origin = (_EPOCH + timedelta(microseconds=origin_us)).astimezone(
            start.tzinfo
        )
        document["session_start_time"] = start.isoformat(
            timespec="microseconds"
        )
        document["timestamps_reference_time"] = origin.isoformat(
            timespec="microseconds"
        )
    for key, value in (
        ("subject", meta.subject),
        ("session", meta.session),
        ("source", meta.source),
    ):
        if value:
            document[key] = dict(value)
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
