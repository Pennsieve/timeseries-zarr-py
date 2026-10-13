import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from timeseries_zarr.recording import (
    RECORDING_FILE,
    recording_document,
    write_recording_json,
)
from timeseries_zarr.types import RecordingMeta

_NOON_UTC_US = 1_710_407_700 * 1_000_000
"""2024-03-14T09:15:00 UTC in microseconds since the epoch."""


def _meta(start=None, subject=None, session=None, source=None):
    return RecordingMeta(
        start=start or datetime(2024, 3, 14, 9, 15, tzinfo=UTC),
        subject={} if subject is None else subject,
        session={} if session is None else session,
        source={} if source is None else source,
    )


def test_recording_document_start_is_the_origin_in_utc():
    doc = recording_document(_meta(), _NOON_UTC_US)
    assert doc["start"] == "2024-03-14T09:15:00.000000+00:00"


def test_recording_document_start_keeps_the_offset_of_the_session_start():
    start = datetime(2024, 3, 14, 4, 15, tzinfo=timezone(timedelta(hours=-5)))
    doc = recording_document(_meta(start=start), _NOON_UTC_US)
    assert doc["start"] == "2024-03-14T04:15:00.000000-05:00"


def test_recording_document_start_is_exact_to_the_microsecond():
    origin_us = 1_700_000_000_123_457
    doc = recording_document(_meta(), origin_us)
    assert doc["start"] == "2023-11-14T22:13:20.123457+00:00"


def test_recording_document_start_comes_from_the_origin_not_the_session_start():
    earlier = _NOON_UTC_US - 90 * 1_000_000
    doc = recording_document(_meta(), earlier)
    assert doc["start"] == "2024-03-14T09:13:30.000000+00:00"


def test_recording_document_start_parses_back_to_the_origin_instant():
    doc = recording_document(_meta(), _NOON_UTC_US + 5)
    parsed = datetime.fromisoformat(doc["start"])
    assert parsed.utcoffset() is not None
    assert parsed == datetime(2024, 3, 14, 9, 15, 0, 5, tzinfo=UTC)


def test_recording_document_rejects_a_start_without_a_utc_offset():
    with pytest.raises(ValueError, match="UTC offset"):
        recording_document(_meta(start=datetime(2024, 3, 14, 9, 15)), 0)


def test_recording_document_omits_objects_that_are_empty():
    assert list(recording_document(_meta(), 0)) == ["start"]


def test_recording_document_copies_the_objects_in_a_fixed_order():
    meta = _meta(
        subject={"id": "sub-0931", "sex": "F"},
        session={"id": "ses-01"},
        source={"system": "Natus"},
    )
    doc = recording_document(meta, _NOON_UTC_US)
    assert list(doc) == ["start", "subject", "session", "source"]
    assert doc["subject"] == {"id": "sub-0931", "sex": "F"}
    assert doc["session"] == {"id": "ses-01"}
    assert doc["source"] == {"system": "Natus"}


def test_recording_document_does_not_alias_the_metadata():
    meta = _meta(subject={"id": "sub-0931"})
    doc = recording_document(meta, 0)
    doc["subject"]["id"] = "changed"
    assert meta.subject == {"id": "sub-0931"}


def test_write_recording_json_round_trips_the_document(tmp_path):
    doc = recording_document(
        _meta(subject={"id": "sub-0931"}, session={"id": "ses-01"}),
        _NOON_UTC_US,
    )
    write_recording_json(tmp_path, doc)
    assert json.loads((tmp_path / RECORDING_FILE).read_text("utf-8")) == doc


def test_write_recording_json_is_indented_utf8_with_one_trailing_newline(
    tmp_path,
):
    write_recording_json(tmp_path, {"start": "x", "session": {"id": "a"}})
    text = (tmp_path / RECORDING_FILE).read_text("utf-8")
    assert text == '{\n "start": "x",\n "session": {\n  "id": "a"\n }\n}\n'


def test_write_recording_json_keeps_non_ascii_text_readable(tmp_path):
    word = "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    write_recording_json(tmp_path, {"subject": {"note": word}})
    raw = (tmp_path / RECORDING_FILE).read_bytes()
    assert word.encode("utf-8") in raw


def test_write_recording_json_overwrites_an_existing_file(tmp_path):
    write_recording_json(tmp_path, {"start": "old"})
    write_recording_json(tmp_path, {"start": "new"})
    stored = json.loads((tmp_path / RECORDING_FILE).read_text("utf-8"))
    assert stored == {"start": "new"}


def test_write_recording_json_creates_no_zarr_node(tmp_path):
    write_recording_json(tmp_path, {"start": "x"})
    assert [p.name for p in tmp_path.iterdir()] == [RECORDING_FILE]


def test_write_recording_json_requires_an_existing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_recording_json(tmp_path / "missing", {"start": "x"})


def test_write_recording_json_rejects_a_value_that_is_not_json(tmp_path):
    with pytest.raises(TypeError):
        write_recording_json(tmp_path, {"subject": {"tags": {1, 2}}})
