"""Reading the interchange files the extractors produce."""

import json

import numpy as np
import pytest

from timeseries_zarr.annotation_ndjson import (
    AnnotationDocumentError,
    NdjsonAnnotationSource,
    build_annotation_sources,
)

SCHEMA = "pennsieve/annotations-1.1"


def _header(**channel):
    base = {
        "id": "clinical-marks",
        "name": "Clinical marks",
        "time_reference": "recording_onset",
    }
    base.update(channel)
    return {"schema": SCHEMA, "channel": base}


def _write(tmp_path, header, marks, name=None):
    stem = name or header["channel"]["id"]
    path = tmp_path / f"{stem}.annotations.ndjson"
    lines = [json.dumps(header), *(json.dumps(mark) for mark in marks)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _source(tmp_path, marks, onset=0, header=None, **kwargs):
    path = _write(tmp_path, header or _header(), marks)
    return NdjsonAnnotationSource(path, onset, **kwargs)


def test_the_header_is_the_first_line_and_the_rest_are_marks(tmp_path):
    source = _source(tmp_path, [{"time_us": 10}, {"time_us": 20}])
    assert source.id == "clinical-marks"
    assert source.num_events() == 2


def test_marks_are_sorted_by_time(tmp_path):
    """The bundle's events array must be non-decreasing."""
    source = _source(tmp_path, [{"time_us": 30}, {"time_us": 10}])
    assert list(source.read_events(0, 2)) == [10, 30]


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "clinical-marks.annotations.ndjson"
    path.write_text(
        json.dumps(_header()) + "\n\n" + json.dumps({"time_us": 5}) + "\n\n"
    )
    assert NdjsonAnnotationSource(path, 0).num_events() == 1


def test_a_recording_onset_file_is_rebased_onto_the_wall_clock(tmp_path):
    source = _source(tmp_path, [{"time_us": 500}], onset=1_000_000)
    assert list(source.read_events(0, 1)) == [1_000_500]


def test_a_unix_epoch_file_is_already_absolute(tmp_path):
    """The extractor declares the frame; only this reader converts."""
    source = _source(
        tmp_path,
        [{"time_us": 1_700_000_000_000_000}],
        onset=1_000_000,
        header=_header(time_reference="unix_epoch"),
    )
    assert list(source.read_events(0, 1)) == [1_700_000_000_000_000]


def test_an_unknown_time_reference_is_rejected(tmp_path):
    with pytest.raises(AnnotationDocumentError, match="time_reference"):
        _source(
            tmp_path, [{"time_us": 1}], header=_header(time_reference="tai")
        )


def test_a_missing_time_reference_is_rejected(tmp_path):
    header = _header()
    del header["channel"]["time_reference"]
    with pytest.raises(AnnotationDocumentError, match="time_reference"):
        _source(tmp_path, [{"time_us": 1}], header=header)


def test_an_unknown_schema_is_rejected_by_name(tmp_path):
    header = _header()
    header["schema"] = "pennsieve/annotations-9.9"
    with pytest.raises(AnnotationDocumentError, match="schema"):
        _source(tmp_path, [{"time_us": 1}], header=header)


def test_a_header_without_a_schema_is_rejected(tmp_path):
    """What tells a truncated file from one whose first line is data."""
    path = tmp_path / "x.annotations.ndjson"
    path.write_text(json.dumps({"time_us": 1}) + "\n")
    with pytest.raises(AnnotationDocumentError, match="schema"):
        NdjsonAnnotationSource(path, 0)


def test_an_empty_file_is_rejected(tmp_path):
    path = tmp_path / "x.annotations.ndjson"
    path.write_text("")
    with pytest.raises(AnnotationDocumentError, match="empty"):
        NdjsonAnnotationSource(path, 0)


def test_a_malformed_line_names_its_line_number(tmp_path):
    path = tmp_path / "x.annotations.ndjson"
    path.write_text(json.dumps(_header()) + "\n{not json}\n")
    with pytest.raises(AnnotationDocumentError, match="line 2"):
        NdjsonAnnotationSource(path, 0)


def test_labels_are_interned_in_sorted_order(tmp_path):
    """Sorted, so an index -- and the colour a viewer derives -- is stable."""
    source = _source(
        tmp_path,
        [
            {"time_us": 1, "label": "Seizure"},
            {"time_us": 2, "label": "Artifact"},
            {"time_us": 3, "label": "Seizure"},
        ],
    )
    assert source.label_names() == ["Artifact", "Seizure"]
    assert list(source.read_labels(0, 3)) == [1, 0, 1]


def test_the_label_order_does_not_depend_on_which_mark_came_first(tmp_path):
    one = _source(
        tmp_path,
        [{"time_us": 1, "label": "b"}, {"time_us": 2, "label": "a"}],
    )
    two = _source(
        tmp_path,
        [{"time_us": 1, "label": "a"}, {"time_us": 2, "label": "b"}],
    )
    assert one.label_names() == two.label_names() == ["a", "b"]


def test_an_unlabelled_file_has_no_label_space(tmp_path):
    source = _source(tmp_path, [{"time_us": 1}])
    assert source.num_labels() == 0
    assert source.label_names() is None


def test_durations_appear_only_when_a_mark_has_one(tmp_path):
    assert _source(tmp_path, [{"time_us": 1}]).max_duration_us() is None
    source = _source(
        tmp_path, [{"time_us": 1, "duration_us": 90}, {"time_us": 2}]
    )
    assert list(source.read_durations(0, 2)) == [90, 0]


def test_values_appear_only_when_a_mark_has_one(tmp_path):
    assert not _source(tmp_path, [{"time_us": 1}]).has_values()
    source = _source(tmp_path, [{"time_us": 1, "value": 7.2}])
    assert source.has_values()
    assert source.read_values(0, 1)[0] == pytest.approx(7.2)


def test_a_text_body_is_encoded_as_written(tmp_path):
    source = _source(
        tmp_path,
        [{"time_us": 1, "text": "Onset right temporal."}],
        header=_header(body_media_type="text/plain"),
    )
    assert source.read_bodies(0, 1) == [b"Onset right temporal."]


def test_a_json_body_is_serialised_compactly(tmp_path):
    source = _source(
        tmp_path,
        [{"time_us": 1, "body": {"band": "80-250", "score": 7.2}}],
        header=_header(body_media_type="application/json"),
    )
    assert json.loads(source.read_bodies(0, 1)[0]) == {
        "band": "80-250",
        "score": 7.2,
    }


def test_no_bodies_without_a_declared_media_type(tmp_path):
    source = _source(tmp_path, [{"time_us": 1, "text": "ignored"}])
    assert source.body_media_type() is None


def test_non_ascii_text_survives_the_read(tmp_path):
    """The database it came from is latin1; this file is UTF-8."""
    source = _source(
        tmp_path,
        [{"time_us": 1, "text": "30° rotation, Müller"}],
        header=_header(body_media_type="text/plain"),
    )
    assert source.read_bodies(0, 1)[0].decode() == "30° rotation, Müller"


def test_channel_names_resolve_to_bundle_indices(tmp_path):
    source = _source(
        tmp_path,
        [{"time_us": 1, "channels": ["LH3", "RH2"]}],
        channel_index_by_name={"LH3": 4, "RH2": 7},
    )
    assert list(source.read_channel_refs(0, 1)[0]) == [4, 7]


def test_a_mark_naming_no_channels_keeps_an_empty_span(tmp_path):
    """How the format says a mark belongs to the whole recording."""
    source = _source(
        tmp_path,
        [{"time_us": 1, "channels": ["LH3"]}, {"time_us": 2}],
        channel_index_by_name={"LH3": 0},
    )
    assert list(source.read_channel_refs(1, 2)[0]) == []


def test_channels_the_bundle_does_not_have_are_trimmed(tmp_path, caplog):
    """A partial conversion still gets the marks on the channels it has."""
    with caplog.at_level("WARNING"):
        source = _source(
            tmp_path,
            [{"time_us": 1, "channels": ["C3", "LH3"]}],
            channel_index_by_name={"C3": 0},
        )
    assert source.num_events() == 1
    assert list(source.read_channel_refs(0, 1)[0]) == [0]
    assert "LH3" in caplog.text


def test_a_mark_naming_only_missing_channels_is_dropped(tmp_path, caplog):
    """Not kept with an empty list, which would move it onto the recording."""
    with caplog.at_level("WARNING"):
        source = _source(
            tmp_path,
            [
                {"time_us": 1, "channels": ["LH3"], "label": "spike"},
                {"time_us": 2, "channels": ["C3"], "label": "sz"},
                {"time_us": 3, "label": "note"},
            ],
            channel_index_by_name={"C3": 0},
        )
    assert source.num_events() == 2
    assert list(source.read_events(0, 2)) == [2, 3]
    assert [list(refs) for refs in source.read_channel_refs(0, 2)] == [[0], []]
    assert source.label_names() == ["note", "sz"]
    assert "dropped 1 marks" in caplog.text


def test_no_warning_when_every_channel_is_present(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        _source(
            tmp_path,
            [{"time_us": 1, "channels": ["C3"]}],
            channel_index_by_name={"C3": 0},
        )
    assert "not in this bundle" not in caplog.text


def test_naming_channels_without_a_bundle_to_resolve_against_is_rejected(
    tmp_path,
):
    with pytest.raises(AnnotationDocumentError, match="needs the bundle"):
        _source(tmp_path, [{"time_us": 1, "channels": ["LH3"]}])


def test_a_stem_that_disagrees_with_the_channel_id_warns(tmp_path, caplog):
    """The merge keys on filename, so a mismatch is worth saying out loud."""
    path = _write(tmp_path, _header(), [{"time_us": 1}], name="something-else")
    NdjsonAnnotationSource(path, 0)
    assert "stem should match" in caplog.text


def test_build_rejects_two_files_declaring_one_channel_id(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = _write(tmp_path / "a", _header(), [{"time_us": 1}])
    second = _write(tmp_path / "b", _header(), [{"time_us": 2}])
    with pytest.raises(AnnotationDocumentError, match="two annotation files"):
        build_annotation_sources([first, second], 0)


def test_build_keeps_the_order_it_was_given(tmp_path):
    first = _write(tmp_path, _header(id="ictal"), [{"time_us": 1}])
    second = _write(tmp_path, _header(id="stability"), [{"time_us": 2}])
    sources = build_annotation_sources([second, first], 0)
    assert [source.id for source in sources] == ["stability", "ictal"]


def test_a_file_with_only_a_header_is_an_empty_channel(tmp_path):
    source = _source(tmp_path, [])
    assert source.num_events() == 0
    assert source.start_us() == 0
    assert source.read_events(0, 0).dtype == np.int64
