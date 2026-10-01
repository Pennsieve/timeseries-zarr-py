"""The annotation profile: marks, their spans, their payloads, their channels."""

import json

import numpy as np
import pytest

from timeseries_zarr.bundle import write_bundle
from timeseries_zarr.types import WriteOpts
from timeseries_zarr.write_annotation import (
    pack_bodies,
    pack_channel_refs,
    write_annotation_channel,
)
from timeseries_zarr.zarr_io import open_group

_OPTS = WriteOpts(inner_len=4, target_shard_bytes=256)


class FakeAnnotations:
    """In-memory AnnotationChannelSource with every column switchable."""

    def __init__(
        self,
        events,
        *,
        durations=None,
        labels=None,
        label_names=None,
        values=None,
        bodies=None,
        media_type=None,
        channel_refs=None,
        id="ann-0",
        name="Annotations",
        unit="",
        start_us=0,
        num_labels=0,
    ):
        self._events = np.asarray(events, dtype=np.int64)
        self._durations = (
            None if durations is None else np.asarray(durations, dtype=np.int64)
        )
        self._labels = (
            None if labels is None else np.asarray(labels, dtype=np.uint16)
        )
        self._label_names = label_names
        self._values = (
            None if values is None else np.asarray(values, dtype=np.float32)
        )
        self._bodies = bodies
        self._media_type = media_type
        self._channel_refs = channel_refs
        self.id = id
        self.name = name
        self.unit = unit
        self._start_us = start_us
        self._num_labels = num_labels

    def start_us(self):
        return self._start_us

    def num_events(self):
        return int(self._events.shape[0])

    def num_labels(self):
        return self._num_labels

    def label_names(self):
        return self._label_names

    def max_duration_us(self):
        if self._durations is None:
            return None
        return int(self._durations.max()) if self._durations.size else 0

    def body_media_type(self):
        return self._media_type

    def has_values(self):
        return self._values is not None

    def has_channel_refs(self):
        return self._channel_refs is not None

    def read_events(self, start, stop):
        return self._events[start:stop]

    def read_durations(self, start, stop):
        return self._durations[start:stop]

    def read_labels(self, start, stop):
        return self._labels[start:stop]

    def read_values(self, start, stop):
        return self._values[start:stop]

    def read_bodies(self, start, stop):
        return list(self._bodies[start:stop])

    def read_channel_refs(self, start, stop):
        return [
            np.asarray(item, dtype=np.uint16)
            for item in self._channel_refs[start:stop]
        ]


def _write(tmp_path, source, onset_us=0):
    parent = open_group(tmp_path / "bundle")
    write_annotation_channel(parent, 0, source, onset_us=onset_us, opts=_OPTS)
    return open_group(tmp_path / "bundle")["0"]


def test_pack_bodies_terminates_every_payload():
    joined, offsets = pack_bodies([b"one", b"two\n"])
    assert bytes(joined) == b"one\ntwo\n"
    assert list(offsets) == [0, 4, 8]


def test_pack_bodies_offsets_bracket_each_payload():
    payloads = [b"alpha", b"", b"gamma"]
    joined, offsets = pack_bodies(payloads)
    for index, payload in enumerate(payloads):
        span = bytes(joined[offsets[index] : offsets[index + 1]])
        assert span == payload + b"\n"


def test_pack_bodies_empty():
    joined, offsets = pack_bodies([])
    assert joined.shape == (0,)
    assert list(offsets) == [0]


def test_pack_channel_refs_uses_an_empty_span_for_the_whole_recording():
    flat, offsets = pack_channel_refs(
        [np.array([3, 4], np.uint16), np.array([], np.uint16)]
    )
    assert list(flat) == [3, 4]
    # The second mark's span is empty, which is how "every channel" is said.
    assert offsets[2] == offsets[1]


def test_pack_channel_refs_offsets_are_monotone_and_bracket_the_ends():
    refs = [
        np.array([1], np.uint16),
        np.array([], np.uint16),
        np.array([2, 3], np.uint16),
    ]
    flat, offsets = pack_channel_refs(refs)
    assert offsets[0] == 0
    assert offsets[-1] == flat.shape[0]
    assert np.all(np.diff(offsets.astype(np.int64)) >= 0)


def test_a_point_channel_writes_only_events(tmp_path):
    grp = _write(tmp_path, FakeAnnotations([10, 20, 30]))
    assert list(grp.array_keys()) == ["events"]
    assert list(grp["events"][:]) == [10, 20, 30]


def test_the_channel_declares_the_event_kind_and_no_rate(tmp_path):
    grp = _write(tmp_path, FakeAnnotations([10]))
    attrs = dict(grp.attrs)
    assert attrs["kind"] == "event"
    # Marks are placed by their own timestamps, not sampled.
    assert "rate_hz" not in attrs


def test_events_are_onset_relative(tmp_path):
    onset = 1_000_000
    grp = _write(
        tmp_path,
        FakeAnnotations([onset + 5, onset + 9], start_us=onset),
        onset_us=onset,
    )
    assert list(grp["events"][:]) == [5, 9]
    assert dict(grp.attrs)["offset_us"] == 0


def test_durations_bring_their_bound(tmp_path):
    grp = _write(tmp_path, FakeAnnotations([10, 20], durations=[90_000_000, 5]))
    assert list(grp["durations"][:]) == [90_000_000, 5]
    # The bound is what lets a window find an interval still open at its edge.
    assert dict(grp.attrs)["max_duration_us"] == 90_000_000


def test_a_channel_without_durations_declares_no_bound(tmp_path):
    grp = _write(tmp_path, FakeAnnotations([10, 20]))
    assert "durations" not in list(grp.array_keys())
    assert "max_duration_us" not in dict(grp.attrs)


def test_labels_are_u2_and_can_be_named(tmp_path):
    grp = _write(
        tmp_path,
        FakeAnnotations(
            [10, 20],
            labels=[0, 1],
            num_labels=2,
            label_names=["seizure", "artifact"],
        ),
    )
    assert grp["labels"].dtype == np.uint16
    assert list(grp["labels"][:]) == [0, 1]
    assert dict(grp.attrs)["label_names"] == ["seizure", "artifact"]


def test_values_carry_the_channels_unit(tmp_path):
    grp = _write(
        tmp_path,
        FakeAnnotations([10, 20], values=[1.5, 2.5], unit="ms"),
    )
    assert list(grp["values"][:]) == [1.5, 2.5]
    assert dict(grp.attrs)["unit"] == "ms"


def test_bodies_round_trip_through_their_offsets(tmp_path):
    payloads = [b'{"score":7.2}', b'{"score":4.9}']
    grp = _write(
        tmp_path,
        FakeAnnotations(
            [10, 20], bodies=payloads, media_type="application/json"
        ),
    )
    joined = bytes(grp["bodies"][:])
    offsets = grp["body_offsets"][:]
    assert dict(grp.attrs)["body_media_type"] == "application/json"
    for index, payload in enumerate(payloads):
        span = joined[offsets[index] : offsets[index + 1]]
        assert json.loads(span) == json.loads(payload)


def test_bodies_read_as_ndjson_without_touching_the_offsets(tmp_path):
    """The point of the newline rule: one array read is a readable file."""
    payloads = [b'{"a":1}', b'{"a":2}', b'{"a":3}']
    grp = _write(
        tmp_path,
        FakeAnnotations(
            [1, 2, 3], bodies=payloads, media_type="application/json"
        ),
    )
    text = bytes(grp["bodies"][:]).decode()
    assert [json.loads(line)["a"] for line in text.splitlines()] == [1, 2, 3]


def test_bodies_are_one_chunk(tmp_path):
    """The chunk object is then itself the payload file, openable with no tooling."""
    payloads = [b"x" * 100 for _ in range(50)]
    grp = _write(
        tmp_path,
        FakeAnnotations(
            list(range(50)), bodies=payloads, media_type="text/plain"
        ),
    )
    bodies = grp["bodies"]
    assert bodies.chunks[0] >= bodies.shape[0]


def test_body_offsets_are_monotone_with_fixed_endpoints(tmp_path):
    payloads = [b"a", b"bb", b"ccc"]
    grp = _write(
        tmp_path,
        FakeAnnotations([1, 2, 3], bodies=payloads, media_type="text/plain"),
    )
    offsets = grp["body_offsets"][:].astype(np.int64)
    assert offsets[0] == 0
    assert offsets[-1] == grp["bodies"].shape[0]
    assert np.all(np.diff(offsets) >= 0)


def test_channel_refs_name_the_electrodes_a_mark_belongs_to(tmp_path):
    grp = _write(
        tmp_path,
        FakeAnnotations(
            [10, 20, 30],
            channel_refs=[[2, 3, 4], [], [7]],
        ),
    )
    flat = grp["channel_refs"][:]
    offsets = grp["channel_ref_offsets"][:].astype(np.int64)
    assert grp["channel_refs"].dtype == np.uint16
    assert list(flat[offsets[0] : offsets[1]]) == [2, 3, 4]
    # An empty span means the mark belongs to the recording, not to electrodes.
    assert offsets[1] == offsets[2]
    assert list(flat[offsets[2] : offsets[3]]) == [7]


def test_channel_ref_offsets_have_one_entry_per_mark_plus_one(tmp_path):
    grp = _write(
        tmp_path,
        FakeAnnotations([1, 2, 3], channel_refs=[[0], [1], [2]]),
    )
    assert grp["channel_ref_offsets"].shape == (4,)


def test_every_per_event_column_has_the_same_length(tmp_path):
    n = 6
    grp = _write(
        tmp_path,
        FakeAnnotations(
            list(range(n)),
            durations=[1] * n,
            labels=[0] * n,
            num_labels=1,
            values=[0.0] * n,
            bodies=[b"x"] * n,
            media_type="text/plain",
            channel_refs=[[0]] * n,
        ),
    )
    for key in ("events", "durations", "labels", "values"):
        assert grp[key].shape == (n,)
    for key in ("body_offsets", "channel_ref_offsets"):
        assert grp[key].shape == (n + 1,)


def test_a_detector_channel_can_carry_every_column(tmp_path):
    grp = _write(
        tmp_path,
        FakeAnnotations(
            [10, 20],
            durations=[5, 5],
            labels=[0, 1],
            num_labels=2,
            values=[1.0, 2.0],
            bodies=[b"{}", b"{}"],
            media_type="application/json",
            channel_refs=[[1], []],
        ),
    )
    for key in (
        "events",
        "durations",
        "labels",
        "values",
        "bodies",
        "body_offsets",
        "channel_refs",
        "channel_ref_offsets",
    ):
        assert key in list(grp.array_keys())


def test_an_empty_channel_writes_empty_columns(tmp_path):
    grp = _write(
        tmp_path,
        FakeAnnotations([], bodies=[], media_type="text/plain"),
    )
    assert grp["events"].shape == (0,)
    assert grp["bodies"].shape == (0,)
    assert list(grp["body_offsets"][:]) == [0]


@pytest.mark.parametrize("n", [1, 5, 9, 33])
def test_columns_survive_more_marks_than_one_shard(tmp_path, n):
    grp = _write(
        tmp_path,
        FakeAnnotations(
            list(range(n)),
            durations=[2] * n,
            bodies=[str(i).encode() for i in range(n)],
            media_type="text/plain",
        ),
    )
    assert list(grp["events"][:]) == list(range(n))
    assert list(grp["durations"][:]) == [2] * n
    text = bytes(grp["bodies"][:]).decode().splitlines()
    assert text == [str(i) for i in range(n)]


def test_a_bundle_places_annotations_after_the_channels_they_reference(
    tmp_path, continuous_source
):
    """channel_refs name indices, so the order they are assigned in matters."""
    samples = np.arange(8, dtype=np.float32)
    final = tmp_path / "bundle"
    write_bundle(
        [
            continuous_source(samples, id="c0"),
            continuous_source(samples, id="c1"),
        ],
        [],
        annotations=[FakeAnnotations([10, 20], channel_refs=[[1], []])],
        staging_dir=tmp_path / "staging",
        final_dir=final,
        opts=_OPTS,
    )
    root = open_group(final)
    assert dict(root["0"].attrs)["id"] == "c0"
    assert dict(root["1"].attrs)["id"] == "c1"
    assert dict(root["2"].attrs)["kind"] == "event"

    flat = root["2"]["channel_refs"][:]
    offsets = root["2"]["channel_ref_offsets"][:].astype(np.int64)
    referenced = int(flat[offsets[0]])
    # The reference resolves to a continuous channel of this bundle.
    assert dict(root[str(referenced)].attrs)["id"] == "c1"


def test_channel_refs_address_channels_present_in_the_bundle(
    tmp_path, continuous_source
):
    samples = np.arange(8, dtype=np.float32)
    final = tmp_path / "bundle"
    write_bundle(
        [continuous_source(samples, id="c0")],
        [],
        annotations=[FakeAnnotations([1], channel_refs=[[0]])],
        staging_dir=tmp_path / "staging",
        final_dir=final,
        opts=_OPTS,
    )
    root = open_group(final)
    present = {int(key) for key in root.group_keys() if key.isdigit()}
    for index in root["1"]["channel_refs"][:]:
        assert int(index) in present


def test_the_bodies_chunk_object_is_the_payload_file(tmp_path):
    """cat on it must print text: no Zstd frame, no shard index footer.

    This is what keeps annotation bodies, the one place PHI can hide in a
    bundle besides meta/, greppable and redactable with ordinary tools.
    """
    _write(
        tmp_path,
        FakeAnnotations(
            [1, 2],
            bodies=[b'{"a":1}', b'{"a":2}'],
            media_type="application/json",
        ),
    )
    chunk = next(
        path
        for path in (tmp_path / "bundle" / "0" / "bodies").rglob("*")
        if path.is_file() and path.name != "zarr.json"
    )
    assert chunk.read_bytes() == b'{"a":1}\n{"a":2}\n'


def test_max_duration_us_is_computed_from_what_was_written(tmp_path):
    """A source that under-reports the bound must not be believed.

    The reader widens its backward search by this number; too small and a
    window silently misses an interval still open at its left edge.
    """

    class Understates(FakeAnnotations):
        def max_duration_us(self):
            return 1  # a true bound would be 500

    grp = _write(tmp_path, Understates([10, 20], durations=[500, 5]))
    assert dict(grp.attrs)["max_duration_us"] == 500


def test_a_negative_duration_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="negative"):
        _write(tmp_path, FakeAnnotations([10], durations=[-1]))
