"""Read the annotation interchange files the extractors produce.

Every annotation source -- baked into an EDF, a sidecar, a detector run, a
database -- is normalised by its own extractor into one NDJSON file, and the
writer takes NWB plus those files as its inputs. Annotations do not travel
inside NWB because annotations from a database have no NWB home at all, so an
NWB-only path would have needed a second path anyway.

One file is one annotation source and becomes one event channel. A recording
with EDF-baked marks, a detector run and a clinical export produces three files
and three channels.

The first line is the header and every line after it is one mark. Only the
header carries `schema`, which is what lets a truncated or concatenated file be
detected rather than read as data. NDJSON rather than one document because a
detector run is millions of marks, and a document has to be parsed whole before
any of it is usable.

## The timebase

The extractor declares the frame and this reader converts it. That split is
forced by the workflow: the extractor runs in a branch parallel to the signal
conversion, so it cannot read the NWB and cannot know the onset.

- `recording_onset` -- microseconds from the start of the recording.
- `unix_epoch` -- microseconds since 1970.

Both are rebased here onto the absolute timeline the rest of the writer works
in, and the bundle rebases everything once more onto its own onset. Get it
wrong and nothing errors: the marks simply sit at the wrong time, which at
2 kHz nobody sees by eye. It is the highest-risk field in the format.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)

SUPPORTED_SCHEMAS = frozenset({"pennsieve/annotations-1.1"})
"""Versions this writer reads. Rejected by name rather than guessed at."""

RECORDING_ONSET = "recording_onset"
UNIX_EPOCH = "unix_epoch"
TIME_REFERENCES = frozenset({RECORDING_ONSET, UNIX_EPOCH})

TEXT_MEDIA_TYPE = "text/plain"
JSON_MEDIA_TYPE = "application/json"

FILE_SUFFIX = ".annotations.ndjson"
"""What the writer discovers. Distinctive enough not to catch asset-properties."""


class AnnotationDocumentError(ValueError):
    """The file is not one this writer can turn into a channel."""


def _require(document: dict[str, Any], key: str, where: str) -> object:
    """Return a required field, naming what was missing rather than KeyError."""
    if key not in document:
        raise AnnotationDocumentError(f"{where} is missing {key!r}")
    return document[key]


def iter_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each line of an NDJSON file as an object.

    Blank lines are skipped so a trailing newline is not an error. A line that
    is not an object is rejected where it is, because the line number is the
    only useful thing to say about a malformed file.
    """
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                raise AnnotationDocumentError(
                    f"{path.name} line {number} is not valid JSON: {error}"
                ) from error
            if not isinstance(value, dict):
                raise AnnotationDocumentError(
                    f"{path.name} line {number} holds a "
                    f"{type(value).__name__}, not an object"
                )
            yield value


class NdjsonAnnotationSource:
    """One annotation file, presented as an annotation channel source.

    Which columns the channel carries follows from the file: durations if any
    mark has one, labels if any is labelled, values if any carries a
    measurement, bodies if the header declares a media type.
    """

    def __init__(
        self,
        path: Path,
        recording_onset_us: int,
        *,
        channel_index_by_name: dict[str, int] | None = None,
    ) -> None:
        """Read one file and bind it as a channel source.

        recording_onset_us is the wall-clock microsecond a recording_onset file
        is measured from. channel_index_by_name resolves channel names to the
        bundle's indices; without it a file naming channels is rejected rather
        than written with references pointing nowhere.
        """
        lines = iter_lines(path)
        try:
            header = next(lines)
        except StopIteration:
            raise AnnotationDocumentError(f"{path.name} is empty") from None

        schema = _require(header, "schema", "the header")
        if schema not in SUPPORTED_SCHEMAS:
            raise AnnotationDocumentError(
                f"{path.name} declares schema {schema!r}: this writer reads "
                f"{sorted(SUPPORTED_SCHEMAS)}"
            )

        channel: dict[str, Any] = _require(  # type: ignore[assignment]
            header, "channel", "the header"
        )
        self._id = str(_require(channel, "id", "the header's channel"))
        self._name = str(channel.get("name", self._id))
        self._unit = str(channel.get("unit", ""))
        media_type = channel.get("body_media_type")
        self._media_type: str | None = (
            None if media_type is None else str(media_type)
        )

        reference = str(
            _require(channel, "time_reference", "the header's channel")
        )
        if reference not in TIME_REFERENCES:
            raise AnnotationDocumentError(
                f"{path.name} declares time_reference {reference!r}: only "
                f"{sorted(TIME_REFERENCES)} are defined"
            )

        # The stem is the channel id by convention, and the branch merge keys
        # on filename. A mismatch means two files could collide on disk while
        # their ids differ, or the reverse, so it is worth saying out loud.
        stem = path.name.removesuffix(FILE_SUFFIX)
        if stem != self._id:
            log.warning(
                "%s declares channel id %r; the stem should match it, since "
                "the workflow merge keys on filename",
                path.name,
                self._id,
            )

        marks = sorted(lines, key=lambda mark: int(mark["time_us"]))
        if channel_index_by_name is not None:
            marks = _drop_missing_channels(
                marks, channel_index_by_name, path.name
            )
        origin = recording_onset_us if reference == RECORDING_ONSET else 0

        self._events = np.array(
            [origin + int(mark["time_us"]) for mark in marks], dtype=np.int64
        )
        self._durations = (
            np.array(
                [int(mark.get("duration_us", 0)) for mark in marks],
                dtype=np.int64,
            )
            if any("duration_us" in mark for mark in marks)
            else None
        )
        self._values = (
            np.array(
                [float(mark.get("value", np.nan)) for mark in marks],
                dtype=np.float32,
            )
            if any("value" in mark for mark in marks)
            else None
        )

        # Sorted rather than first-seen, so a label keeps its index -- and so
        # the colour a viewer derives from it -- across re-runs and across
        # files that saw the same labels in a different order.
        names = sorted(
            {str(mark["label"]) for mark in marks if "label" in mark}
        )
        index_of = {name: index for index, name in enumerate(names)}
        self._label_names = names or None
        self._labels = (
            np.array(
                [index_of.get(str(mark.get("label")), 0) for mark in marks],
                dtype=np.uint16,
            )
            if names
            else None
        )

        self._bodies = (
            [self._encode_body(mark) for mark in marks]
            if self._media_type is not None
            else None
        )
        self._refs = self._resolve_refs(marks, channel_index_by_name)

    def _encode_body(self, mark: dict[str, Any]) -> bytes:
        """Return one mark's payload as bytes, in the channel's media type."""
        if self._media_type == JSON_MEDIA_TYPE:
            return json.dumps(
                mark.get("body", mark.get("text", {})),
                separators=(",", ":"),
            ).encode()
        return str(mark.get("text", mark.get("body", ""))).encode()

    def _resolve_refs(
        self,
        marks: list[dict[str, Any]],
        index_by_name: dict[str, int] | None,
    ) -> list[npt.NDArray[np.uint16]] | None:
        """Turn each mark's channel names into this bundle's channel indices.

        A mark naming none keeps an empty array, which is how the format says
        it belongs to the recording rather than to named electrodes.
        """
        if not any("channels" in mark for mark in marks):
            return None
        if index_by_name is None:
            raise AnnotationDocumentError(
                "this file names channels, so it needs the bundle's channel "
                "names to resolve them against"
            )
        resolved: list[npt.NDArray[np.uint16]] = []
        for mark in marks:
            names = [str(name) for name in mark.get("channels", [])]
            resolved.append(
                np.array(
                    [index_by_name[name] for name in names], dtype=np.uint16
                )
            )
        return resolved

    @property
    def id(self) -> str:
        """The header's channel id."""
        return self._id

    @property
    def name(self) -> str:
        """The channel's display label."""
        return self._name

    @property
    def unit(self) -> str:
        """The unit of the per-mark value, when the header declares one."""
        return self._unit

    def start_us(self) -> int:
        """Return the wall-clock microseconds of the first mark."""
        return int(self._events[0]) if self._events.size else 0

    def num_events(self) -> int:
        """Return the number of marks."""
        return int(self._events.shape[0])

    def num_labels(self) -> int:
        """Return the size of the label space, 0 when nothing is labelled."""
        return len(self._label_names) if self._label_names else 0

    def label_names(self) -> list[str] | None:
        """Return the label names, index-aligned with the labels column."""
        return self._label_names

    def max_duration_us(self) -> int | None:
        """Return a bound on the durations, or None when the marks are points.

        The writer recomputes this from what it writes; this only declares that
        the column is there.
        """
        if self._durations is None:
            return None
        return int(self._durations.max()) if self._durations.size else 0

    def body_media_type(self) -> str | None:
        """Return the payload media type, or None when there are no payloads."""
        return self._media_type

    def has_values(self) -> bool:
        """Whether any mark carries a measurement."""
        return self._values is not None

    def has_channel_refs(self) -> bool:
        """Whether any mark names the channels it belongs to."""
        return self._refs is not None

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of absolute-microsecond timestamps."""
        return self._events[start:stop]

    def read_durations(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of durations in microseconds."""
        assert self._durations is not None
        return self._durations[start:stop]

    def read_labels(self, start: int, stop: int) -> npt.NDArray[np.uint16]:
        """Return the [start, stop) window of label indices."""
        assert self._labels is not None
        return self._labels[start:stop]

    def read_values(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the [start, stop) window of per-mark measurements."""
        assert self._values is not None
        return self._values[start:stop]

    def read_bodies(self, start: int, stop: int) -> list[bytes]:
        """Return one encoded payload per mark in [start, stop)."""
        assert self._bodies is not None
        return self._bodies[start:stop]

    def read_channel_refs(
        self, start: int, stop: int
    ) -> list[npt.NDArray[np.uint16]]:
        """Return the channel indices each mark in [start, stop) applies to."""
        assert self._refs is not None
        return self._refs[start:stop]


def _drop_missing_channels(
    marks: list[dict[str, Any]],
    index_by_name: dict[str, int],
    file_name: str,
) -> list[dict[str, Any]]:
    """Keep only the channel names this bundle has, and warn about the rest.

    The extractor runs parallel to signal conversion, so it can't know when
    only some channels were converted. A mark left naming none of the bundle's
    channels is dropped, not kept with an empty list, because an empty list
    means the mark belongs to the whole recording.
    """
    kept: list[dict[str, Any]] = []
    missing: set[str] = set()
    trimmed = 0
    for mark in marks:
        if "channels" not in mark:
            kept.append(mark)
            continue
        names = [str(name) for name in mark["channels"]]
        present = [name for name in names if name in index_by_name]
        missing.update(name for name in names if name not in index_by_name)
        if names and not present:
            continue
        if len(present) == len(names):
            kept.append(mark)
        else:
            trimmed += 1
            kept.append({**mark, "channels": present})

    if missing:
        log.warning(
            "%s names %d channels not in this bundle %s; trimmed them from %d "
            "marks and dropped %d marks that named only those channels",
            file_name,
            len(missing),
            sorted(missing),
            trimmed,
            len(marks) - len(kept),
        )
    return kept


def build_annotation_sources(
    paths: list[Path],
    recording_onset_us: int,
    *,
    channel_index_by_name: dict[str, int] | None = None,
) -> list[NdjsonAnnotationSource]:
    """Return one source per file, in the order the paths were given.

    One file is one channel, so this order is the order the annotation channels
    are indexed in. Two files declaring the same channel id are rejected: the
    bundle would carry one channel where the workflow produced two.
    """
    sources = [
        NdjsonAnnotationSource(
            path,
            recording_onset_us,
            channel_index_by_name=channel_index_by_name,
        )
        for path in paths
    ]
    seen: set[str] = set()
    for source in sources:
        if source.id in seen:
            raise AnnotationDocumentError(
                f"two annotation files declare channel id {source.id!r}"
            )
        seen.add(source.id)
    return sources
