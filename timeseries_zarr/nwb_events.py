"""NWB adapters presenting interval and event tables as event channels."""

import json
import logging
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final

import numpy as np
import numpy.typing as npt
from pynwb.epoch import TimeIntervals

from timeseries_zarr.columns import text_width
from timeseries_zarr.constants import MAX_LABELS, MICROSECONDS_PER_SECOND
from timeseries_zarr.types import EventColumn

logger = logging.getLogger(__name__)

INTERVAL_COLUMNS: Final = frozenset(
    {"start_time", "stop_time", "tags", "timeseries"}
)
"""Columns of a TimeIntervals table that are not free-form text."""

UNTAGGED: Final = "untagged"
"""Label name of the category of rows that carry no tags."""


def _json_value(value: object) -> object:
    """Return value as something json can write, with NaN and infinity as null."""
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


class NwbIntervalSource:
    """An event channel backed by one NWB TimeIntervals table.

    Each row is one event placed at its start_time, with its length as the
    duration. The rows are sorted by start time, stably. The table's tags
    become label categories, its free-form columns become one JSON object of
    text per row, and its timeseries references become the channels the event
    applies to.
    """

    def __init__(
        self,
        table: TimeIntervals,
        session_start_time: datetime,
        channel_ids_by_series: Mapping[str, Sequence[str]],
    ) -> None:
        """Bind a TimeIntervals table as an event channel.

        session_start_time places the table's seconds, which count from it, in
        absolute microseconds. channel_ids_by_series maps the name of each
        series the bundle holds to the ids of that series' channels. A
        reference to a series the mapping lacks is ignored with a warning.
        Each distinct tag tuple, with its tags sorted, is one label category,
        in sorted order, and a row without tags is the category "untagged". A
        table with no tags in any row has no labels. Raises ValueError if a
        row stops before it starts or if the tag tuples outnumber MAX_LABELS.
        """
        self._id = str(table.name)
        session_us = round(
            session_start_time.timestamp() * MICROSECONDS_PER_SECOND
        )
        self._session_us = session_us
        starts = np.asarray(table["start_time"][:], dtype=np.float64)
        stops = np.asarray(table["stop_time"][:], dtype=np.float64)
        durations = np.rint((stops - starts) * MICROSECONDS_PER_SECOND)
        if (durations < 0).any():
            row = int(np.flatnonzero(durations < 0)[0])
            raise ValueError(f"row {row} of {self._id} stops before it starts")
        order = np.argsort(starts, kind="stable")
        offsets = np.rint(starts * MICROSECONDS_PER_SECOND).astype(np.int64)
        self._events: npt.NDArray[np.int64] = (session_us + offsets)[order]
        self._durations: npt.NDArray[np.int64] = durations.astype(np.int64)[
            order
        ]
        rows = len(starts)
        self._labels, self._label_names = self._read_labels(table, rows, order)
        self._texts = self._read_texts(table, rows, order)
        self._channel_ids = self._read_channels(
            table, rows, order, channel_ids_by_series
        )

    def _read_labels(
        self, table: TimeIntervals, rows: int, order: npt.NDArray[np.intp]
    ) -> tuple[npt.NDArray[np.uint16] | None, list[str]]:
        """Return the sorted rows' label indices and the category names."""
        if "tags" not in table.colnames:
            return None, []
        tuples = [
            tuple(sorted(str(tag) for tag in table["tags"][row]))
            for row in range(rows)
        ]
        categories = sorted(set(tuples))
        if categories in ([], [()]):
            return None, []
        if len(categories) > MAX_LABELS:
            raise ValueError(
                f"{len(categories)} tag combinations exceed {MAX_LABELS} labels"
            )
        index = {category: i for i, category in enumerate(categories)}
        labels = np.array([index[t] for t in tuples], dtype=np.uint16)[order]
        names = [",".join(c) if c else UNTAGGED for c in categories]
        return labels, names

    def _read_texts(
        self, table: TimeIntervals, rows: int, order: npt.NDArray[np.intp]
    ) -> list[str] | None:
        """Return one JSON object of the free-form columns per sorted row."""
        extra = [c for c in table.colnames if c not in INTERVAL_COLUMNS]
        if not extra or not rows:
            return None
        texts = [
            json.dumps(
                {c: _json_value(table[c][row]) for c in extra},
                separators=(",", ":"),
                ensure_ascii=False,
            )
            for row in range(rows)
        ]
        return [texts[i] for i in order]

    def _read_channels(
        self,
        table: TimeIntervals,
        rows: int,
        order: npt.NDArray[np.intp],
        channel_ids_by_series: Mapping[str, Sequence[str]],
    ) -> list[list[str]] | None:
        """Return the channel ids each sorted row applies to, or None."""
        if "timeseries" not in table.colnames:
            return None
        unresolved = 0
        per_row: list[list[str]] = []
        for row in range(rows):
            ids: list[str] = []
            for reference in table["timeseries"][row]:
                series = reference.timeseries
                if series is None:
                    continue
                if series.name in channel_ids_by_series:
                    ids.extend(channel_ids_by_series[series.name])
                else:
                    unresolved += 1
            per_row.append(list(dict.fromkeys(ids)))
        if unresolved:
            logger.warning(
                "%s: %d timeseries references name a series with no channel "
                "in the bundle and are ignored",
                self._id,
                unresolved,
            )
        if not any(per_row):
            return None
        return [per_row[i] for i in order]

    @property
    def id(self) -> str:
        """The table's name."""
        return self._id

    @property
    def name(self) -> str:
        """The table's name, the same value as id."""
        return self._id

    @property
    def unit(self) -> str | None:
        """Always None: an interval has no measurement."""
        return None

    def start_us(self) -> int:
        """Return the earlier of the session start and the first event."""
        if self._events.shape[0]:
            return min(self._session_us, int(self._events[0]))
        return self._session_us

    def num_events(self) -> int:
        """Return the number of rows."""
        return int(self._events.shape[0])

    def columns(self) -> frozenset[EventColumn]:
        """Return durations, plus labels, text, and channels where present.

        A table with no rows offers none.
        """
        if not self.num_events():
            return frozenset()
        present: set[EventColumn] = {"durations"}
        if self._labels is not None:
            present.add("labels")
        if self._texts is not None:
            present.add("text")
        if self._channel_ids is not None:
            present.add("channels")
        return frozenset(present)

    def label_names(self) -> list[str] | None:
        """Return the joined tags of each category, or None without labels."""
        return list(self._label_names) if self._labels is not None else None

    def num_labels(self) -> int:
        """Return the number of tag categories, or 0 without labels."""
        return len(self._label_names)

    def max_duration_us(self) -> int | None:
        """Return the longest duration, or None for a table with no rows."""
        if not self.num_events():
            return None
        return int(self._durations.max())

    def text_width(self) -> int:
        """Return the byte width of the longest text, or 0 without text."""
        return 0 if self._texts is None else text_width(self._texts)

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of event times in microseconds."""
        return self._events[start:stop]

    def read_durations(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of durations in microseconds."""
        return self._durations[start:stop]

    def read_labels(self, start: int, stop: int) -> npt.NDArray[np.uint16]:
        """Return the [start, stop) window of label indices.

        Raises ValueError if the table has no tags.
        """
        if self._labels is None:
            raise ValueError("the table has no tags, so no labels column")
        return self._labels[start:stop]

    def read_text(self, start: int, stop: int) -> list[str]:
        """Return the [start, stop) window of JSON texts.

        Raises ValueError if the table has no free-form columns.
        """
        if self._texts is None:
            raise ValueError("the table has no free-form columns, so no text")
        return self._texts[start:stop]

    def read_channel_ids(self, start: int, stop: int) -> list[list[str]]:
        """Return the channel ids of each event in [start, stop).

        Raises ValueError if no row references a bundled series.
        """
        if self._channel_ids is None:
            raise ValueError("no row references a channel, so no channels")
        return self._channel_ids[start:stop]

    def waveform_mean(self) -> npt.NDArray[np.float32]:
        """Raise ValueError, because an interval table has no snippets."""
        raise ValueError("an interval table has no waveform_mean column")

    def waveform_period_us(self) -> float:
        """Raise ValueError, because an interval table has no snippets."""
        raise ValueError("an interval table has no waveform_mean column")
