"""Bundle, channel, and window objects over a stored TS-Zarr bundle.

The public entry point is :func:`open_bundle`. All times in this API
are float seconds relative to bundle onset; microseconds are internal.
Statistic members are stored with ``offset_uv`` subtracted; readers in
this module add it back, so returned values are physical microvolts.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from timeseries_zarr.reading.serving import ServingError

US = 1_000_000.0


def _is_level_name(name: str) -> bool:
    return name.isdigit()


@dataclass(frozen=True)
class Level:
    """One stored pyramid level of a channel."""

    k: int
    period_us: float
    group: zarr.Group

    @property
    def rate_hz(self) -> float:
        """Bins per second at this level."""
        return US / self.period_us

    @property
    def n_bins(self) -> int:
        """Number of bins along the shared bin axis."""
        member = next(iter(self.group.array_keys()))
        return int(self.group[member].shape[0])


@dataclass
class Window:
    """A served view of one channel over ``[t0, t1)``.

    ``level`` is 0 for raw. ``mean`` and ``env`` are physical values
    (``offset_uv`` restored). ``t`` holds bin centers (or sample
    times) in seconds from bundle onset.
    """

    level: int
    period_s: float
    t: np.ndarray
    mean: np.ndarray | None = None
    env: np.ndarray | None = None
    raw: np.ndarray | None = None
    marked: bool = False
    note: str = ""


class _Channel:
    """Shared behavior of continuous and event channels."""

    def __init__(self, group: zarr.Group, key: str) -> None:
        self.group = group
        self.key = key
        self.attrs: dict[str, Any] = dict(group.attrs)

    @property
    def id(self) -> str:  # noqa: A003 - spec attribute name
        """Upstream identifier (opaque/pseudonymous)."""
        return str(self.attrs["id"])

    @property
    def name(self) -> str:
        """Display label."""
        return str(self.attrs["name"])

    @property
    def kind(self) -> str:
        """``continuous`` or ``event``."""
        return str(self.attrs["kind"])

    @property
    def offset_us(self) -> int:
        """Microseconds from bundle onset of sample/event 0."""
        return int(self.attrs["offset_us"])

    def _levels(self) -> list[Level]:
        levels = []
        for name in self.group.group_keys():
            if _is_level_name(name):
                grp = self.group[name]
                levels.append(
                    Level(int(name), float(grp.attrs["period_us"]), grp)
                )
        return sorted(levels, key=lambda lv: lv.k)

    @property
    def levels(self) -> list[Level]:
        """Stored level groups, finest first."""
        return self._levels()


class ContinuousChannel(_Channel):
    """A sampled-signal channel: ``raw`` plus statistic level groups."""

    @property
    def rate_hz(self) -> float:
        """Native sampling rate."""
        return float(self.attrs["rate_hz"])

    @property
    def unit(self) -> str:
        """Physical unit of samples."""
        return str(self.attrs.get("unit", "uV"))

    @property
    def offset_uv(self) -> float:
        """Per-channel offset subtracted before statistic folding."""
        return float(self.attrs.get("offset_uv", 0.0))

    @property
    def has_raw(self) -> bool:
        """False for a viewing-only bundle (raw omitted)."""
        return "raw" in self.group.array_keys()

    @property
    def duration_s(self) -> float:
        """Signal extent in seconds (from raw or the finest level)."""
        if self.has_raw:
            return self.group["raw"].shape[0] / self.rate_hz
        finest = self.levels[0]
        return finest.n_bins * finest.period_us / US

    def _bin_slice(
        self, level: Level, t0: float, t1: float
    ) -> tuple[int, int]:
        """Bin index range covering ``[t0, t1)`` at ``level``."""
        rel0 = t0 * US - self.offset_us
        rel1 = t1 * US - self.offset_us
        i0 = max(0, int(np.floor(rel0 / level.period_us)))
        i1 = min(
            level.n_bins, int(np.ceil(rel1 / level.period_us))
        )
        return i0, max(i0, i1)

    def pick_level(
        self, t0: float, t1: float, pixels: int
    ) -> Level | None:
        """Coarsest level giving >= 1 bin per pixel column, or None.

        ``None`` means the window is too short for any level: serve
        raw (raise :class:`ServingError` if the bundle is
        viewing-only and raw is absent).
        """
        window_us = (t1 - t0) * US
        best = None
        for lv in self.levels:  # finest first
            if window_us / lv.period_us >= pixels:
                best = lv  # keep coarsening while resolution holds
            else:
                break
        return best

    def window(self, t0: float, t1: float, pixels: int = 2000) -> Window:
        """Serve an envelope/mean view of ``[t0, t1)`` at ``pixels``."""
        lv = self.pick_level(t0, t1, pixels)
        if lv is None:
            if not self.has_raw:
                raise ServingError(
                    f"window shorter than the finest level of "
                    f"viewing-only channel {self.key!r}; no raw to "
                    f"fall back to"
                )
            return self.raw_window(t0, t1)
        i0, i1 = self._bin_slice(lv, t0, t1)
        off = self.offset_uv
        env = lv.group["env"][i0:i1].astype(np.float64) + off
        mean = lv.group["mean"][i0:i1].astype(np.float64) + off
        centers = (
            self.offset_us + (np.arange(i0, i1) + 0.5) * lv.period_us
        ) / US
        return Window(
            level=lv.k,
            period_s=lv.period_us / US,
            t=centers,
            mean=mean,
            env=env,
        )

    def raw_window(self, t0: float, t1: float) -> Window:
        """Read raw samples over ``[t0, t1)``."""
        if not self.has_raw:
            raise ServingError(
                f"channel {self.key!r} is viewing-only (no raw)"
            )
        rate = self.rate_hz
        n = self.group["raw"].shape[0]
        i0 = max(0, int(np.floor((t0 * US - self.offset_us) / US * rate)))
        i1 = min(n, int(np.ceil((t1 * US - self.offset_us) / US * rate)))
        i1 = max(i0, i1)
        raw = self.group["raw"][i0:i1].astype(np.float64)
        t = (self.offset_us / US) + np.arange(i0, i1) / rate
        return Window(
            level=0, period_s=1.0 / rate, t=t, raw=raw, mean=raw
        )

    def mean_series(
        self, level: Level, i0: int, i1: int
    ) -> np.ndarray:
        """Physical mean values for bins ``[i0, i1)`` of ``level``."""
        return (
            level.group["mean"][i0:i1].astype(np.float64)
            + self.offset_uv
        )

    def filtered_window(
        self,
        t0: float,
        t1: float,
        low_hz: float | None,
        high_hz: float | None,
        *,
        pixels: int = 2000,
        order: int = 4,
        broadband: bool = False,
    ) -> Window:
        """Serve a filtered view under the rho rule (see filtering)."""
        from timeseries_zarr.reading.filtering import (  # noqa: PLC0415 - import cycle
            filtered_window,
        )

        return filtered_window(
            self,
            t0,
            t1,
            low_hz=low_hz,
            high_hz=high_hz,
            pixels=pixels,
            order=order,
            broadband=broadband,
        )


class EventChannel(_Channel):
    """A sparse channel: sorted ``events`` plus optional columns."""

    def __init__(self, group: zarr.Group, key: str) -> None:
        """Wrap an event channel group and lazily cache events."""
        super().__init__(group, key)
        self._events_us: np.ndarray | None = None

    @property
    def label_names(self) -> list[str] | None:
        """Names for label categories, index-aligned."""
        names = self.attrs.get("label_names")
        return list(names) if names is not None else None

    @property
    def n_events(self) -> int:
        """Total event count."""
        return int(self.group["events"].shape[0])

    def _events(self) -> np.ndarray:
        if self._events_us is None:
            self._events_us = self.group["events"][:]
        return self._events_us

    def _column(
        self, name: str, i0: int, i1: int
    ) -> np.ndarray | None:
        if name in self.group.array_keys():
            return self.group[name][i0:i1]
        return None

    def between(
        self, t0: float, t1: float, *, label_set: str = "labels"
    ) -> dict[str, np.ndarray]:
        """Events with ``t0 <= time < t1``; times in seconds.

        Returns a dict with ``times_s``, ``index``, and any of
        ``durations_s``, ``labels``, ``values`` that are stored.
        """
        ev = self._events()
        i0 = int(np.searchsorted(ev, int(t0 * US), side="left"))
        i1 = int(np.searchsorted(ev, int(t1 * US), side="left"))
        out: dict[str, np.ndarray] = {
            "times_s": ev[i0:i1] / US,
            "index": np.arange(i0, i1),
        }
        dur = self._column("durations", i0, i1)
        if dur is not None:
            out["durations_s"] = dur / US
        lab = self._column(label_set, i0, i1)
        if lab is not None:
            out["labels"] = lab
        val = self._column("values", i0, i1)
        if val is not None:
            out["values"] = val
        return out

    def overlapping(self, t0: float, t1: float) -> dict[str, np.ndarray]:
        """Interval events overlapping ``[t0, t1)`` (stabbing query).

        Uses ``max_duration_us`` to bound the backward search, per the
        spec; requires ``durations``.
        """
        if "durations" not in self.group.array_keys():
            return self.between(t0, t1)
        max_dur = int(self.attrs["max_duration_us"])
        cand = self.between(t0 - max_dur / US, t1)
        end_s = cand["times_s"] + cand["durations_s"]
        keep = end_s >= t0
        return {k: v[keep] for k, v in cand.items()}

    def body(self, index: int) -> str | dict[str, Any]:
        """Decode one event body (JSON-parsed when declared JSON)."""
        offs = self.group["body_offsets"][index : index + 2]
        o0, o1 = int(offs[0]), int(offs[1])
        text = (
            self.group["bodies"][o0:o1].tobytes().decode("utf-8").rstrip("\n")
        )
        media = str(self.attrs.get("body_media_type", "text/plain"))
        if media == "application/json":
            return json.loads(text)
        return text

    def rates(
        self,
        t0: float,
        t1: float,
        pixels: int = 500,
        *,
        label_set: str = "labels",
    ) -> Window:
        """Per-label event rates from the count pyramid.

        Returns a :class:`Window` whose ``mean`` holds counts per bin
        (shape ``(bins, n_labels)`` or ``(bins,)``), served from the
        coarsest count level with >= 1 bin per pixel. Exact at every
        level (counts fold by sum).
        """
        member = (
            "counts"
            if label_set == "labels"
            else f"counts.{label_set}"
        )
        window_us = (t1 - t0) * US
        chosen = None
        for lv in self.levels:
            if member not in lv.group.array_keys():
                continue
            if window_us / lv.period_us >= pixels:
                chosen = lv
            else:
                break
        if chosen is None:
            raise ServingError(
                f"no count level of {self.key!r} serves {pixels}px "
                f"over {t1 - t0:.1f}s; slice `events` instead"
            )
        n = chosen.group[member].shape[0]
        rel0 = t0 * US - self.offset_us
        rel1 = t1 * US - self.offset_us
        i0 = max(0, int(np.floor(rel0 / chosen.period_us)))
        i1 = min(n, int(np.ceil(rel1 / chosen.period_us)))
        counts = chosen.group[member][i0 : max(i0, i1)]
        centers = (
            self.offset_us
            + (np.arange(i0, max(i0, i1)) + 0.5) * chosen.period_us
        ) / US
        return Window(
            level=chosen.k,
            period_s=chosen.period_us / US,
            t=centers,
            mean=counts,
        )


class Bundle:
    """An open TS-Zarr bundle."""

    def __init__(self, root: zarr.Group, path: str | Path) -> None:
        """Index the channel groups of an opened root."""
        self.root = root
        self.path = path
        self.channels: dict[str, _Channel] = {}
        for name in root.group_keys():
            if name == "meta":
                continue
            grp = root[name]
            kind = grp.attrs.get("kind")
            if kind == "continuous":
                self.channels[name] = ContinuousChannel(grp, name)
            elif kind == "event":
                self.channels[name] = EventChannel(grp, name)

    def __getitem__(self, key: str) -> _Channel:
        """Return the channel group named ``key``."""
        return self.channels[key]

    def __iter__(self):  # noqa: ANN204 - iterator convenience
        """Iterate over channel keys."""
        return iter(self.channels)

    @property
    def continuous(self) -> dict[str, ContinuousChannel]:
        """Continuous channels by key."""
        return {
            k: c
            for k, c in self.channels.items()
            if isinstance(c, ContinuousChannel)
        }

    @property
    def events(self) -> dict[str, EventChannel]:
        """Event channels by key."""
        return {
            k: c
            for k, c in self.channels.items()
            if isinstance(c, EventChannel)
        }

    @property
    def meta(self) -> dict[str, Any] | None:
        """The ``meta/`` attributes, or ``None`` when de-identified.

        Fetched directly, never via the consolidated block, from
        which the spec excludes it.
        """
        try:
            grp = zarr.open_group(
                store=self.root.store, path="meta", mode="r",
                use_consolidated=False,
            )
        except FileNotFoundError:
            return None
        return dict(grp.attrs)

    @property
    def start_us(self) -> int | None:
        """Wall-clock onset from ``meta/session.start_us``, if present."""
        meta = self.meta
        if meta is None:
            return None
        start = meta.get("session", {}).get("start_us")
        return int(start) if start is not None else None

    def montage(
        self, weights: dict[str, float]
    ) -> "Montage":  # noqa: F821 - forward ref
        """Build a linear re-reference over continuous channels.

        ``weights`` maps channel key -> coefficient, e.g.
        ``{"0": 1.0, "1": -1.0}`` for a bipolar pair.
        """
        from timeseries_zarr.reading.montage import (  # noqa: PLC0415 - import cycle
            Montage,
        )

        return Montage(self, weights)


def open_bundle(path: str | Path) -> Bundle:
    """Open a bundle from a local path or a store URL."""
    root = zarr.open_group(str(path), mode="r")
    return Bundle(root, path)
