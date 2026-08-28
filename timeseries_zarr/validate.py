"""Spec-conformance validator for TS-Zarr bundles.

Checks structure, the attribute vocabulary, fold exactness (spot
checks against raw where raw is stored), event-column invariants,
and the PHI-surface rules (meta/ excluded from consolidated
metadata; no wall-clock key outside meta/). Returns findings rather
than raising, so producers can run it in CI.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

FOLD = 4
ZARR_FORMAT = 3
ENV_COLS = 2
PERIOD_RTOL = 1e-9
CHANNEL_KEYS = {
    "id", "kind", "name", "offset_us", "rate_hz", "unit",
    "offset_uv", "body_media_type", "label_names", "max_duration_us",
}
LEVEL_KEYS = {"period_us"}
SPOT_SAMPLES = 4096


@dataclass(frozen=True)
class Finding:
    """One validator finding."""

    severity: str  # "error" | "warning" | "info"
    where: str
    message: str

    def __str__(self) -> str:
        """Render as one report line."""
        return f"[{self.severity.upper():7s}] {self.where}: {self.message}"


class _Report:
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def error(self, where: str, msg: str) -> None:
        self.findings.append(Finding("error", where, msg))

    def warn(self, where: str, msg: str) -> None:
        self.findings.append(Finding("warning", where, msg))

    def info(self, where: str, msg: str) -> None:
        self.findings.append(Finding("info", where, msg))

    def check(self, cond: bool, where: str, msg: str) -> bool:  # noqa: ANN001
        if not cond:
            self.error(where, msg)
        return bool(cond)


def validate_bundle(path: str | Path) -> list[Finding]:
    """Validate the bundle at ``path``; return all findings."""
    path = Path(path)
    rep = _Report()
    root = zarr.open_group(str(path), mode="r")

    _check_root(path, rep)
    for name in sorted(root.group_keys()):
        if name == "meta":
            continue
        _check_channel(root[name], name, rep)
    _check_phi_surface(path, rep)
    return rep.findings


def errors(findings: list[Finding]) -> list[Finding]:
    """Only the error-severity findings."""
    return [f for f in findings if f.severity == "error"]


def _check_root(path: Path, rep: _Report) -> None:
    doc = json.loads((path / "zarr.json").read_text())
    rep.check(
        doc.get("zarr_format") == ZARR_FORMAT, "/", "root must be Zarr v3"
    )
    cm = doc.get("consolidated_metadata")
    if cm is None:
        rep.warn("/", "no consolidated metadata (one fetch becomes many)")
        return
    keys = cm.get("metadata", {})
    rep.check(
        not any(k == "meta" or k.startswith("meta/") for k in keys),
        "/",
        "meta/ must be excluded from consolidated metadata "
        "(PHI-free hot path)",
    )


def _check_channel(grp: zarr.Group, name: str, rep: _Report) -> None:
    attrs = dict(grp.attrs)
    for req in ("id", "kind", "name", "offset_us"):
        if not rep.check(req in attrs, name, f"missing attribute {req!r}"):
            return
    kind = attrs["kind"]
    if not rep.check(
        kind in ("continuous", "event"), name, f"unknown kind {kind!r}"
    ):
        return
    rep.check(
        int(attrs["offset_us"]) >= 0, name, "offset_us must be >= 0"
    )
    for key in attrs:
        if key not in CHANNEL_KEYS:
            rep.warn(name, f"attribute {key!r} is outside the vocabulary")
    if kind == "continuous":
        rep.check("rate_hz" in attrs, name, "continuous needs rate_hz")
        _check_continuous(grp, name, rep)
    else:
        rep.check(
            "rate_hz" not in attrs, name, "rate_hz is continuous-only"
        )
        _check_event(grp, name, rep)


def _levels_of(grp: zarr.Group) -> list[tuple[int, zarr.Group]]:
    out = []
    for child in grp.group_keys():
        if child.isdigit():
            out.append((int(child), grp[child]))
    return sorted(out)


def _check_continuous(grp: zarr.Group, name: str, rep: _Report) -> None:
    attrs = dict(grp.attrs)
    rate = float(attrs["rate_hz"])
    has_raw = "raw" in grp.array_keys()
    levels = _levels_of(grp)
    if not has_raw:
        rep.info(name, "viewing-only channel (raw omitted)")
        rep.check(bool(levels), name, "viewing-only needs >= 1 level")
    prev_period = None
    for k, lvl in levels:
        where = f"{name}/{k}"
        lattrs = dict(lvl.attrs)
        if not rep.check(
            "period_us" in lattrs, where, "level needs period_us"
        ):
            continue
        period = float(lattrs["period_us"])
        if prev_period is not None:
            rep.check(
                abs(period / prev_period - FOLD) < PERIOD_RTOL,
                where,
                f"period must step 4x (got {period / prev_period:g}x)",
            )
        elif k == 1:
            expect = FOLD / rate * 1e6
            if abs(period - expect) > 1e-6 * expect:
                rep.warn(
                    where,
                    f"level-1 period {period:g}us != 4/rate "
                    f"({expect:g}us)",
                )
        prev_period = period
        if not rep.check(
            "env" in lvl.array_keys() and "mean" in lvl.array_keys(),
            where,
            "level needs env and mean members",
        ):
            continue
        env, mean = lvl["env"], lvl["mean"]
        rep.check(
            env.ndim == ENV_COLS and env.shape[1] == ENV_COLS,
            where,
            f"env must be (bins, 2), got {env.shape}",
        )
        rep.check(
            mean.shape == (env.shape[0],),
            where,
            "mean must share the env bin axis",
        )
        for opt in ("ssq", "valid"):
            if opt in lvl.array_keys():
                rep.check(
                    lvl[opt].shape == (env.shape[0],),
                    where,
                    f"{opt} must share the bin axis",
                )
    if has_raw and levels:
        _spot_check_fold(grp, name, levels[0], rep)


def _spot_check_fold(
    grp: zarr.Group,
    name: str,
    first_level: tuple[int, zarr.Group],
    rep: _Report,
) -> None:
    k, lvl = first_level
    w = FOLD**k
    n = min(SPOT_SAMPLES, grp["raw"].shape[0])
    nb = n // w
    if nb == 0:
        return
    raw = grp["raw"][: nb * w].astype(np.float64)
    raw -= float(dict(grp.attrs).get("offset_uv", 0.0))
    blocks = raw.reshape(nb, w)
    env = lvl["env"][:nb].astype(np.float64)
    mean = lvl["mean"][:nb].astype(np.float64)
    want_min = blocks.min(axis=1)
    want_max = blocks.max(axis=1)
    want_mean = blocks.mean(axis=1)  # NaN-propagating, like the spec
    tol = 1e-3 * max(1.0, float(np.nanmax(np.abs(raw))))

    def close(a: np.ndarray, b: np.ndarray) -> bool:
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            return False
        fin = ~np.isnan(a)
        return bool(np.all(np.abs(a[fin] - b[fin]) <= tol))

    rep.check(
        close(env[:, 0], want_min) and close(env[:, 1], want_max),
        f"{name}/{k}",
        "env fold does not match raw (spot check)",
    )
    rep.check(
        close(mean, want_mean),
        f"{name}/{k}",
        "mean fold does not match raw (spot check, NaN-propagating)",
    )


def _check_event(grp: zarr.Group, name: str, rep: _Report) -> None:
    attrs = dict(grp.attrs)
    if not rep.check(
        "events" in grp.array_keys(), name, "event channel needs events"
    ):
        return
    ev = grp["events"][:]
    rep.check(
        ev.dtype == np.int64, name, f"events must be int64, got {ev.dtype}"
    )
    rep.check(
        len(ev) == 0 or bool(np.all(np.diff(ev) >= 0)),
        name,
        "events must be non-decreasing",
    )
    rep.check(
        len(ev) == 0 or int(ev.min()) >= 0,
        name,
        "events must be onset-relative (>= 0)",
    )
    if "durations" in grp.array_keys() and rep.check(
        "max_duration_us" in attrs,
        name,
        "durations require max_duration_us",
    ):
        rep.check(
                int(grp["durations"][:].max())
                <= int(attrs["max_duration_us"]),
                name,
                "a duration exceeds max_duration_us",
            )
    if "bodies" in grp.array_keys():
        _check_bodies(grp, name, len(ev), rep)
    for k, lvl in _levels_of(grp):
        for member in lvl.array_keys():
            if member == "counts" or member.startswith("counts."):
                total = int(lvl[member][:].sum())
                rep.check(
                    total == len(ev),
                    f"{name}/{k}",
                    f"{member} sums to {total}, expected {len(ev)}",
                )


def _check_bodies(
    grp: zarr.Group, name: str, n_events: int, rep: _Report
) -> None:
    if not rep.check(
        "body_offsets" in grp.array_keys(),
        name,
        "bodies require body_offsets",
    ):
        return
    offs = grp["body_offsets"][:].astype(np.int64)
    bod = grp["bodies"][:]
    rep.check(
        len(offs) == n_events + 1,
        name,
        f"body_offsets must have n+1 entries "
        f"({len(offs)} for {n_events} events)",
    )
    rep.check(
        offs[0] == 0
        and offs[-1] == len(bod)
        and bool(np.all(np.diff(offs) >= 0)),
        name,
        "body_offsets must be monotone with matching endpoints",
    )
    text = bod.tobytes().decode("utf-8", errors="replace")
    rep.check(
        text == "" or text.endswith("\n"),
        name,
        "bodies must be newline-terminated (NDJSON/plain lines)",
    )
    lines = text.rstrip("\n").split("\n") if text else []
    if dict(grp.attrs).get("body_media_type") == "application/json":
        bad = 0
        for line in lines[:200]:
            try:
                json.loads(line)
            except json.JSONDecodeError:
                bad += 1
        rep.check(
            bad == 0, name, f"{bad} bodies are not valid JSON lines"
        )
    meta = grp["bodies"].metadata
    plain = (
        tuple(meta.chunk_grid.chunk_shape) == tuple(grp["bodies"].shape)
        and not _has_compressor(meta)
    )
    if not plain:
        rep.warn(
            name,
            "bodies should be one uncompressed chunk so the chunk "
            "object is the plain text file",
        )


def _has_compressor(meta) -> bool:  # noqa: ANN001 - zarr metadata
    for codec in meta.codecs:
        cname = getattr(codec, "codec_name", type(codec).__name__)
        if "zstd" in str(cname).lower() or "blosc" in str(cname).lower():
            return True
    return False


def _check_phi_surface(path: Path, rep: _Report) -> None:
    for zj in path.rglob("zarr.json"):
        if "meta" in zj.relative_to(path).parts:
            continue
        try:
            doc = json.loads(zj.read_text())
        except json.JSONDecodeError:
            continue
        text = json.dumps(doc.get("attributes", {}))
        where = str(zj.relative_to(path))
        if '"start_us"' in text:
            rep.error(
                where,
                "wall-clock key start_us outside meta/ "
                "(timeline must be onset-relative)",
            )
        if '"subject"' in text:
            rep.error(where, "subject metadata outside meta/")
