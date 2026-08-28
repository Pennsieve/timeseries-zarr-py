"""Build a compact, spec-complete bundle for reading-layer tests.

A trimmed sibling of the paper repo's ``generate_test_bundle.py``:
120 s instead of 600, bin threshold 64 so small signals still grow
several levels. Channels:

* ``0`` continuous 512 Hz (montage input A)
* ``1`` continuous 512 Hz, offset_uv=1000, correlated with 0
* ``2`` continuous 256 Hz with a NaN gap, ssq + valid
* ``3`` event: annotations with durations/labels/text bodies
* ``4`` event: 3000 detections, values + JSON bodies + count levels
* ``5`` continuous 512 Hz whose offset_us is not bin-aligned
* ``6`` continuous 512 Hz viewing-only (no raw)
"""

import json
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

F = 4
BIN_THRESHOLD = 64
DUR_S = 120.0
START_US = 1_700_000_000_000_000


def _write_array(group, name, data, attributes=None, plain=False):
    if plain:
        arr = group.create_array(
            name, shape=data.shape, dtype=data.dtype,
            chunks=data.shape, compressors=(),
            attributes=attributes or {},
        )
    else:
        chunks = (min(8192, max(1, data.shape[0])), *data.shape[1:])
        arr = group.create_array(
            name, shape=data.shape, dtype=data.dtype,
            chunks=chunks, compressors=ZstdCodec(level=3),
            attributes=attributes or {},
        )
    arr[:] = data
    return arr


def _base_stats(raw):
    n = (len(raw) // F) * F
    blocks = raw[:n].reshape(-1, F)
    mn, mx = blocks.min(axis=1), blocks.max(axis=1)
    sm = blocks.sum(axis=1)
    sq = (blocks.astype(np.float64) ** 2).sum(axis=1)
    ct = np.isfinite(blocks).sum(axis=1).astype(np.int64)
    if len(raw) > n:
        tail = raw[n:]
        mn = np.append(mn, tail.min())
        mx = np.append(mx, tail.max())
        sm = np.append(sm, tail.sum())
        sq = np.append(sq, (tail.astype(np.float64) ** 2).sum())
        ct = np.append(ct, np.isfinite(tail).sum())
    return mn, mx, sm, sq, ct


def _fold(stats):
    mn, mx, sm, sq, ct = stats
    n = (len(mn) // F) * F
    out = []
    for a, red in ((mn, np.min), (mx, np.max), (sm, np.sum),
                   (sq, np.sum), (ct, np.sum)):
        head = red(a[:n].reshape(-1, F), axis=1)
        if len(a) > n:
            head = np.append(head, red(a[n:]))
        out.append(head)
    return tuple(out)


def _write_continuous(root, key, raw, rate_hz, *, offset_uv=None,
                      offset_us=0, with_ssq=False, write_raw=True):
    attrs = {"id": f"fix-{key}", "kind": "continuous",
             "name": f"ch{key}", "offset_us": int(offset_us),
             "rate_hz": float(rate_hz), "unit": "uV"}
    if offset_uv is not None:
        attrs["offset_uv"] = float(offset_uv)
    ch = root.create_group(key, attributes=attrs)
    if write_raw:
        _write_array(ch, "raw", raw.astype(np.float32))
    folded = raw.astype(np.float64)
    if offset_uv is not None:
        folded = folded - offset_uv
    has_gaps = not np.all(np.isfinite(folded))
    stats = _base_stats(folded)
    period_us, k = 1e6 / rate_hz * F, 1
    while len(stats[0]) >= BIN_THRESHOLD or k == 1:
        mn, mx, sm, sq, ct = stats
        lvl = ch.create_group(str(k), attributes={"period_us": period_us})
        _write_array(lvl, "env",
                     np.stack([mn, mx], 1).astype(np.float32))
        with np.errstate(invalid="ignore"):
            mean = np.where(ct > 0, sm / np.maximum(ct, 1), np.nan)
        _write_array(lvl, "mean", mean.astype(np.float32))
        if with_ssq:
            with np.errstate(invalid="ignore"):
                ssq = np.where(ct > 0, sq / np.maximum(ct, 1), np.nan)
            _write_array(lvl, "ssq", ssq.astype(np.float32))
        if has_gaps:
            _write_array(lvl, "valid", ct.astype(np.uint16))
        nxt = _fold(stats)
        if len(nxt[0]) < BIN_THRESHOLD:
            break
        stats, period_us, k = nxt, period_us * F, k + 1
    return ch


def _bodies(payloads):
    blobs = [(s.rstrip("\n") + "\n").encode() for s in payloads]
    bodies = np.frombuffer(b"".join(blobs), dtype=np.uint8).copy()
    offs = np.zeros(len(blobs) + 1, dtype=np.uint64)
    np.cumsum([len(b) for b in blobs], out=offs[1:])
    return bodies, offs


def build_bundle(out: Path) -> Path:  # noqa: PLR0915
    """Write the fixture bundle at ``out`` and return it."""
    rng = np.random.default_rng(7)
    store = zarr.storage.LocalStore(out)
    root = zarr.create_group(store=store, zarr_format=3)

    fs = 512.0
    n = int(fs * DUR_S)
    t = np.arange(n) / fs
    common = np.cumsum(rng.normal(0, 0.5, n))
    x0 = 20 * np.sin(2 * np.pi * 1.5 * t) + common + rng.normal(0, 5, n)
    x1 = 1000.0 + 10 * np.sin(2 * np.pi * 0.7 * t) + common \
        + rng.normal(0, 5, n)
    _write_continuous(root, "0", x0, fs)
    _write_continuous(root, "1", x1, fs, offset_uv=1000.0)

    fs2 = 256.0
    n2 = int(fs2 * DUR_S)
    x2 = 15 * np.sin(2 * np.pi * 3 * np.arange(n2) / fs2) \
        + rng.normal(0, 3, n2)
    x2[int(40 * fs2):int(42 * fs2)] = np.nan
    _write_continuous(root, "2", x2, fs2, with_ssq=True)

    _write_continuous(root, "5", x0[: n // 2], fs, offset_us=3000)
    _write_continuous(root, "6", x0, fs, write_raw=False)

    marks = [
        (10.0, 2.0, 1, "spike run, reviewed"),
        (30.0, 45.0, 0, "long seizure, evolving"),
        (100.0, 0.5, 2, "note"),
    ]
    m_ts = np.array([int(s * 1e6) for s, *_ in marks], dtype=np.int64)
    m_dur = np.array([int(d * 1e6) for _, d, *_ in marks],
                     dtype=np.int64)
    m_lab = np.array([lab for *_, lab, _ in marks], dtype=np.uint16)
    bodies, offs = _bodies([m[3] for m in marks])
    ch3 = root.create_group("3", attributes={
        "id": "fix-3", "kind": "event", "name": "marks",
        "offset_us": 0, "body_media_type": "text/plain",
        "label_names": ["seizure", "spike", "note"],
        "max_duration_us": int(m_dur.max())})
    _write_array(ch3, "events", m_ts)
    _write_array(ch3, "durations", m_dur)
    _write_array(ch3, "labels", m_lab)
    _write_array(ch3, "bodies", bodies, plain=True)
    _write_array(ch3, "body_offsets", offs)

    n_det = 3000
    d_ts = np.sort(rng.uniform(0, DUR_S, n_det))
    d_val = rng.beta(2, 5, n_det).astype(np.float32)
    d_lab = (rng.uniform(size=n_det) < 0.3).astype(np.uint16)
    payloads = [json.dumps({"score": round(float(v), 4)})
                for v in d_val]
    bodies4, offs4 = _bodies(payloads)
    ch4 = root.create_group("4", attributes={
        "id": "fix-4", "kind": "event", "name": "detections",
        "offset_us": 0, "unit": "a.u.",
        "body_media_type": "application/json",
        "label_names": ["ripple", "fast-ripple"]})
    d_us = (d_ts * 1e6).astype(np.int64)
    _write_array(ch4, "events", d_us)
    _write_array(ch4, "labels", d_lab)
    _write_array(ch4, "values", d_val)
    _write_array(ch4, "bodies", bodies4, plain=True)
    _write_array(ch4, "body_offsets", offs4)
    base = 100_000
    nbins = int(np.ceil(DUR_S * 1e6 / base))
    idx = np.minimum(d_us // base, nbins - 1)
    counts = np.zeros((nbins, 2), dtype=np.int64)
    np.add.at(counts, (idx, d_lab.astype(np.int64)), 1)
    period, k = float(base), 1
    while counts.shape[0] >= BIN_THRESHOLD or k == 1:
        lvl = ch4.create_group(str(k), attributes={"period_us": period})
        _write_array(lvl, "counts", counts.astype(np.uint32))
        m = (counts.shape[0] // F) * F
        nxt = counts[:m].reshape(-1, F, 2).sum(axis=1)
        if counts.shape[0] > m:
            nxt = np.vstack([nxt, counts[m:].sum(axis=0)])
        if nxt.shape[0] < BIN_THRESHOLD:
            break
        counts, period, k = nxt, period * F, k + 1

    root.create_group("meta", attributes={
        "subject": {"subject_id": "sub-fixture"},
        "session": {"session_id": "ses-01", "start_us": START_US},
        "source": {"generator": "bundle_fixture.py"},
    })
    zarr.consolidate_metadata(store)
    zj = out / "zarr.json"
    doc = json.loads(zj.read_text())
    cm = doc.get("consolidated_metadata", {}).get("metadata", {})
    for key in [k for k in cm
                if k == "meta" or k.startswith("meta/")]:
        del cm[key]
    zj.write_text(json.dumps(doc, indent=1))
    return out
