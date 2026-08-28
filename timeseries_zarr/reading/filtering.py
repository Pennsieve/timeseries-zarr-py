"""Filtered views from decimated mean levels, under the rho rule.

The reference implementation the paper's tier measurements assume:
a causal Butterworth redesigned at the serving level's rate. Level
selection walks coarse-to-fine until the rho tier admits the level
(silent or marked); if no level is admissible the request is refused
and falls back to raw when raw is stored (a viewing-only bundle
raises). Seeks are warmed up with real pre-roll sized by the
pole-decay law: n = ln(1/eps) / (-ln |p_max|) samples; where no
earlier data exists, the fetch-free fallback is tiled reflection of a
~50 ms block (the best-measured fallback).

Requires scipy (install extra: ``timeseries-zarr-py[serving]``).
"""

import math

import numpy as np

from timeseries_zarr.reading.bundle import (
    ContinuousChannel,
    Level,
    Window,
)
from timeseries_zarr.reading.serving import (
    PREROLL_EPSILON,
    REFLECTION_BLOCK_S,
    FilterSpec,
    ServingDecision,
    ServingError,
    Tier,
    classify,
)

US = 1_000_000.0


def _scipy_signal():  # noqa: ANN202 - lazy optional import
    try:
        from scipy import signal  # noqa: PLC0415 - optional dep
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "filtered views need scipy; install "
            "timeseries-zarr-py[serving]"
        ) from err
    return signal


def decide_level(
    channel: ContinuousChannel,
    spec: FilterSpec,
    t0: float,
    t1: float,
    pixels: int,
    *,
    broadband: bool = False,
) -> tuple[Level | None, ServingDecision]:
    """Pick the serving level for a filtered view.

    Returns ``(level, decision)``. ``level`` is ``None`` when the
    request must read raw -- either refused at every stored level, or
    the window is too short for any level's resolution.
    """
    window_us = (t1 - t0) * US
    candidates = [
        lv
        for lv in channel.levels
        if window_us / lv.period_us >= pixels
    ]
    for lv in reversed(candidates):  # coarsest admissible first
        rho, tier = classify(spec, lv.rate_hz, broadband=broadband)
        if tier is not Tier.REFUSE:
            return lv, ServingDecision(lv.k, lv.rate_hz, rho, tier)
    rho, _ = classify(spec, channel.rate_hz, broadband=broadband)
    return None, ServingDecision(0, channel.rate_hz, rho, Tier.REFUSE)


def preroll_samples(sos: np.ndarray, epsilon: float) -> int:
    """Pre-roll length from the pole-decay law (paper, warm-up thm)."""
    signal = _scipy_signal()
    _, poles, _ = signal.sos2zpk(sos)
    if len(poles) == 0:  # pragma: no cover - FIR corner
        return 0
    p_max = float(np.max(np.abs(poles)))
    if p_max >= 1.0:  # pragma: no cover - unstable design
        raise ServingError("filter design is unstable at this rate")
    if p_max == 0.0:
        return 0
    return int(math.ceil(math.log(1.0 / epsilon) / -math.log(p_max)))


def _tiled_reflection_pad(x: np.ndarray, need: int, rate_hz: float) -> np.ndarray:
    """Fetch-free pad preceding ``x``: tiled reflection of a block."""
    block_len = max(1, int(round(REFLECTION_BLOCK_S * rate_hz)))
    block_len = min(block_len, len(x))
    block = x[:block_len]
    tiles: list[np.ndarray] = []
    total = 0
    flip = True  # nearest tile mirrors about the boundary
    while total < need:
        tiles.insert(0, block[::-1] if flip else block)
        total += block_len
        flip = not flip
    pad = np.concatenate(tiles)
    return pad[-need:]


def _design(spec: FilterSpec, rate_hz: float) -> np.ndarray:
    signal = _scipy_signal()
    if spec.btype == "bandpass":
        wn: float | list[float] = [float(spec.low_hz), float(spec.high_hz)]
    elif spec.btype == "lowpass":
        wn = float(spec.high_hz)
    else:
        wn = float(spec.low_hz)
    return signal.butter(
        spec.order, wn, btype=spec.btype, fs=rate_hz, output="sos"
    )


def _fill_gaps(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate NaN gaps for the filter; return (filled, gap mask).

    IIR state cannot cross a NaN, so gaps are linearly interpolated
    for filtering and reported back so clients can blank them.
    """
    gaps = ~np.isfinite(x)
    if not gaps.any():
        return x, gaps
    filled = x.copy()
    idx = np.arange(len(x))
    good = ~gaps
    if not good.any():
        raise ServingError("window contains no finite samples")
    filled[gaps] = np.interp(idx[gaps], idx[good], x[good])
    return filled, gaps


def filtered_window(
    channel: ContinuousChannel,
    t0: float,
    t1: float,
    *,
    low_hz: float | None,
    high_hz: float | None,
    pixels: int = 2000,
    order: int = 4,
    broadband: bool = False,
    epsilon: float = PREROLL_EPSILON,
) -> Window:
    """Serve a filtered view of ``[t0, t1)`` under the rho rule.

    The returned window's ``mean`` holds the filtered trace; its
    ``marked`` flag is True in the indicator tier. Raises
    :class:`ServingError` when every stored level refuses and the
    bundle is viewing-only.
    """
    spec = FilterSpec(low_hz, high_hz, order)
    level, decision = decide_level(
        channel, spec, t0, t1, pixels, broadband=broadband
    )

    if level is None:
        if not channel.has_raw:
            raise ServingError(
                f"rho={decision.rho:.2f} > 0.5 at every stored level "
                f"of viewing-only channel {channel.key!r}; the "
                f"requested band needs raw"
            )
        rate = channel.rate_hz
        sos = _design(spec, rate)
        pre = preroll_samples(sos, epsilon)
        seek0 = max(channel.offset_us / US, t0 - pre / rate)
        win = channel.raw_window(seek0, t1)
        assert win.raw is not None
        x, gaps = _fill_gaps(win.raw)
        y = _scipy_signal().sosfilt(sos, x)
        y[gaps] = np.nan
        keep = win.t >= t0
        return Window(
            level=0,
            period_s=1.0 / rate,
            t=win.t[keep],
            mean=y[keep],
            marked=True,
            note=f"served from raw (rho={decision.rho:.2f} refused "
            f"all levels)",
        )

    sos = _design(spec, level.rate_hz)
    pre_bins = preroll_samples(sos, epsilon)
    i0, i1 = channel._bin_slice(level, t0, t1)
    j0 = max(0, i0 - pre_bins)
    x = channel.mean_series(level, j0, i1)
    x, gaps = _fill_gaps(x)
    short = pre_bins - (i0 - j0)
    if short > 0 and len(x) > 0:
        x = np.concatenate(
            [_tiled_reflection_pad(x, short, level.rate_hz), x]
        )
        gaps = np.concatenate([np.zeros(short, dtype=bool), gaps])
    y = _scipy_signal().sosfilt(sos, x)
    y[gaps] = np.nan
    n_pre = len(x) - (i1 - i0)
    centers = (
        channel.offset_us + (np.arange(i0, i1) + 0.5) * level.period_us
    ) / US
    return Window(
        level=level.k,
        period_s=level.period_us / US,
        t=centers,
        mean=y[n_pre:],
        marked=decision.tier is Tier.MARKED,
        note=f"rho={decision.rho:.2f} ({decision.tier.value})",
    )
