"""Linear re-referencing from stored mean levels.

Every fixed linear re-reference (bipolar, common average, linked
ears, Laplacian) is *exact* at every pyramid level (paper, Theorem:
linearity), provided the input channels share a rate and their bins
align. The montaged envelope is unrecoverable from per-channel
statistics; what this module offers instead is the provably
conservative interval band computed from per-channel ``env`` members
(containment guaranteed, ~2x true width at median).
"""

import numpy as np

from timeseries_zarr.reading.bundle import (
    Bundle,
    ContinuousChannel,
    Window,
)
from timeseries_zarr.reading.serving import ServingError

US = 1_000_000.0


class MontageAlignmentError(ServingError):
    """Channels cannot be montaged exactly from stored levels."""


class Montage:
    """A weighted sum of continuous channels, served from levels."""

    def __init__(self, bundle: Bundle, weights: dict[str, float]) -> None:
        """Resolve channels and verify a shared sampling rate."""
        self.bundle = bundle
        self.weights = dict(weights)
        chans: list[ContinuousChannel] = []
        for key in self.weights:
            ch = bundle[key]
            if not isinstance(ch, ContinuousChannel):
                raise MontageAlignmentError(
                    f"channel {key!r} is not continuous"
                )
            chans.append(ch)
        rates = {c.rate_hz for c in chans}
        if len(rates) != 1:
            raise MontageAlignmentError(
                f"cross-rate montage must read raw: rates {sorted(rates)}"
            )
        self.channels = chans
        self.rate_hz = rates.pop()

    def _aligned_levels(self, k: int) -> list:
        levels = []
        base = None
        for ch in self.channels:
            match = [lv for lv in ch.levels if lv.k == k]
            if not match:
                raise MontageAlignmentError(
                    f"channel {ch.key!r} has no level {k}"
                )
            lv = match[0]
            if base is None:
                base = lv.period_us
            elif lv.period_us != base:
                raise MontageAlignmentError(
                    f"level {k} period mismatch on {ch.key!r}"
                )
            # bins align only when channel offsets are congruent
            # modulo the bin period at this level
            if (
                ch.offset_us - self.channels[0].offset_us
            ) % lv.period_us != 0:
                raise MontageAlignmentError(
                    f"channel {ch.key!r} offset is not bin-aligned "
                    f"with {self.channels[0].key!r} at level {k}; "
                    f"read raw for this pairing"
                )
            levels.append(lv)
        return levels

    def window(
        self,
        t0: float,
        t1: float,
        pixels: int = 2000,
        *,
        with_band: bool = False,
    ) -> Window:
        """Serve the exact montaged mean over ``[t0, t1)``.

        With ``with_band=True`` the window's ``env`` carries the
        conservative interval band from per-channel envelopes:
        guaranteed to contain the true montaged envelope, roughly 2x
        its width at median. Off by default, matching the paper's
        rendering policy.
        """
        first = self.channels[0]
        lv0 = first.pick_level(t0, t1, pixels)
        if lv0 is None:
            return self._raw_window(t0, t1)
        levels = self._aligned_levels(lv0.k)
        i0, i1 = first._bin_slice(levels[0], t0, t1)
        n = i1 - i0
        mean = np.zeros(n, dtype=np.float64)
        lo = np.zeros(n, dtype=np.float64)
        hi = np.zeros(n, dtype=np.float64)
        for ch, lv in zip(self.channels, levels, strict=True):
            w = self.weights[ch.key]
            shift = int(
                (first.offset_us - ch.offset_us) // lv.period_us
            )
            j0, j1 = i0 + shift, i1 + shift
            if j0 < 0 or j1 > lv.n_bins:
                raise MontageAlignmentError(
                    f"window not covered by channel {ch.key!r}"
                )
            m = ch.mean_series(lv, j0, j1)
            mean += w * m
            if with_band:
                env = (
                    lv.group["env"][j0:j1].astype(np.float64)
                    + ch.offset_uv
                )
                if w >= 0:
                    lo += w * env[:, 0]
                    hi += w * env[:, 1]
                else:
                    lo += w * env[:, 1]
                    hi += w * env[:, 0]
        centers = (
            first.offset_us + (np.arange(i0, i1) + 0.5) * levels[0].period_us
        ) / US
        env = np.stack([lo, hi], axis=1) if with_band else None
        return Window(
            level=levels[0].k,
            period_s=levels[0].period_us / US,
            t=centers,
            mean=mean,
            env=env,
        )

    def _raw_window(self, t0: float, t1: float) -> Window:
        first = self.channels[0]
        base = first.raw_window(t0, t1)
        assert base.raw is not None
        acc = self.weights[first.key] * base.raw
        for ch in self.channels[1:]:
            w = self.bundle[ch.key]
            assert isinstance(w, ContinuousChannel)
            other = w.raw_window(t0, t1)
            assert other.raw is not None
            n = min(len(acc), len(other.raw))
            acc = acc[:n] + self.weights[ch.key] * other.raw[:n]
            base.t = base.t[:n]
        base.raw = acc
        base.mean = acc
        return base
