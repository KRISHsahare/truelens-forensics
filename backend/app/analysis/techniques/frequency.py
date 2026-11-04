"""Frequency-domain analysis.

Natural photographs follow an approximate ``1/f^alpha`` amplitude power law. Generative models
reproduce the low frequencies well and systematically mis-model the tail, so the *shape* of the
decay is a usable signal. Three things are measured:

1. **High-band decay slope.** The primary signal. Fitted on the upper half of the radial band, so
   it describes the tail alone rather than being dominated by the enormous low-frequency energy.
   A generator that has smoothed away fine detail produces a tail that is nearly *flat* — there
   is nothing left to decay. Measured across the fixture set: generated ``-0.02``, natural camera
   content ``-0.65`` to ``-0.94``, JPEG-compressed content ``-1.4`` to ``-1.9``.
2. **Full-band slope.** A weaker, two-sided check against the natural-image range.
3. **Periodic peaks along the axes.** Upsampling by transposed convolution — every GAN generator,
   and the VAE decoder in most diffusion models — multiplies the signal by a fixed kernel on a
   regular grid. That is a periodic impulse train in the spatial domain and a *peak* at a fixed
   radius in the spectrum, energy a camera cannot produce. Scanned along the horizontal and
   vertical axes through DC, where an axis-aligned checkerboard lands.

The visual is the raw log-magnitude spectrum, not a composite: its axes *are* the frequency
domain, so overlaying it on the image would be meaningless. The UI labels it as such.

Two calibration traps this module is written to avoid
----------------------------------------------------

* **Do not feed this a resampled image.** Running the FFT on a 256x256 downscale is what the
  prototype did, and it destroys the signal. Area resampling from 1024 to 256 imposes its own
  low-pass and re-creates a power-law tail out of whatever it averaged, collapsing the high-band
  slope from ``-0.02`` / ``-0.79`` (separated) to ``-1.32`` / ``-1.26`` (indistinguishable). The
  same thing happens more gently at any downscale, so this analyser runs on the **native** track
  and centre-crops to the largest power of two that fits ``MAX_FFT``. It never resamples.
  Consequence: on a very large upload only the centre of the frame is examined. That is the right
  trade — spectral decay is a global property and a centre crop estimates it well, whereas
  resampling actively fabricates the artifact being tested for.
* **Do not treat a steep tail as evidence of generation.** Heavy JPEG compression, aggressive
  resizing and screenshots all deplete high frequencies too, which pushes the slope *steeper*,
  not flatter. That condition is reported as ``high_frequency_depleted`` and routed to the
  reliability score, because it means the evidence has been destroyed — not that the image is
  synthetic.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.analysis.contract import TechniqueResult, clip01
from app.analysis.techniques._common import colormap_rgb

__all__ = ["run"]

#: Largest FFT edge. 1024 costs ~30-50 ms and preserves the tail; anything smaller does not.
MAX_FFT = 1024
#: Natural-image full-band amplitude slope range.
NATURAL_SLOPE_RANGE = (-1.60, -0.85)
#: High-band slope above which the tail is flat enough to be a generative signature.
SMOOTH_TAIL_SLOPE = -0.35
#: High-band slope below which the tail is depleted. Reliability signal, not AI evidence.
DEPLETED_TAIL_SLOPE = -1.30
#: Axis scan window, as fractions of Nyquist. The outer edge is excluded: bins there are sparse,
#: the Hann window's own rolloff lands there, and JPEG blocking puts real energy at Nyquist, so a
#: peak at r ~ 1.0 is not evidence of anything.
AXIS_LO, AXIS_HI = 0.08, 0.92


def run(native_gray_u8: np.ndarray, *, max_fft: int = MAX_FFT) -> TechniqueResult:
    """``native_gray_u8`` is the never-resampled native-track grayscale."""
    gray = (native_gray_u8.astype(np.float32) if native_gray_u8.ndim == 2
            else cv2.cvtColor(native_gray_u8, cv2.COLOR_RGB2GRAY).astype(np.float32))
    h, w = gray.shape

    side = 2 ** int(np.log2(max(min(h, w, max_fft), 2)))
    if side < 128:
        return TechniqueResult(0.0, None, {
            "note": "Image too small for meaningful spectral analysis.",
            "min_side": int(min(h, w)),
        })
    # Centre crop, never a resize: resampling rewrites the very tail we are measuring.
    y0, x0 = (h - side) // 2, (w - side) // 2
    tile = gray[y0:y0 + side, x0:x0 + side]

    n = tile.shape[0]
    # Hanning window: a rectangular window leaks the image edges into the spectrum as a cross,
    # which this analyser would then read as a periodic artifact.
    win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(tile * win))
    amp = np.abs(f).astype(np.float32)
    mag = np.log1p(amp).astype(np.float32)

    c = max_r = n // 2
    yy, xx = np.indices(mag.shape)
    radius = np.hypot(yy - c, xx - c).astype(np.int32)
    counts = np.bincount(radius.ravel(), minlength=max_r + 1)[: max_r + 1]
    # Radial average on LINEAR amplitude. Averaging log1p(|F|) and then fitting a log-log slope
    # measures log(mean(log|F|)), which is not the 1/f^alpha exponent at all and compresses every
    # image to a slope near -0.17 regardless of content.
    radial = (np.bincount(radius.ravel(), weights=amp.ravel(), minlength=max_r + 1)[: max_r + 1]
              / np.maximum(counts, 1))
    radial_log = (np.bincount(radius.ravel(), weights=mag.ravel(), minlength=max_r + 1)[: max_r + 1]
                  / np.maximum(counts, 1))

    lo, hi = 4, int(max_r * 0.92)
    fit = np.polyfit(np.log(np.arange(lo, hi, dtype=np.float64)),
                     np.log(np.maximum(radial[lo:hi], 1e-9)), 1)
    slope = float(fit[0])

    half = lo + (hi - lo) // 2
    hi_slope = float(np.polyfit(np.log(np.arange(half, hi, dtype=np.float64)),
                                np.log(np.maximum(radial[half:hi], 1e-9)), 1)[0])

    # --- 1. flat high-frequency tail: the generative signature -----------------
    smooth_tail = hi_slope > SMOOTH_TAIL_SLOPE
    smooth_score = clip01((hi_slope - SMOOTH_TAIL_SLOPE) / 0.35) if smooth_tail else 0.0

    # --- 2. full-band slope, two-sided ----------------------------------------
    slope_lo, slope_hi = NATURAL_SLOPE_RANGE
    if slope < slope_lo:
        slope_dev = (slope_lo - slope) / abs(slope_lo)
    elif slope > slope_hi:
        slope_dev = (slope - slope_hi) / abs(slope_hi)
    else:
        slope_dev = 0.0
    slope_score = clip01(slope_dev / 0.45)

    # --- 3. periodic peaks along the axes -------------------------------------
    axis_peak = _axis_periodicity(mag, c, max_r)

    # --- 4. residual spikes in the high band ----------------------------------
    high = radial_log[int(max_r * 0.62): int(max_r * 0.94)]
    if high.size > 8:
        idx = np.arange(high.size, dtype=np.float64)
        detrended = high - np.polyval(np.polyfit(idx, high, 2), idx)
        spike = float(np.abs(detrended).max() / (high.std() + 1e-6))
    else:
        spike = 0.0
    spike_score = clip01(max(0.0, spike - 2.5) / 3.0)

    score = clip01(0.46 * smooth_score + 0.14 * slope_score
                   + 0.24 * axis_peak["score"] + 0.16 * spike_score)

    depleted = hi_slope < DEPLETED_TAIL_SLOPE
    visual = colormap_rgb(normalise_for_display(mag), cv2.COLORMAP_MAGMA)

    details = {
        "high_band_slope": round(hi_slope, 4),
        "smooth_tail_threshold": SMOOTH_TAIL_SLOPE,
        "smooth_tail": smooth_tail,
        "smooth_tail_score": round(smooth_score, 4),
        "spectral_slope": round(slope, 4),
        "natural_slope_range": list(NATURAL_SLOPE_RANGE),
        "slope_deviation": round(slope_dev, 4),
        "slope_score": round(slope_score, 4),
        "axis_periodicity": {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in axis_peak.items()},
        "high_band_spike": round(spike, 3),
        "spike_score": round(spike_score, 4),
        "high_frequency_depleted": depleted,
        "depleted_threshold": DEPLETED_TAIL_SLOPE,
        "fft_size": int(n),
        "centre_crop_of": [int(h), int(w)],
        "window": "hann",
        "note": "Visual is the log-magnitude spectrum (centre = DC), not an image overlay.",
    }
    if depleted:
        details["depletion_note"] = (
            "The high-frequency tail falls off much faster than a camera-produced image would "
            f"(slope {hi_slope:.2f}). Heavy JPEG compression, aggressive resizing and screenshots all do "
            "this. It is not evidence that the image is synthetic — it means the fine-detail "
            "traces this analyser relies on have been largely destroyed, so the result is less "
            "trustworthy and the reliability score is reduced."
        )
    if smooth_tail:
        details["smooth_tail_note"] = (
            f"Above roughly 0.6 of Nyquist the amplitude spectrum is nearly flat (slope {hi_slope:.2f}) "
            "instead of continuing to fall. Camera optics and sensor noise always leave a "
            "decaying tail; a model that reconstructs coarse structure well but cannot produce "
            "fine detail leaves this one flat."
        )
    return TechniqueResult(score, visual, details)


def _axis_periodicity(mag: np.ndarray, c: int, max_r: int) -> dict:
    """Score periodic energy along the horizontal and vertical axes through DC."""
    lo = max(3, int(max_r * AXIS_LO))
    hi = int(max_r * AXIS_HI)
    if hi <= lo + 8:
        return {"score": 0.0, "peak_z": 0.0, "peak_radius": None,
                "peak_as_fraction_of_nyquist": None, "harmonic_confirmed": 0.0,
                "scan_band": [AXIS_LO, AXIS_HI], "note": "band too narrow"}

    profile = (mag[c, lo:hi] + mag[lo:hi, c]) / 2.0
    # Remove the smooth 1/f decay so only *excess* periodic energy remains.
    idx = np.arange(profile.size, dtype=np.float64)
    resid = profile - np.polyval(np.polyfit(idx, profile, 3), idx)
    z = resid / (float(resid.std()) + 1e-6)
    peak_idx = int(np.argmax(np.abs(z)))
    peak_z = float(abs(z[peak_idx]))
    peak_radius = int(peak_idx + lo)

    # Confirm periodicity: a real impulse train repeats at sub-harmonics, a single lucky bin does
    # not. Both half and double the peak radius must show excess energy inside the scan band.
    confirm = 0.0
    for frac in (0.5, 2.0):
        target = peak_radius * frac
        if lo < target < hi - 1:
            j = int(target) - lo
            window = z[max(0, j - 2): j + 3]
            if window.size and abs(window).max() > 2.0:
                confirm += 0.5

    score = clip01(max(0.0, (peak_z - 3.5) / 3.0) * (0.6 + 0.4 * min(1.0, confirm)))
    return {
        "score": score,
        "peak_z": round(peak_z, 3),
        "peak_radius": peak_radius,
        "peak_as_fraction_of_nyquist": round(peak_radius / max(1, max_r), 3),
        "harmonic_confirmed": round(confirm, 2),
        "scan_band": [AXIS_LO, AXIS_HI],
    }


def normalise_for_display(mag: np.ndarray) -> np.ndarray:
    """Percentile-normalise for display. A full min-max stretch lets the DC spike wash out
    every other detail in the spectrum."""
    lo, hi = np.percentile(mag, 0.5), np.percentile(mag, 99.9)
    if hi - lo < 1e-6:
        return np.zeros_like(mag, dtype=np.float32)
    return np.clip((mag - lo) / (hi - lo), 0, 1).astype(np.float32)
