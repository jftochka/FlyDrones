"""Mastering: the difference between a render and a track somebody can play.

A render out of :mod:`flydrones.music.synth` is normalised to a peak, which is
the wrong thing to normalise to — peak says nothing about how loud a thing
sounds, and two pieces matched by peak differ by ten decibels by ear. This
module finishes the job the way a mastering engineer would:

1. **Measure loudness properly** — ITU-R BS.1770-4: K-weight the signal (a
   high-pass for the head and a high shelf for the ear), take the mean square
   over 400 ms blocks at 75% overlap, and gate twice, absolutely at -70 LUFS and
   then relative to the ungated mean, so that the silence between phrases is not
   averaged into the answer.
2. **Set the level to a target** rather than to a peak. -14 LUFS is what the
   streaming services normalise to, so a track delivered there is neither turned
   down on the way in nor left quieter than everything around it.
3. **Catch the peaks that are left** with a look-ahead limiter, and measure the
   result as a *true* peak — four times oversampled, because a signal can sit
   under 0 dBFS at every sample and still reconstruct above it between two of
   them, which is exactly what a lossy encoder then clips.
4. **Fade**, so the piece begins and ends rather than being cut out of a longer
   one, and write **24-bit** PCM, which has headroom to spare and needs no
   dither.

Everything is a pure function of an array, so the whole chain is measurable and
tested rather than trusted.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy import signal as sig

SR = 44100
TARGET_LUFS = -14.0      # what the streaming services normalise to
CEILING_DBTP = -1.0      # true-peak ceiling: leaves room for a lossy encoder


# ------------------------------------------------------------------ loudness

def _k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    """BS.1770 pre-filter: a +4 dB shelf above 1.5 kHz, then a high-pass at 38 Hz.

    The standard publishes the two biquads as coefficients at 48 kHz only. These
    are the analogue prototypes behind them, bilinear-transformed at whatever
    rate is in hand — which reproduces the published coefficients exactly at
    48 kHz (``tests/test_master.py`` checks that) and measures correctly at
    44.1 kHz instead of being a quarter of a decibel out.
    """
    g, q, fc = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    k = math.tan(math.pi * fc / sr)
    vh = 10 ** (g / 20)
    vb = vh ** 0.4996667741545416
    d = 1 + k / q + k * k
    b1 = np.array([(vh + vb * k / q + k * k) / d, 2 * (k * k - vh) / d, (vh - vb * k / q + k * k) / d])
    a1 = np.array([1.0, 2 * (k * k - 1) / d, (1 - k / q + k * k) / d])

    fc2, q2 = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * fc2 / sr)
    d = 1 + k / q2 + k * k
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, 2 * (k * k - 1) / d, (1 - k / q2 + k * k) / d])

    return sig.lfilter(b2, a2, sig.lfilter(b1, a1, x, axis=-1), axis=-1)


def loudness_lufs(audio: np.ndarray, sr: int = SR) -> float:
    """Integrated loudness, gated, in LUFS. ``-inf`` for silence."""
    x = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    k = _k_weight(x, sr)
    block, hop = int(0.4 * sr), max(1, int(0.1 * sr))
    if k.shape[1] < block:
        block, hop = k.shape[1], max(1, k.shape[1])
    starts = range(0, k.shape[1] - block + 1, hop)
    # channel weights: 1.0 for L and R (BS.1770 only lifts the surrounds)
    power = np.array([float(np.sum(np.mean(k[:, i:i + block] ** 2, axis=1))) for i in starts])
    if power.size == 0:
        return float("-inf")
    loud = -0.691 + 10 * np.log10(np.maximum(power, 1e-30))
    keep = loud > -70.0                                   # the absolute gate
    if not keep.any():
        return float("-inf")
    relative = -0.691 + 10 * np.log10(np.mean(power[keep])) - 10.0   # and the relative one
    keep &= loud > relative
    if not keep.any():
        return float("-inf")
    return float(-0.691 + 10 * np.log10(np.mean(power[keep])))


def true_peak_dbtp(audio: np.ndarray, sr: int = SR, oversample: int = 4) -> float:
    """The peak of the reconstructed waveform, not of the samples, in dBTP."""
    x = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    up = sig.resample_poly(x, oversample, 1, axis=-1)
    peak = float(np.max(np.abs(up))) if up.size else 0.0
    return 20 * math.log10(peak) if peak > 0 else float("-inf")


# ------------------------------------------------------------------- shaping

def _release(reduction: np.ndarray, coeff: float, block: int = 1 << 19) -> np.ndarray:
    """A decaying running maximum: ``out[i] = max(r[j] * coeff**(i-j))`` for j <= i.

    The closed form multiplies by ``coeff**-j``, which overflows a float inside a
    few million samples, so it is taken a block at a time with the previous
    block's tail carried in. That keeps it exact and keeps it in numpy — a
    sample loop over a three-minute track is seconds of Python.
    """
    out = np.empty_like(reduction)
    carry = 0.0
    for i in range(0, reduction.size, block):
        chunk = reduction[i:i + block]
        k = np.arange(chunk.size, dtype=np.float64)
        scale = coeff ** k
        run = np.maximum.accumulate(chunk / scale)
        out[i:i + chunk.size] = np.maximum(run * scale, carry * scale)
        carry = float(out[i + chunk.size - 1])
    return out


def limit(audio: np.ndarray, sr: int = SR, ceiling: float = 0.89,
          lookahead_s: float = 0.005, release_s: float = 0.12, oversample: int = 4) -> np.ndarray:
    """A look-ahead **true-peak** limiter: never above ``ceiling``, and never a click.

    The gain reduction is computed in decibels from a forward-looking peak, held
    through a release so a run of peaks is one gain move rather than a hundred,
    and smeared back over the look-ahead window — so the level is already down
    before the peak arrives instead of being clamped as it lands.

    The peak it looks at is taken from the signal oversampled by ``oversample``,
    and the resulting gain curve is folded back down by taking the largest
    reduction in each group. Limiting to sample peaks instead leaves a dense mix
    a decibel over the ceiling once it is reconstructed, and the only way to fix
    that afterwards is to turn the whole track down — which is the loudness you
    just set.
    """
    x = np.atleast_2d(np.asarray(audio, dtype=np.float32))
    if x.size == 0:
        return x.astype(np.float32)
    os_ = max(1, int(oversample))
    fine = sig.resample_poly(x, os_, 1, axis=-1) if os_ > 1 else x
    look = max(1, int(lookahead_s * sr * os_))
    peak = np.max(np.abs(fine), axis=0).astype(np.float64)
    ahead = ndi.maximum_filter1d(peak, size=look, origin=-(look // 2), mode="nearest")
    reduction = np.maximum(0.0, 20 * np.log10(np.maximum(ahead, 1e-12) / ceiling))
    held = _release(reduction, math.exp(-1.0 / max(1.0, release_s * sr * os_)))
    # The smear: a centred average of the reduction curve, taken as a maximum so
    # it can soften the ramp into a peak but never let one through.
    smooth = held if look <= 1 else np.maximum(held, ndi.uniform_filter1d(held, size=look, mode="nearest"))
    if os_ > 1:  # back to the base rate, keeping the largest reduction in each group
        n = x.shape[1]
        pad = np.full(n * os_, smooth[-1] if smooth.size else 0.0)
        pad[: min(smooth.size, pad.size)] = smooth[: pad.size]
        smooth = pad.reshape(n, os_).max(axis=1)
    return (x * (10 ** (-smooth / 20)).astype(np.float32)).astype(np.float32)


def fade(audio: np.ndarray, sr: int = SR, fade_in_s: float = 0.35,
         fade_out_s: float = 4.0) -> np.ndarray:
    """Equal-power fades, so a piece starts and ends instead of being cut."""
    x = np.array(np.atleast_2d(audio), dtype=np.float32, copy=True)
    n = x.shape[1]
    a = min(int(fade_in_s * sr), n)
    b = min(int(fade_out_s * sr), n - a if n > a else 0)
    if a > 1:
        x[:, :a] *= np.sin(np.linspace(0, math.pi / 2, a, dtype=np.float32)) ** 2
    if b > 1:
        x[:, n - b:] *= np.cos(np.linspace(0, math.pi / 2, b, dtype=np.float32)) ** 2
    return x


def master(audio: np.ndarray, sr: int = SR, target_lufs: float = TARGET_LUFS,
           ceiling_dbtp: float = CEILING_DBTP, fade_in_s: float = 0.35,
           fade_out_s: float = 4.0) -> tuple[np.ndarray, dict]:
    """Fade, set the loudness, limit to a true-peak ceiling. Returns the report too."""
    faded = fade(audio, sr, fade_in_s, fade_out_s)
    before = loudness_lufs(faded, sr)
    if not math.isfinite(before):
        return faded, {"lufs_before": None, "lufs": None,
                       "true_peak_dbtp": round(true_peak_dbtp(faded, sr), 2), "peak": 0.0, "gain_db": 0.0}
    ceiling = float(10 ** (ceiling_dbtp / 20))
    # Limiting a track costs it a little loudness, so the gain and the limiter
    # are run twice with the second pass correcting for what the first lost.
    # One correction is enough: the error after it is hundredths of a decibel.
    trim, after, x = 0.0, before, faded
    for _ in range(2):
        x = limit((faded * 10 ** ((target_lufs + trim - before) / 20)).astype(np.float32), sr, ceiling=ceiling)
        after = loudness_lufs(x, sr)
        if not math.isfinite(after):
            break
        trim += target_lufs - after
    tp = true_peak_dbtp(x, sr)
    if math.isfinite(tp) and tp > ceiling_dbtp:   # the reconstruction still pokes through
        x = (x * 10 ** ((ceiling_dbtp - tp) / 20)).astype(np.float32)
        after = loudness_lufs(x, sr)
    return x, {
        "lufs_before": round(before, 2) if math.isfinite(before) else None,
        "lufs": round(after, 2) if math.isfinite(after) else None,
        "true_peak_dbtp": round(true_peak_dbtp(x, sr), 2),
        "peak": round(float(np.max(np.abs(x))), 4) if x.size else 0.0,
        "gain_db": round(target_lufs + trim - before, 2),
    }


# -------------------------------------------------------------------- output

def write_wav(path: str | Path, audio: np.ndarray, sr: int = SR, bits: int = 24) -> Path:
    """Stereo PCM at 16 or 24 bits. 24 is the master; 16 gets TPDF dither."""
    if bits not in (16, 24):
        raise ValueError(f"wav depth must be 16 or 24, not {bits}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(np.atleast_2d(np.asarray(audio, dtype=np.float64)).T, -1.0, 1.0)
    if bits == 16:
        # Two rectangular draws make a triangular distribution, which decorrelates
        # the quantisation error instead of leaving it as harmonics of the signal.
        rng = np.random.default_rng(0)
        lsb = 1.0 / 32768.0
        x = np.clip(x + (rng.random(x.shape) - rng.random(x.shape)) * lsb, -1.0, 1.0)
        pcm = (x * 32767.0).astype("<i2").tobytes()
    else:
        ints = np.ascontiguousarray(np.clip(np.round(x * 8388607.0), -8388608, 8388607), dtype="<i4")
        pcm = ints.view(np.uint8).reshape(-1, 4)[:, :3].copy().tobytes()   # little-endian: drop the top byte
    with wave.open(str(path), "wb") as f:
        f.setnchannels(x.shape[1])
        f.setsampwidth(bits // 8)
        f.setframerate(sr)
        f.writeframes(pcm)
    return path
