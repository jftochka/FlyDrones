"""An offline synthesiser, so a flight can be listened to without installing anything.

Everything else in ``music/`` hands notes to a program that makes sound —
SuperDirt, Pure Data, a browser. This renders them itself, with numpy, into a
WAV file. It exists because "listen to it" should not require a SuperCollider
install, and because a file is the only form of a piece you can send someone.

The design follows the brief it was written for: **melodic, elastic, bright**.

* **Melodic** — the compositor already quantises pitch to a scale; here every
  voice is a patch with a register, a glide and a place in the stereo field, so
  the lines stay separate instead of piling up in the middle.
* **Elastic** — three kinds of give. The onsets carry the millisecond the
  neurons fired, not the control grid. A note glides into the next one from the
  same voice when they are close together. And note lengths scale with the
  tempo the flight is running at, so the phrasing breathes as the fly works
  harder.
* **Bright** — additive synthesis with the partials written out, each decaying
  faster than the one below it, so the attack is bright and the tail is not
  harsh; then a stereo delay, a plate-ish reverb and an air shelf on top.

Additive rather than a filter because a time-varying filter needs a sample loop
and this needs to run in numpy, and because the harmonic count can be capped at
Nyquist per note, which is band-limiting for free: no aliasing, at any pitch.

    frames = [comp.tick(info) for ...]
    audio = render(frames, cfg)
    write_wav("flight.wav", audio)
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy import signal as sig

SR = 44100
TWO_PI = 2.0 * math.pi


def midi_to_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((float(note) - 69.0) / 12.0)


@dataclass
class Patch:
    """One instrument. Everything a voice needs to become a sound."""

    wave: str = "saw"  # saw | square | tri | sine | fm | noise
    partials: int = 16  # how many harmonics before Nyquist takes over
    tilt: float = 1.0  # amplitude fall-off across the partials: bigger is softer
    decay_tilt: float = 0.55  # how much faster the high partials die: this is the "bright but not harsh"
    attack: float = 0.01
    decay: float = 0.35
    sustain: float = 0.3
    release: float = 0.25
    unison: int = 1
    detune_cents: float = 0.0
    glide_s: float = 0.0  # portamento from the last note of this voice, if it was recent
    glide_window_s: float = 0.7
    vibrato: float = 0.0  # depth, semitones
    vibrato_hz: float = 5.0
    fm_ratio: float = 2.0
    fm_index: float = 2.0
    fm_decay: float = 0.25
    gain: float = 0.5
    pan_width: float = 1.0  # how far this voice is allowed from the middle
    send_delay: float = 0.3
    send_reverb: float = 0.3
    stretch: float = 1.0  # multiplies the note length the compositor asked for

    @staticmethod
    def from_dict(d: dict) -> Patch:
        known = {f.name for f in Patch.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown patch setting(s): {', '.join(sorted(unknown))}")
        return Patch(**d)


# A kit that sounds like the brief with no configuration at all. Every generic
# sound name the compositor uses has an entry; configs/bright.yaml overrides them.
DEFAULT_PATCHES: dict[str, Patch] = {
    "flylead": Patch(wave="saw", partials=18, tilt=1.15, decay_tilt=0.6, attack=0.012, decay=0.45,
                     sustain=0.22, release=0.4, unison=2, detune_cents=8, glide_s=0.07, vibrato=0.05,
                     gain=0.42, send_delay=0.45, send_reverb=0.4),
    "flybass": Patch(wave="tri", partials=10, tilt=1.6, decay_tilt=0.4, attack=0.014, decay=0.55,
                     sustain=0.35, release=0.45, unison=2, detune_cents=5, glide_s=0.11,
                     gain=0.5, pan_width=0.35, send_delay=0.12, send_reverb=0.22, stretch=1.2),
    "flyair": Patch(wave="sine", partials=6, tilt=1.8, decay_tilt=0.3, attack=0.25, decay=1.1,
                    sustain=0.5, release=0.9, unison=2, detune_cents=11, vibrato=0.03,
                    gain=0.24, send_delay=0.35, send_reverb=0.7, stretch=1.8),
    "flyspark": Patch(wave="fm", fm_ratio=3.5, fm_index=2.6, fm_decay=0.09, attack=0.002, decay=0.22,
                      sustain=0.0, release=0.2, gain=0.2, send_delay=0.6, send_reverb=0.55),
    "flyperc": Patch(wave="square", partials=9, tilt=1.4, decay_tilt=0.9, attack=0.002, decay=0.1,
                     sustain=0.0, release=0.12, gain=0.3, send_delay=0.5, send_reverb=0.35),
    "flyloom": Patch(wave="square", partials=12, tilt=1.05, decay_tilt=0.7, attack=0.02, decay=0.18,
                     sustain=0.12, release=0.2, unison=2, detune_cents=14, gain=0.22,
                     send_delay=0.55, send_reverb=0.5),
    "flycrash": Patch(wave="fm", fm_ratio=1.41, fm_index=5.0, fm_decay=0.45, attack=0.004, decay=1.4,
                      sustain=0.0, release=1.6, gain=0.5, send_delay=0.4, send_reverb=0.8, stretch=1.4),
    "flysting": Patch(wave="fm", fm_ratio=2.01, fm_index=4.2, fm_decay=0.3, attack=0.003, decay=0.9,
                      sustain=0.0, release=1.0, gain=0.42, send_delay=0.45, send_reverb=0.7),
    "flyturn": Patch(wave="tri", partials=12, tilt=1.25, decay_tilt=0.5, attack=0.02, decay=0.4,
                     sustain=0.25, release=0.5, unison=2, detune_cents=6, glide_s=0.09, vibrato=0.04,
                     gain=0.34, send_delay=0.4, send_reverb=0.45, stretch=1.15),
}


# ----------------------------------------------------------------- one note
def envelope(n: int, sr: int, attack: float, decay: float, sustain: float, release: float,
             hold: float) -> np.ndarray:
    """Attack, decay to the sustain level, hold, release. Linear attack, exponential tails."""
    t = np.arange(n, dtype=np.float32) / sr
    a = max(1e-4, attack)
    env = np.minimum(t / a, 1.0).astype(np.float32)
    tail = np.maximum(0.0, t - a)
    env *= (sustain + (1.0 - sustain) * np.exp(-tail / max(1e-3, decay))).astype(np.float32)
    off = a + max(0.0, hold)
    after = np.maximum(0.0, t - off)
    env *= np.exp(-after / max(1e-3, release * 0.6)).astype(np.float32)
    return env


def partial_weights(patch: Patch, f0: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """(harmonic numbers, amplitudes) — capped at Nyquist, which is the anti-aliasing."""
    limit = int(min(patch.partials, max(1, (sr * 0.46) // max(1.0, f0))))
    k = np.arange(1, limit + 1, dtype=np.float32)
    if patch.wave == "square":
        k = k[k % 2 == 1]
    elif patch.wave == "tri":
        k = k[k % 2 == 1]
    if k.size == 0:
        k = np.array([1.0], dtype=np.float32)
    amp = 1.0 / k**patch.tilt
    if patch.wave == "tri":
        amp = 1.0 / k**2.0
    if patch.wave == "sine":
        k, amp = k[:1], amp[:1]
    return k, (amp / np.sum(amp)).astype(np.float32)


def render_note(patch: Patch, note: float, velocity: float, seconds: float, sr: int = SR,
                glide_from: float | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """One note, mono, as float32. The pitch curve is where the elasticity lives."""
    rng = rng or np.random.default_rng(0)
    seconds = float(max(0.02, seconds))
    n = int((seconds + patch.release) * sr)
    if n <= 1:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / sr

    # pitch: glide in from the previous note, then a little vibrato once it settles
    target = midi_to_hz(note)
    if glide_from and patch.glide_s > 0:
        k = np.minimum(1.0, t / max(1e-4, patch.glide_s)).astype(np.float32)
        k = k * k * (3 - 2 * k)  # smoothstep: a slide, not a ramp
        semis = np.log2(max(1e-6, glide_from) / target) * 12.0 * (1.0 - k)
    else:
        semis = np.zeros(n, dtype=np.float32)
    if patch.vibrato > 0:
        onset = np.clip((t - 0.12) / 0.25, 0.0, 1.0)
        semis = semis + patch.vibrato * onset * np.sin(TWO_PI * patch.vibrato_hz * t + rng.random() * TWO_PI)
    freq = (target * 2.0 ** (semis / 12.0)).astype(np.float32)
    phase = (TWO_PI * np.cumsum(freq) / sr).astype(np.float32)

    env = envelope(n, sr, patch.attack, patch.decay, patch.sustain, patch.release, seconds - patch.attack)
    out = np.zeros(n, dtype=np.float32)

    if patch.wave == "fm":
        # one modulator, its index falling away: a struck bell rather than a drone
        index = patch.fm_index * np.exp(-t / max(1e-3, patch.fm_decay))
        out = np.sin(phase + index * np.sin(phase * patch.fm_ratio)).astype(np.float32)
    elif patch.wave == "noise":
        out = rng.standard_normal(n).astype(np.float32) * 0.5
    else:
        detunes = [0.0] if patch.unison < 2 else np.linspace(-1.0, 1.0, patch.unison)
        for d in detunes:
            ph = phase * float(2.0 ** (d * patch.detune_cents / 1200.0))
            k, amp = partial_weights(patch, float(freq.max()), sr)
            # every harmonic decays faster than the one below it: bright attack, warm tail
            decays = patch.decay / (k**patch.decay_tilt)
            partial_env = np.exp(-t[None, :] / np.maximum(1e-3, decays)[:, None]).astype(np.float32)
            out += (amp[:, None] * partial_env * np.sin(np.outer(k, ph))).sum(0).astype(np.float32)
        out /= max(1, len(detunes))

    return (out * env * float(np.clip(velocity, 0.0, 1.5)) * patch.gain).astype(np.float32)


# -------------------------------------------------------------------- effects
def _feedback(signal: np.ndarray, delay: int, gain: float) -> np.ndarray:
    """y[n] = x[n] + g·y[n−D], done a delay-length block at a time.

    The feedback tap is always at least one block behind, so the whole thing
    vectorises: no per-sample Python loop anywhere in this file.
    """
    if delay < 1 or gain == 0:
        return signal
    y = signal.astype(np.float32).copy()
    for i in range(delay, len(y), delay):
        block = min(delay, len(y) - i)
        y[i : i + block] += gain * y[i - delay : i - delay + block]
    return y


def comb(signal: np.ndarray, delay: int, gain: float, damp: float) -> np.ndarray:
    """A damped comb: the reverb's tail loses its highs as it decays, like a room."""
    out = _feedback(signal, delay, gain * (1.0 - damp))
    if damp > 0:  # one-pole lowpass over the whole thing, cheap and close enough
        a = float(np.clip(damp, 0.0, 0.95))
        kernel_len = 64
        kernel = (1 - a) * a ** np.arange(kernel_len, dtype=np.float32)
        out = np.convolve(out, kernel / kernel.sum(), mode="same").astype(np.float32)
    return out


def allpass(signal: np.ndarray, delay: int, gain: float) -> np.ndarray:
    if delay < 1:
        return signal
    delayed = np.concatenate([np.zeros(delay, np.float32), signal[:-delay]])
    return (_feedback(delayed - gain * signal, delay, gain) + gain * signal).astype(np.float32)


def reverb(signal: np.ndarray, sr: int = SR, size: float = 0.8, damp: float = 0.3) -> np.ndarray:
    """Schroeder: four combs in parallel, two allpasses in series. Small, and it sings."""
    times = np.array([0.0297, 0.0371, 0.0411, 0.0437]) * (0.6 + 0.8 * float(size))
    gains = np.clip(0.78 + 0.18 * float(size), 0.0, 0.96)
    wet = np.zeros_like(signal)
    for t in times:
        wet += comb(signal, int(t * sr), float(gains), float(damp))
    wet /= len(times)
    for t, g in ((0.005, 0.7), (0.0017, 0.7)):
        wet = allpass(wet, int(t * sr), g)
    return wet.astype(np.float32)


def butter(audio: np.ndarray, hz: float, sr: int, kind: str = "highpass", order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth. Used for the three cuts that decide whether a mix
    is bright or merely loud at the bottom."""
    if hz <= 0 or hz >= sr / 2:
        return audio
    sos = sig.butter(order, hz / (sr / 2), btype=kind, output="sos")
    return sig.sosfiltfilt(sos, audio, axis=-1).astype(np.float32)


def compress(audio: np.ndarray, sr: int = SR, threshold: float = 0.16, ratio: float = 3.0,
             window_s: float = 0.03, smooth_s: float = 0.09) -> np.ndarray:
    """Glue, not loudness war: one gentle gain curve over the whole mix.

    The gain is computed from a moving RMS and then smoothed, which is a
    compressor with a symmetrical attack and release — enough to stop a giant
    fibre landing on top of a quiet passage and taking the level with it, and
    cheap enough to stay in numpy.
    """
    if ratio <= 1.0:
        return audio
    mono = audio.mean(axis=0)
    w = max(4, int(window_s * sr))
    power = np.convolve(mono**2, np.hanning(w) / np.sum(np.hanning(w)), mode="same")
    env = np.sqrt(np.maximum(power, 1e-12)).astype(np.float32)
    gain = np.where(env > threshold, (threshold / env) ** (1.0 - 1.0 / ratio), 1.0).astype(np.float32)
    k = max(4, int(smooth_s * sr))
    kernel = np.hanning(k).astype(np.float32)
    gain = np.convolve(gain, kernel / kernel.sum(), mode="same").astype(np.float32)
    return (audio * gain).astype(np.float32)


def air(signal: np.ndarray, amount: float, sr: int = SR) -> np.ndarray:
    """A high shelf, as the signal minus a smoothed copy of itself: the shimmer."""
    if amount <= 0:
        return signal
    k = max(2, int(sr / 4200))
    kernel = np.ones(k, dtype=np.float32) / k
    low = np.convolve(signal, kernel, mode="same").astype(np.float32)
    return (signal + amount * (signal - low)).astype(np.float32)


# --------------------------------------------------------------------- render
@dataclass
class SynthConfig:
    sample_rate: int = SR
    master_gain: float = 0.9
    peak: float = 0.89  # where the normaliser lands the loudest sample
    air: float = 0.35
    delay_s: float = 0.375
    delay_feedback: float = 0.34
    delay_mix: float = 0.22
    ping_pong: bool = True
    reverb_mix: float = 0.3
    reverb_size: float = 0.82
    reverb_damp: float = 0.28
    compress: float = 3.0  # ratio; 1 turns the master compressor off
    compress_threshold: float = 0.16
    highpass_hz: float = 55.0  # nothing below this belongs in the mix
    send_highpass_hz: float = 280.0  # keep the bass out of the reverb, which is where mud comes from
    low_tilt: float = 0.3  # how much of the low mid to take back out: 0.3 is about -3 dB below 220 Hz
    low_tilt_hz: float = 220.0
    tail_s: float = 3.0  # how long the tails are allowed to ring after the last note
    elastic: float = 1.0  # how much the tempo stretches note lengths; 0 turns it off
    seed: int = 0
    patches: dict[str, Patch] = field(default_factory=lambda: dict(DEFAULT_PATCHES))

    @staticmethod
    def from_config(cfg: dict) -> SynthConfig:
        s = dict((cfg.get("music", {}) or {}).get("synth", {}) or {})
        patches = {**DEFAULT_PATCHES}
        for name, spec in (s.pop("patches", None) or {}).items():
            base = patches.get(name, Patch())
            patches[name] = replace(base, **{k: v for k, v in (spec or {}).items()}) if spec else base
        for group, keys in (("delay", ("time_s", "feedback", "mix", "ping_pong")),
                            ("reverb", ("mix", "size", "damp"))):
            block = s.pop(group, None) or {}
            for key in keys:
                if key in block:
                    name = f"{group}_{key}" if key not in ("time_s",) else "delay_s"
                    name = {"delay_ping_pong": "ping_pong"}.get(name, name)
                    s[name] = block[key]
        known = {f.name for f in SynthConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(s) - known
        if unknown:
            raise ValueError(f"unknown synth setting(s): {', '.join(sorted(unknown))}")
        return SynthConfig(patches=patches, **s)


def render(frames, cfg: dict | SynthConfig | None = None, seconds: float | None = None) -> np.ndarray:
    """A list of :class:`~flydrones.music.events.Frame` -> stereo float32 audio."""
    sc = cfg if isinstance(cfg, SynthConfig) else SynthConfig.from_config(cfg or {})
    sr = sc.sample_rate
    notes = [(n, f.cps) for f in frames for n in f.notes]
    if not notes:
        return np.zeros((2, sr), dtype=np.float32)
    base_cps = float(np.median([cps for _, cps in notes if cps > 0] or [0.5]))
    t0 = min(n.t for n, _ in notes)
    span = (seconds if seconds else max(n.t for n, _ in notes) - t0) + sc.tail_s
    total = int(span * sr) + sr
    dry = np.zeros((2, total), dtype=np.float32)
    sends = np.zeros((2, total, 2), dtype=np.float32)  # [channel, sample, (delay, reverb)]
    rng = np.random.default_rng(sc.seed)
    last: dict[str, tuple[float, float]] = {}  # voice -> (end time, pitch) for the glide

    for note, cps in notes:
        patch = sc.patches.get(note.sound, DEFAULT_PATCHES.get(note.sound, Patch()))
        start = note.t - t0
        if seconds and start > seconds:
            break
        # elastic: a slower cycle means longer notes, a faster one means tighter ones
        stretch = patch.stretch
        if sc.elastic and cps > 0:
            stretch *= 1.0 + sc.elastic * (base_cps / cps - 1.0) * 0.6
        length = float(np.clip(note.duration * stretch, 0.05, 6.0))
        prev = last.get(note.voice)
        glide = prev[1] if prev and start - prev[0] < patch.glide_window_s else None
        mono = render_note(patch, note.note, note.velocity, length, sr, glide_from=glide, rng=rng)
        last[note.voice] = (start + length, midi_to_hz(note.note))
        if mono.size == 0:
            continue
        i = int(start * sr)
        end = min(total, i + mono.size)
        if end <= i:
            continue
        chunk = mono[: end - i]
        pan = 0.5 + (float(note.pan) - 0.5) * patch.pan_width
        left = math.cos(pan * math.pi / 2)
        right = math.sin(pan * math.pi / 2)
        dry[0, i:end] += chunk * left
        dry[1, i:end] += chunk * right
        sends[0, i:end, 0] += chunk * left * patch.send_delay
        sends[1, i:end, 0] += chunk * right * patch.send_delay
        sends[0, i:end, 1] += chunk * left * patch.send_reverb
        sends[1, i:end, 1] += chunk * right * patch.send_reverb

    out = dry.copy()
    # Sends are high-passed before they reach the effects: reverb on a bass note
    # is the single reliable way to turn a bright mix into a muddy one.
    if sc.send_highpass_hz > 0:
        for ch in (0, 1):
            for bus in (0, 1):
                sends[ch, :, bus] = butter(sends[ch, :, bus], sc.send_highpass_hz, sr, "highpass")

    # stereo delay, crossed over if it is a ping-pong
    d = int(sc.delay_s * sr)
    if sc.delay_mix > 0 and d > 1:
        a = _feedback(sends[0, :, 0], d, sc.delay_feedback)
        b = _feedback(sends[1, :, 0], d if not sc.ping_pong else int(d * 1.5), sc.delay_feedback)
        delayed = np.stack([np.concatenate([np.zeros(d, np.float32), a[:-d]]),
                            np.concatenate([np.zeros(d, np.float32), b[:-d]])])
        if sc.ping_pong:
            delayed = np.stack([delayed[0] * 0.85 + delayed[1] * 0.3, delayed[1] * 0.85 + delayed[0] * 0.3])
        out += sc.delay_mix * delayed
    if sc.reverb_mix > 0:
        wet = np.stack([reverb(sends[0, :, 1], sr, sc.reverb_size, sc.reverb_damp),
                        reverb(sends[1, :, 1], sr, sc.reverb_size, sc.reverb_damp)])
        out += sc.reverb_mix * wet
    if sc.air > 0:
        out = np.stack([air(out[0], sc.air, sr), air(out[1], sc.air, sr)])

    out -= out.mean(axis=1, keepdims=True)  # no DC
    out = butter(out, sc.highpass_hz, sr, "highpass")
    if sc.low_tilt > 0:  # a low shelf, as the mix minus a little of its own low end
        out = (out - sc.low_tilt * butter(out, sc.low_tilt_hz, sr, "lowpass")).astype(np.float32)
    out = compress(out, sr, sc.compress_threshold, sc.compress)
    out = np.tanh(out * sc.master_gain * 1.15).astype(np.float32)  # soft knee, not a wall
    peak = float(np.max(np.abs(out))) or 1.0
    out *= sc.peak / peak
    if seconds:
        out = out[:, : int((seconds + sc.tail_s) * sr)]
    return out.astype(np.float32)


def write_wav(path: str | Path, audio: np.ndarray, sr: int = SR) -> Path:
    """16-bit stereo, which anything will play."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(audio.T, -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())
    return path


def describe(audio: np.ndarray, sr: int = SR) -> dict:
    """Measure the thing rather than trusting it: level, brightness, width.

    ``centroid_hz`` is the spectral centre of mass — the usual number behind the
    word "bright" — and ``width`` is how much the two channels differ.
    """
    mono = audio.mean(axis=0)
    rms = float(np.sqrt(np.mean(mono**2)))
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    freqs = np.fft.rfftfreq(len(mono), 1 / sr)
    total = float(spectrum.sum()) or 1.0
    centroid = float((spectrum * freqs).sum() / total)
    high = float(spectrum[freqs > 4000].sum() / total)
    side = audio[0] - audio[1]
    return {
        "seconds": round(audio.shape[1] / sr, 2),
        "peak": round(float(np.max(np.abs(audio))), 3),
        "rms": round(rms, 4),
        "rms_dbfs": round(20 * math.log10(max(1e-9, rms)), 1),
        "centroid_hz": round(centroid, 1),
        "above_4k": round(high, 4),
        "width": round(float(np.sqrt(np.mean(side**2)) / max(1e-9, rms)), 3),
        "clipped": int(np.sum(np.abs(audio) >= 0.999)),
    }
