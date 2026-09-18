"""The synthesiser: a flight rendered to audio, measured rather than listened to."""

from __future__ import annotations

import math
import wave

import numpy as np
import pytest

from flydrones.config import load_config
from flydrones.music.events import Frame, NoteEvent
from flydrones.music.synth import (
    DEFAULT_PATCHES,
    SR,
    Patch,
    SynthConfig,
    describe,
    envelope,
    midi_to_hz,
    partial_weights,
    render,
    render_note,
    reverb,
    write_wav,
)

BRIGHT = "configs/bright.yaml"


def peak_hz(audio: np.ndarray, sr: int = SR) -> float:
    """The loudest frequency in a buffer."""
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return float(np.fft.rfftfreq(len(audio), 1 / sr)[int(spectrum.argmax())])


def frames_of(notes: list[NoteEvent], cps: float = 0.5) -> list[Frame]:
    return [Frame(t=n.t, notes=[n], cps=cps) for n in notes]


# ------------------------------------------------------------------- one note
def test_a_note_is_a_note():
    audio = render_note(DEFAULT_PATCHES["flylead"], 69, 0.8, 0.5)  # A4
    assert audio.dtype == np.float32 and np.isfinite(audio).all()
    assert 0.0 < np.max(np.abs(audio)) < 1.0
    assert abs(peak_hz(audio) - 440.0) < 6.0, "A4 should come out at 440 Hz"
    assert np.abs(audio[:20]).max() < np.abs(audio).max() * 0.2, "it should not click on"
    assert np.abs(audio[-200:]).max() < np.abs(audio).max() * 0.05, "or off"


def test_the_envelope_rises_holds_and_falls():
    env = envelope(int(SR * 1.2), SR, attack=0.05, decay=0.3, sustain=0.5, release=0.2, hold=0.6)
    assert env[0] < 0.05 and env[int(SR * 0.05)] > 0.9  # attack
    assert 0.3 < env[int(SR * 0.5)] < 0.8  # sustain, on its way down
    assert env[-1] < 0.05  # release
    assert np.isfinite(env).all() and env.max() <= 1.0


def test_partials_stop_at_nyquist_so_nothing_aliases():
    low = partial_weights(DEFAULT_PATCHES["flylead"], midi_to_hz(36), SR)[0]
    high = partial_weights(DEFAULT_PATCHES["flylead"], midi_to_hz(103), SR)[0]
    assert len(low) > len(high), "a high note has to be built from fewer harmonics"
    assert (high * midi_to_hz(103)).max() < SR / 2, "a partial above Nyquist would fold back as noise"
    for name, patch in DEFAULT_PATCHES.items():
        k, amp = partial_weights(patch, 440.0, SR)
        assert len(k) >= 1 and amp.sum() == pytest.approx(1.0, abs=1e-5), name


def test_a_glide_starts_at_the_note_it_came_from():
    patch = Patch(wave="sine", attack=0.001, glide_s=0.2, decay=2.0, sustain=1.0)
    audio = render_note(patch, 69, 1.0, 0.6, glide_from=midi_to_hz(57))  # an octave below
    start, end = audio[: int(SR * 0.04)], audio[int(SR * 0.45) : int(SR * 0.55)]
    assert abs(peak_hz(start) - midi_to_hz(57)) < 30, "the slide should begin where the last note was"
    assert abs(peak_hz(end) - 440.0) < 10, "and arrive"


def test_fm_is_brighter_than_a_sine():
    bell = render_note(DEFAULT_PATCHES["flyspark"], 84, 0.8, 0.3)
    sine = render_note(Patch(wave="sine"), 84, 0.8, 0.3)
    assert describe(np.stack([bell, bell]))["centroid_hz"] > describe(np.stack([sine, sine]))["centroid_hz"]


# --------------------------------------------------------------------- effects
def test_reverb_adds_a_tail_and_stays_finite():
    x = np.zeros(SR, dtype=np.float32)
    x[:100] = 1.0
    wet = reverb(x, SR, size=0.8, damp=0.3)
    assert np.isfinite(wet).all()
    late = np.abs(wet[int(SR * 0.4) :]).max()
    assert 0 < late < 1.0, "a tail, not a runaway"
    assert np.abs(wet[int(SR * 0.05) : int(SR * 0.2)]).max() > late, "and it should decay"


# ---------------------------------------------------------------------- render
def note(t: float, voice: str = "wing_left", sound: str = "flylead", pitch: int = 69, **kw) -> NoteEvent:
    return NoteEvent(t=t, voice=voice, sound=sound, note=pitch, velocity=0.8, duration=0.3, **kw)


def test_rendering_a_flight_makes_a_stereo_track():
    audio = render(frames_of([note(0.0), note(0.5, pitch=72), note(1.0, pitch=76)]), {})
    assert audio.shape[0] == 2 and audio.dtype == np.float32
    assert np.isfinite(audio).all()
    assert 0.5 < np.max(np.abs(audio)) < 1.0, "it should be normalised, and not clipped"
    d = describe(audio)
    assert d["clipped"] == 0 and d["rms"] > 0.005
    assert d["centroid_hz"] > 300


def test_panning_puts_a_voice_on_one_side():
    left = render(frames_of([note(0.0, pan=0.0)]), {})
    right = render(frames_of([note(0.0, pan=1.0)]), {})
    assert np.abs(left[0]).sum() > 3 * np.abs(left[1]).sum()
    assert np.abs(right[1]).sum() > 3 * np.abs(right[0]).sum()


def test_the_same_score_renders_to_the_same_audio():
    score = frames_of([note(0.0), note(0.4, pitch=74), note(0.9, pitch=78)])
    assert np.array_equal(render(score, {}), render(score, {}))


def test_note_lengths_follow_the_tempo():
    """Elasticity, measured on a patch that cannot hide it.

    The stretch is relative to the piece's own median tempo, so it takes a score
    with two tempos in it to see — which is also the only place it matters. The
    patch is a flat plateau (no decay, instant release) so the note's length is
    the thing being measured and not the envelope's shape.
    """
    flat = Patch(wave="sine", attack=0.005, decay=30.0, sustain=1.0, release=0.02, gain=1.0,
                 send_delay=0.0, send_reverb=0.0)
    sc = SynthConfig(patches={"flybass": flat}, reverb_mix=0.0, delay_mix=0.0, air=0.0,
                     compress=1.0, tail_s=0.3)
    score = [Frame(t=0.0, notes=[note(0.0, sound="flybass", pitch=57)], cps=0.25),
             Frame(t=4.0, notes=[note(4.0, sound="flybass", pitch=57)], cps=1.0)]
    audio = np.abs(render(score, sc)).max(axis=0)
    held = lambda seg: np.sum(seg > 0.3 * seg.max()) / SR  # noqa: E731
    slow = held(audio[: int(SR * 3.5)])
    fast = held(audio[int(SR * 4.0) :])
    assert slow > fast * 1.6, f"slow tempo held {slow:.2f} s, fast tempo {fast:.2f} s"
    assert 0.15 < fast < slow < 1.5, "both should still be notes, not drones"


def test_an_empty_score_is_silence_not_a_crash():
    audio = render([], {})
    assert audio.shape[0] == 2 and np.abs(audio).max() == 0.0


# ------------------------------------------------------------------------ file
def test_the_wav_is_a_wav(tmp_path):
    audio = render(frames_of([note(0.0), note(0.3, pitch=73)]), {})
    path = write_wav(tmp_path / "t.wav", audio)
    with wave.open(str(path)) as f:
        assert f.getnchannels() == 2 and f.getsampwidth() == 2 and f.getframerate() == SR
        data = np.frombuffer(f.readframes(f.getnframes()), "<i2").reshape(-1, 2) / 32768
    assert data.shape[0] == audio.shape[1]
    assert np.allclose(data.T, audio, atol=1e-3)


# ---------------------------------------------------------------------- config
def test_the_synth_config_reads_the_music_block():
    sc = SynthConfig.from_config(load_config(BRIGHT))
    assert sc.delay_s > 0 and 0 < sc.reverb_mix < 1 and sc.air > 0
    assert sc.patches["flylead"].glide_s > 0, "the lead should glide: that is the elastic part"
    assert sc.patches["flybass"].send_reverb < sc.patches["flyair"].send_reverb
    with pytest.raises(ValueError):
        SynthConfig.from_config({"music": {"synth": {"loudness": 11}}})
    with pytest.raises(ValueError):
        Patch.from_dict({"wave": "saw", "sparkle": 3})


def test_the_bright_config_is_bright():
    cfg = load_config(BRIGHT)
    assert cfg["music"]["scale"] == "lydian", "lydian is the bright mode; that is the whole point"
    assert cfg["music"]["tempo_from"] == "drive" and cfg["music"]["tempo_swing"] >= 0.3
    sc = SynthConfig.from_config(cfg)
    for voice in cfg["music"]["voices"].values():
        sound = voice.get("sound")
        if sound:
            assert sound in sc.patches, f"{sound} has no patch"
    assert sc.highpass_hz >= 40 and sc.send_highpass_hz > 200, "the reverb should not see the bass"


def test_a_brighter_patch_measures_brighter():
    dull = Patch(wave="saw", partials=4, tilt=2.0, decay_tilt=1.4)
    bright = Patch(wave="saw", partials=24, tilt=0.9, decay_tilt=0.4)
    quiet = describe(np.stack([render_note(dull, 57, 0.9, 0.5)] * 2))
    loud = describe(np.stack([render_note(bright, 57, 0.9, 0.5)] * 2))
    assert loud["centroid_hz"] > quiet["centroid_hz"] * 1.5
    assert loud["above_4k"] > quiet["above_4k"]


def test_describe_measures_what_it_says_it_does():
    sr = SR
    t = np.arange(sr) / sr
    low = np.sin(2 * math.pi * 200 * t).astype(np.float32)
    high = np.sin(2 * math.pi * 6000 * t).astype(np.float32)
    assert describe(np.stack([low, low]))["centroid_hz"] < 400
    assert describe(np.stack([high, high]))["centroid_hz"] > 5000
    assert describe(np.stack([low, low]))["width"] < 0.01  # mono
    assert describe(np.stack([low, -low]))["width"] > 1.5  # out of phase is as wide as it gets
