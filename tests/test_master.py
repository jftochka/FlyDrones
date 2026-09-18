"""Mastering, checked against the standard rather than against itself.

A loudness meter that is self-consistent and wrong is worse than none, so the
first two tests here are calibration against ITU-R BS.1770-4's own published
numbers: the filter coefficients at 48 kHz and the tone that must read -3.01.
"""

from __future__ import annotations

import math
import wave

import numpy as np
import pytest

from flydrones.music.master import (
    CEILING_DBTP,
    TARGET_LUFS,
    _k_weight,
    fade,
    limit,
    loudness_lufs,
    master,
    true_peak_dbtp,
    write_wav,
)

SR = 44100


def tone(hz: float, seconds: float = 10.0, amp: float = 1.0, sr: int = SR, channels: int = 2) -> np.ndarray:
    x = amp * np.sin(2 * math.pi * hz * np.arange(int(seconds * sr)) / sr)
    out = np.zeros((2, x.size), dtype=np.float32)
    for c in range(channels):
        out[c] = x
    return out


def test_the_k_weighting_is_the_one_the_standard_publishes():
    """BS.1770 gives the two biquads as coefficients at 48 kHz. Ours must be those."""
    impulse = np.zeros((1, 8), dtype=np.float64)
    impulse[0, 0] = 1.0
    got = _k_weight(impulse, 48000)[0]
    from scipy import signal as sig
    want = sig.lfilter([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621],
                       sig.lfilter([1.53512485958697, -2.69169618940638, 1.19839281085285],
                                   [1.0, -1.69065929318241, 0.73248077421585], impulse[0]))
    assert np.allclose(got, want, atol=1e-8)


def test_the_meter_reads_the_calibration_tone():
    """A 0 dBFS 1 kHz sine in one channel is -3.01 LKFS, by definition."""
    assert loudness_lufs(tone(1000, channels=1), SR) == pytest.approx(-3.01, abs=0.1)
    assert loudness_lufs(tone(1000, channels=2), SR) == pytest.approx(0.0, abs=0.1)
    assert loudness_lufs(tone(1000, amp=0.1, channels=2), SR) == pytest.approx(-20.0, abs=0.1)


def test_the_meter_reads_the_same_at_any_sample_rate():
    """The standard only publishes 48 kHz; a 44.1 kHz render must still measure."""
    a = loudness_lufs(tone(1000, amp=0.25, sr=44100), 44100)
    b = loudness_lufs(tone(1000, amp=0.25, sr=48000), 48000)
    assert a == pytest.approx(b, abs=0.05)


def test_the_gate_ignores_the_silence_between_phrases():
    """Padding a piece with silence must not make it measure quieter."""
    loud = tone(400, seconds=8, amp=0.4)
    padded = np.concatenate([loud, np.zeros((2, SR * 20), np.float32)], axis=1)
    assert loudness_lufs(padded, SR) == pytest.approx(loudness_lufs(loud, SR), abs=0.3)
    assert loudness_lufs(np.zeros((2, SR), np.float32), SR) == -math.inf


def test_true_peak_sees_what_the_sample_peak_misses():
    """A sine at exactly Nyquist/2, offset so no sample lands on its crest."""
    n = SR * 2
    x = 0.98 * np.sin(2 * math.pi * (SR / 4) * np.arange(n) / SR + math.pi / 4)
    a = np.stack([x, x]).astype(np.float32)
    sample_peak = 20 * math.log10(float(np.max(np.abs(a))))
    assert true_peak_dbtp(a, SR) > sample_peak + 0.5


def test_the_limiter_holds_the_ceiling_without_touching_what_is_already_under_it():
    quiet = tone(220, seconds=3, amp=0.2)
    assert np.allclose(limit(quiet, SR, ceiling=0.5), quiet, atol=1e-6)

    loud = tone(220, seconds=3, amp=0.2)
    loud[:, SR: SR + 500] *= 4.0                       # one transient, well over
    out = limit(loud, SR, ceiling=0.5)
    assert true_peak_dbtp(out, SR) <= 20 * math.log10(0.5) + 0.05
    assert float(np.max(np.abs(out[:, : SR // 2]))) == pytest.approx(0.2, abs=0.01)


def test_the_limiter_does_not_click():
    """A gain curve with a step in it is audible; the derivative bounds it."""
    x = tone(220, seconds=3, amp=0.3)
    x[:, SR: SR + 200] *= 6.0
    out = limit(x, SR, ceiling=0.4)
    gain = np.abs(out[0]) / np.maximum(np.abs(x[0]), 1e-6)
    assert float(np.max(np.abs(np.diff(gain[np.abs(x[0]) > 0.05])))) < 0.2


def test_mastering_lands_on_the_target_and_under_the_ceiling():
    rng = np.random.default_rng(0)
    music = (tone(220, seconds=20, amp=0.25)
             + 0.1 * rng.normal(0, 1, (2, SR * 20)).astype(np.float32))
    music[:, SR * 5: SR * 5 + 3000] *= 5.0
    out, report = master(music, SR)
    assert report["lufs"] == pytest.approx(TARGET_LUFS, abs=0.3)
    assert report["true_peak_dbtp"] <= CEILING_DBTP + 0.02
    assert loudness_lufs(out, SR) == pytest.approx(TARGET_LUFS, abs=0.3)


def test_mastering_a_quiet_piece_turns_it_up():
    out, report = master(tone(440, seconds=15, amp=0.02), SR)
    assert report["gain_db"] > 15
    assert report["lufs"] == pytest.approx(TARGET_LUFS, abs=0.3)


def test_silence_masters_to_silence_rather_than_to_an_error():
    out, report = master(np.zeros((2, SR * 2), np.float32), SR)
    assert report["lufs"] is None and float(np.max(np.abs(out))) == 0.0


def test_the_fades_are_at_the_ends_and_nowhere_else():
    x = np.ones((2, SR * 10), np.float32)
    out = fade(x, SR, fade_in_s=0.5, fade_out_s=2.0)
    assert out[0, 0] < 0.01 and out[0, -1] < 0.01
    assert out[0, SR] == pytest.approx(1.0, abs=1e-6)       # past the fade in
    assert out[0, SR * 5] == pytest.approx(1.0, abs=1e-6)   # and the middle is untouched
    assert np.all(np.diff(out[0, : int(0.5 * SR)]) >= -1e-7)


@pytest.mark.parametrize("bits", [16, 24])
def test_the_wav_round_trips_at_both_depths(tmp_path, bits):
    audio = tone(440, seconds=1, amp=0.5)
    path = write_wav(tmp_path / f"t{bits}.wav", audio, SR, bits=bits)
    with wave.open(str(path)) as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (2, bits // 8, SR)
        raw = f.readframes(f.getnframes())
    if bits == 24:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        wide = np.zeros((b.shape[0], 4), np.uint8)
        wide[:, :3] = b
        wide[:, 3] = np.where(b[:, 2] > 127, 255, 0)        # sign extension
        back = wide.view("<i4").reshape(-1, 2).T.astype(np.float64) / 8388608.0
    else:
        back = np.frombuffer(raw, "<i2").reshape(-1, 2).T.astype(np.float64) / 32768.0
    assert back.shape == audio.shape
    assert float(np.max(np.abs(back - audio))) < (1e-4 if bits == 16 else 1e-6)


def test_twenty_four_bits_is_quieter_than_sixteen_where_it_counts():
    """The whole reason for 24-bit delivery: the noise floor of the format."""
    quiet = tone(440, seconds=2, amp=1e-4)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        errors = {}
        for bits in (16, 24):
            path = write_wav(Path(d) / f"q{bits}.wav", quiet, SR, bits=bits)
            with wave.open(str(path)) as f:
                raw = f.readframes(f.getnframes())
            if bits == 16:
                back = np.frombuffer(raw, "<i2").reshape(-1, 2).T.astype(np.float64) / 32768.0
            else:
                b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
                wide = np.zeros((b.shape[0], 4), np.uint8)
                wide[:, :3] = b
                wide[:, 3] = np.where(b[:, 2] > 127, 255, 0)
                back = wide.view("<i4").reshape(-1, 2).T.astype(np.float64) / 8388608.0
            errors[bits] = float(np.sqrt(np.mean((back - quiet) ** 2)))
    assert errors[24] < errors[16] / 100


def test_a_bad_bit_depth_is_refused():
    with pytest.raises(ValueError):
        write_wav("/tmp/no.wav", np.zeros((2, 10), np.float32), SR, bits=8)
