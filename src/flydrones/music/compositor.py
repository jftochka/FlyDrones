"""The compositor: one control tick of a fly flying a drone -> musical events.

The brain is not written to make music, so the mapping is engineered, and like
every other engineered part of this project it is written down rather than
hidden: which neuron group feeds which voice, what firing rate opens it, and
what range it plays lives in the ``music:`` block of ``defaults.yaml``.

Two rules keep it musical rather than merely reactive:

* **A voice has a threshold with hysteresis and a refractory period.** A rate
  turned straight into a note stream at the control rate is a buzz, not a
  phrase. Voices open above ``on_hz``, stay open until ``off_hz`` and never
  retrigger faster than ``min_gap_s`` — the same shape as the neuron they are
  listening to.
* **Rates are read as a distance from rest.** ``DNg02`` idles near 31 Hz in
  MiniFly and somewhere else entirely in MaleCNS, so thresholds are deltas
  above the baseline the decoder already measures during warm-up. The same
  score settings then work on a brain they were not written for.

Timing comes from the brain where it can: the spike raster carries real spike
times at the LIF timestep, so a note is placed at the millisecond its group
first fired inside the tick instead of on the 20 Hz control grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .events import ControlEvent, Frame, NoteEvent, Scale

SECTIONS = ("ground", "hover", "climb", "descend", "turn", "escape")


def _num(x, default: float = 0.0) -> float:
    """Anything that is not a finite number is worth nothing to a synthesiser."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


@dataclass
class VoiceSpec:
    """One instrument, listening to one weighted sum of neuron groups."""

    name: str
    groups: dict[str, float] = field(default_factory=dict)
    sound: str = "flylead"
    mode: str = "repeat"  # repeat = arpeggio while excited, trigger = one note per burst
    on_hz: float = 6.0
    off_hz: float = 3.0
    span_hz: float = 25.0
    low: int = 48
    high: int = 72
    every_s: float = 0.4  # note spacing at the threshold
    fastest_s: float = 0.12  # note spacing when the group is fully excited
    min_gap_s: float = 0.08
    duration: float = 0.25
    pan: float = 0.5
    orbit: int = 0
    channel: int = 1
    gain: float = 0.9
    rectify: str = "pos"  # pos = only excitation above rest, abs = either direction
    note: str = ""  # what this voice is, for the docs and the patches

    @staticmethod
    def from_dict(name: str, d: dict) -> VoiceSpec:
        d = dict(d or {})
        groups = d.pop("groups", None)
        one = d.pop("group", None)
        if groups is None:
            groups = {one: 1.0} if one else {}
        if isinstance(groups, (list, tuple)):
            groups = {g: 1.0 / len(groups) for g in groups}
        known = {f.name for f in VoiceSpec.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"voice {name}: unknown setting(s) {', '.join(sorted(unknown))}")
        return VoiceSpec(name=name, groups={str(k): float(v) for k, v in groups.items()}, **d)


class Voice:
    """The state a threshold needs: whether it is open, and when it last spoke."""

    def __init__(self, spec: VoiceSpec, scale: Scale):
        self.spec = spec
        self.scale = scale
        self.open = False
        self.level = 0.0
        self.last_note_t = -1e9
        self.notes = 0

    def excitation(self, rates: dict[str, float], baselines: dict[str, float]) -> float:
        x = sum(w * (_num(rates.get(g)) - _num(baselines.get(g))) for g, w in self.spec.groups.items())
        return abs(x) if self.spec.rectify == "abs" else max(0.0, x)

    def update(self, t: float, rates: dict[str, float], baselines: dict[str, float], onset_t: float | None = None):
        s = self.spec
        x = self.level = self.excitation(rates, baselines)
        was_open = self.open
        self.open = x >= s.on_hz if not was_open else x > s.off_hz
        if not self.open:
            return None
        if s.mode == "trigger" and was_open:
            return None  # one note per burst, however long the burst lasts
        norm = _clamp01((x - s.on_hz) / s.span_hz) if s.span_hz > 0 else 1.0
        gap = t - self.last_note_t
        if gap < s.min_gap_s:
            return None
        if s.mode == "repeat" and gap < s.every_s + (s.fastest_s - s.every_s) * norm:
            return None
        self.last_note_t = t
        self.notes += 1
        return NoteEvent(
            t=max(t, _num(onset_t, t)) if onset_t is not None else t,
            voice=s.name,
            sound=s.sound,
            note=self.scale.quantise(norm, s.low, s.high),
            velocity=round(_clamp01(0.35 + 0.65 * norm) * s.gain, 4),
            duration=s.duration,
            pan=s.pan,
            orbit=s.orbit,
            channel=s.channel,
            rate_hz=round(x, 3),
        )


class Compositor:
    """Turns a stream of :class:`~flydrones.runtime.TickInfo` into a stream of events."""

    def __init__(self, cfg: dict, brain=None, baselines: dict[str, float] | None = None):
        m = dict(cfg.get("music", {}) or {})
        self.cfg = m
        self.scale = Scale.parse(m.get("scale", "minor_pentatonic"), m.get("root", "C3"))
        self.cps = self.base_cps = float(m.get("cps", 0.5))
        self.tempo_from = str(m.get("tempo_from", "fixed"))
        self.tempo_swing = float(m.get("tempo_swing", 0.4))
        self.smoothing = float(m.get("smoothing", 0.35))
        self.relative = bool(m.get("relative_to_baseline", True))
        self.max_alt = float(cfg.get("safety", {}).get("max_alt_m", 2.2))
        self.loom_max_hz = float(m.get("loom_max_hz", 110.0))
        self.bright_max_hz = float((cfg.get("inputs", {}).get("R16_L") or {}).get("max_hz", 40.0))
        self.drive_span_hz = float(m.get("drive_span_hz", 25.0))
        self.voices = [Voice(VoiceSpec.from_dict(k, v), self.scale) for k, v in (m.get("voices") or {}).items()]
        self.baselines: dict[str, float] = dict(baselines or {})
        self.rates: dict[str, float] = {}
        self.cycle = 0.0
        self.section = "ground"
        self.frames = 0
        self.notes = 0
        self._prev_brain_ms: float | None = None
        self._pos_voice: np.ndarray | None = None
        self._groups: list[str] = []
        if brain is not None:
            self._index_raster(brain)

    # ------------------------------------------------------------------ setup
    def _index_raster(self, brain) -> None:
        """Map raster positions back to neuron groups, so notes get spike times."""
        self._groups = sorted({g for v in self.voices for g in v.spec.groups})
        pos = np.full(len(brain.record), -1, dtype=np.int16)
        where = {int(n): i for i, n in enumerate(np.asarray(brain.record))}
        for gi, name in enumerate(self._groups):
            idx = brain.connectome.group(name)
            for n in np.asarray(idx).tolist():
                p = where.get(int(n))
                if p is not None:
                    pos[p] = gi
        self._pos_voice = pos if (pos >= 0).any() else None

    def set_baselines(self, baselines: dict[str, float]) -> None:
        """Resting rates from the decoder's warm-up; thresholds are deltas on these."""
        self.baselines = {k: _num(v) for k, v in (baselines or {}).items()}

    # ------------------------------------------------------------------ timing
    def _onsets(self, info, dt: float) -> dict[str, float]:
        """First spike time (seconds into this tick) for each group we listen to."""
        raster = getattr(info, "raster", None)
        if self._pos_voice is None or not raster:
            return {}
        start_ms = _num(getattr(info, "brain_ms", 0.0)) - dt * 1000.0
        out: dict[str, float] = {}
        for t_ms, positions in raster:
            ids = self._pos_voice[np.asarray(positions, dtype=np.int64)]
            for gi in np.unique(ids[ids >= 0]).tolist():
                # clamped to the tick: a note belongs to the frame that made it,
                # whatever a stale raster or a resumed brain clock says
                out.setdefault(self._groups[gi], min(dt, max(0.0, (_num(t_ms) - start_ms) / 1000.0)))
            if len(out) == len(self._groups):
                break
        return out

    def _dt(self, info) -> float:
        ms = _num(getattr(info, "brain_ms", 0.0))
        dt = (ms - self._prev_brain_ms) / 1000.0 if self._prev_brain_ms is not None else 0.0
        self._prev_brain_ms = ms
        return dt if 0 < dt < 1.0 else 0.05

    # ------------------------------------------------------------------ mapping
    def _section(self, cmd, tel) -> str:
        if getattr(cmd, "escape", False):
            return "escape"
        if not getattr(tel, "flying", False):
            return "ground"
        if _num(cmd.throttle) > 0.15:
            return "climb"
        if _num(cmd.throttle) < -0.15:
            return "descend"
        if abs(_num(cmd.yaw)) > 0.2:
            return "turn"
        return "hover"

    def _delta(self, group: str) -> float:
        base = self.baselines.get(group, 0.0) if self.relative else 0.0
        return _num(self.rates.get(group)) - _num(base)

    def _mean(self, *groups: str) -> float:
        return sum(self._delta(g) for g in groups) / max(1, len(groups))

    def _controls(self, t: float, info, dt: float) -> list[ControlEvent]:
        # The voices are configurable; these controls are not, because they are the
        # standard read-outs of the standard groups. Rename LPLC2 or R1-R6 in your
        # own config and `loom` and `bright` go quiet — the voices still follow.
        tel, cmd = info.tel, info.cmd
        alt = _num(getattr(tel, "alt_m", 0.0))
        loom = max(self._delta("LPLC2_L"), self._delta("LPLC2_R"), self._delta("LC4_L"), self._delta("LC4_R"))
        drive = self._mean("DNg02_L", "DNg02_R")
        bright = (_num(self.rates.get("R16_L")) + _num(self.rates.get("R16_R"))) / 2.0
        c = {
            "alt": _clamp01(alt / max(0.2, self.max_alt)),
            "alt_m": alt,
            "climb": _num(cmd.throttle),
            "turn": _num(cmd.yaw),
            "fwd": _num(cmd.forward),
            "escape": 1.0 if getattr(cmd, "escape", False) else 0.0,
            "battery": _clamp01(_num(getattr(tel, "battery_pct", 100.0), 100.0) / 100.0),
            "drive": _clamp01(drive / max(1e-6, self.drive_span_hz)),
            "balance": max(-1.0, min(1.0, (self._delta("DNg02_R") - self._delta("DNg02_L")) / max(1e-6, self.drive_span_hz))),
            "loom": _clamp01(loom / max(1e-6, self.loom_max_hz)),
            "bright": _clamp01(bright / max(1e-6, self.bright_max_hz)),
            "spikes": _num(getattr(info, "spikes", 0)) / max(1e-6, dt) / 1000.0,  # kilospikes per second
            "cycle": self.cycle,
            "cps": self.cps,
        }
        return [ControlEvent(t=t, name=k, value=round(float(v), 5)) for k, v in c.items()]

    # ------------------------------------------------------------------- public
    def tick(self, info) -> Frame:
        """One control tick in, one :class:`Frame` of notes and controls out."""
        t = _num(getattr(info, "t", 0.0))
        dt = self._dt(info)
        a = _clamp01(self.smoothing)
        for k, v in (getattr(info, "rates", None) or {}).items():
            self.rates[k] = (1 - a) * self.rates.get(k, _num(v)) + a * _num(v)

        onsets = self._onsets(info, dt)
        frame = Frame(t=t, cycle=self.cycle, cps=self.cps, section=self._section(info.cmd, info.tel))
        for voice in self.voices:
            onset = min((onsets[g] for g in voice.spec.groups if g in onsets), default=None)
            ev = voice.update(t, self.rates, self.baselines if self.relative else {}, t + onset if onset is not None else None)
            if ev is not None:
                frame.notes.append(ev)
        frame.notes.sort(key=lambda e: e.t)
        frame.controls = self._controls(t, info, dt)

        # tempo and cycle position advance after the frame is stamped, so a
        # replay of the same tick lands on the same cycle it was composed on.
        signal = {"drive": self._mean("DNg02_L", "DNg02_R") / max(1e-6, self.drive_span_hz),
                  "alt": _num(getattr(info.tel, "alt_m", 0.0)) / max(0.2, self.max_alt)}.get(self.tempo_from)
        self.cps = (self.base_cps if signal is None
                    else self.base_cps * (1.0 + self.tempo_swing * (_clamp01(signal) - 0.5) * 2.0))
        self.cycle += self.cps * dt
        self.section = frame.section
        self.frames += 1
        self.notes += len(frame.notes)
        return frame

    def summary(self) -> str:
        parts = ", ".join(f"{v.spec.name} {v.notes}" for v in self.voices if v.notes)
        return f"{self.notes} notes in {self.frames} ticks ({parts or 'silence'})"
