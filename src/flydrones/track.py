"""A flight, written down so it can be watched.

The browser demo flies its own copy of the brain. This is the other direction:
a Python flight — with the mushroom body learning and the compass turning,
neither of which the browser engine has — recorded frame by frame so
``docs/live/replay.html`` can render it in the same 3D room.

It is a *trajectory*, not a simulation: where the drone was, which way it was
pointing, and what the brain knew at the time. That is all a renderer needs, it
survives any later change to the physics, and it is small — a minute of flight
is about 80 kB of JSON.

    rec = TrackRecorder(title="Learning to avoid the chair")
    rec.chapter(0.0, "lap 1")
    rec.add(info)            # once per control tick; it decimates to 25 Hz
    rec.event(12.3, "escape", "the giant fibre fired")
    rec.save("flight.json")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

VERSION = 1

# One row per recorded frame, in this order. Named here rather than as JSON keys
# because a flight is thousands of rows and the names would be most of the file.
FIELDS = ("t", "x", "y", "z", "yaw", "throttle", "yaw_cmd", "forward",
          "escape", "memory", "heading", "goal", "mbon", "kc", "loom", "drive", "hit")


def _num(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


@dataclass
class TrackRecorder:
    """Decimates a control loop into something a renderer can play back."""

    title: str = "A flight"
    subtitle: str = ""
    hz: float = 25.0
    room: str = "bedroom"
    frames: list[list[float]] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    chapters: list[dict] = field(default_factory=list)
    _next_t: float = -1e9

    # ------------------------------------------------------------------ input
    def add(self, info, hit: bool = False) -> bool:
        """One control tick. Returns whether it was kept."""
        t = _num(getattr(info, "t", 0.0))
        if t < self._next_t:
            return False
        self._next_t = t + 1.0 / max(1e-6, self.hz)
        tel, cmd = info.tel, info.cmd
        cog = getattr(info, "cognition", None)
        heading = _num(getattr(cog, "heading_deg", float("nan")), -1.0) if cog else -1.0
        goal = getattr(cog, "goal_deg", None) if cog else None
        row = [
            round(t, 3),
            round(_num(getattr(tel, "x_m", 0.0)), 3),
            round(_num(getattr(tel, "y_m", 0.0)), 3),
            round(_num(getattr(tel, "alt_m", 0.0)), 3),
            round(_num(getattr(tel, "yaw_deg", 0.0)), 1),
            round(_num(cmd.throttle), 3),
            round(_num(cmd.yaw), 3),
            round(_num(cmd.forward), 3),
            1 if getattr(cmd, "escape", False) else 0,
            round(_num(getattr(cog, "memory", 0.0)), 3) if cog else 0.0,
            round(heading, 1),
            -1.0 if goal is None else round(_num(goal), 1),
            round(_num(getattr(cog, "mbon_hz", 0.0)), 1) if cog else 0.0,
            round(_num(getattr(cog, "kc_active", 0.0)), 3) if cog else 0.0,
            round(min(1.0, max(0.0, _num(info.rates.get("LPLC2_L")) / 110.0)), 3),
            round(min(1.0, max(0.0, (_num(info.rates.get("DNg02_L")) + _num(info.rates.get("DNg02_R"))) / 2 / 60.0)), 3),
            1 if hit else 0,
        ]
        self.frames.append(row)
        return True

    def event(self, t: float, kind: str, text: str) -> None:
        """Something worth a caption: an escape, a bump, a lesson, a goal."""
        self.events.append({"t": round(_num(t), 2), "kind": str(kind), "text": str(text)})

    def chapter(self, t: float, name: str, text: str = "") -> None:
        """A section of the flight: one lap, one show."""
        self.chapters.append({"t": round(_num(t), 2), "name": str(name), "text": str(text)})

    # ----------------------------------------------------------------- output
    @property
    def seconds(self) -> float:
        return float(self.frames[-1][0] - self.frames[0][0]) if self.frames else 0.0

    def as_dict(self) -> dict:
        return {
            "version": VERSION,
            "title": self.title,
            "subtitle": self.subtitle,
            "room": self.room,
            "hz": self.hz,
            "seconds": round(self.seconds, 2),
            "fields": list(FIELDS),
            "chapters": self.chapters,
            "events": self.events,
            "frames": self.frames,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), separators=(",", ":")), encoding="utf-8")
        return path

    def summary(self) -> str:
        return (f"{len(self.frames)} frames, {self.seconds:.1f} s, {len(self.events)} events, "
                f"{len(self.chapters)} chapters")


def load(path: str | Path) -> dict:
    """Read a track back, with the rows turned into dicts."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = data.get("fields", list(FIELDS))
    data["rows"] = [dict(zip(fields, row)) for row in data.get("frames", [])]
    return data


class FlightNarrator:
    """Watches a flight and writes the captions: escapes, bumps, lessons, goals.

    The same events the station announces on air, for flights that have no
    station attached to them.
    """

    def __init__(self, recorder: TrackRecorder, memory_step: float = 0.08, min_gap_s: float = 2.0):
        self.rec = recorder
        self.memory_step = float(memory_step)
        self.min_gap_s = float(min_gap_s)  # captions have to be readable, so one per kind at a time
        self._last_of: dict[str, float] = {}
        self._escaping = False
        self._collisions = 0
        self._memory_mark = 0.0
        self._goal: float | None = None
        self.counts = {"escapes": 0, "bumps": 0, "lessons": 0, "goals": 0}

    def _say(self, t: float, kind: str, text: str) -> bool:
        if t - self._last_of.get(kind, -1e9) < self.min_gap_s:
            return False
        self._last_of[kind] = t
        self.rec.event(t, kind, text)
        return True

    def watch(self, info, drone=None) -> bool:
        """Call once per tick, before or after ``TrackRecorder.add``. Returns True on a bump."""
        t = _num(getattr(info, "t", 0.0))
        cmd, cog = info.cmd, getattr(info, "cognition", None)
        hit = False
        if getattr(cmd, "escape", False) and not self._escaping:
            self.counts["escapes"] += 1
            self._say(t, "escape", "the giant fibre fired")
        self._escaping = bool(getattr(cmd, "escape", False))
        hits = int(getattr(drone, "collisions", self._collisions))
        if hits > self._collisions:
            hit = True
            self.counts["bumps"] += 1
            self._say(t, "bump", "it hit the chair")
        self._collisions = hits
        if cog is not None:
            if cog.memory > self._memory_mark + self.memory_step:
                self._memory_mark = cog.memory
                self.counts["lessons"] += 1
                self._say(t, "learning", f"it has learned something: memory {cog.memory:.2f}")
            elif cog.memory < self._memory_mark - self.memory_step:
                self._memory_mark = cog.memory
                self._say(t, "forgetting", f"the memory is fading: {cog.memory:.2f}")
            if cog.goal_deg != self._goal:
                self._goal = cog.goal_deg
                if cog.goal_deg is not None:
                    self.counts["goals"] += 1
                    self._say(t, "goal", f"a new heading to hold: {cog.goal_deg:.0f}°")
        return hit
