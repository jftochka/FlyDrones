"""Safety governor: the last word before any command reaches a motor.

The brain is a research model. It can be silent, saturate or do something
unexpected, so every command passes through hard limits that do not depend on
the brain at all.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .motor.command import AXES, FlightCommand


@dataclass
class Telemetry:
    t: float = field(default_factory=time.monotonic)
    alt_m: float | None = None
    vz_mps: float | None = None
    yaw_deg: float | None = None
    yaw_rate_dps: float = 0.0
    x_m: float | None = None
    y_m: float | None = None
    battery_pct: float | None = None
    flying: bool = False


class SafetyGovernor:
    def __init__(self, cfg: dict):
        s = cfg.get("safety", {})
        link = dict(s.get("link", {}) or {})
        # The link is part of the airframe when the airframe is on LTE: a score
        # that stays low means the drone is about to be on its own, so it stops
        # going anywhere first and lands if it does not come back.
        self.link_enabled = bool(link.get("enabled", True))
        self.link_hold = float(link.get("hold_below", 0.35))
        self.link_land = float(link.get("land_below", 0.15))
        self.link_timeout = float(link.get("timeout_s", 5.0))
        self.link_grace = float(link.get("grace_s", 2.0))
        self._link_bad_since: float | None = None
        self._link_lost_since: float | None = None
        self.max = {"throttle": s.get("max_throttle", 0.6), "yaw": s.get("max_yaw", 0.6),
                    "forward": s.get("max_forward", 0.4), "lateral": s.get("max_lateral", 0.4)}
        self.slew = float(s.get("slew_per_s", 2.5))
        self.min_alt = float(s.get("min_alt_m", 0.3))
        self.max_alt = float(s.get("max_alt_m", 2.0))
        self.fence = float(s.get("geofence_radius_m", 3.0))
        self.brain_timeout = float(s.get("brain_timeout_s", 0.5))
        self.min_batt = float(s.get("min_battery_pct", 20))
        self.max_flight = float(s.get("max_flight_s", 180))
        self._prev = FlightCommand()
        self._start = None
        self.land_requested = False
        self.kill = False
        self.events: list[str] = []

    def _event(self, msg: str) -> None:
        if not self.events or self.events[-1] != msg:
            self.events.append(msg)

    def check_link(self, link, now: float | None = None) -> tuple[bool, bool, str]:
        """(hold, land, why) from one link reading. One bad poll is not a lost link."""
        if link is None or not self.link_enabled:
            self._link_bad_since = self._link_lost_since = None
            return False, False, ""
        now = time.monotonic() if now is None else now
        stale = link.age_s(now) > self.link_timeout
        score = 0.0 if stale else float(link.score)
        why = f"lte {link.grade} ({score:.2f})" if not stale else f"lte silent for {link.age_s(now):.0f}s"
        for threshold, attr in ((self.link_hold, "_link_bad_since"), (self.link_land, "_link_lost_since")):
            if score < threshold or stale:
                if getattr(self, attr) is None:
                    setattr(self, attr, now)
            else:
                setattr(self, attr, None)
        held = self._link_bad_since is not None and now - self._link_bad_since >= self.link_grace
        lost = self._link_lost_since is not None and now - self._link_lost_since >= self.link_grace
        return held, lost, why

    def filter(self, cmd: FlightCommand, tel: Telemetry, dt: float, brain_age_s: float = 0.0, link=None) -> FlightCommand:
        now = tel.t
        if self._start is None:
            self._start = now
        notes = []
        if self.kill:
            return FlightCommand.hover("KILL")
        if brain_age_s > self.brain_timeout:
            cmd = FlightCommand.hover("brain timeout -> hover")
            self._event("brain timeout")
        hold_for_link, land_for_link, link_why = self.check_link(link)
        if land_for_link:
            self.land_requested = True
            notes.append(f"{link_why} -> land")
        elif hold_for_link:
            # the note goes on the command itself, so it is not also in `notes`
            # and printed twice on every tick the link is down
            cmd = FlightCommand.hover(f"{link_why} -> hold")
            self._event(f"{link_why} -> hold")
        out = {a: max(-self.max[a], min(self.max[a], getattr(cmd, a))) for a in AXES}

        if tel.alt_m is not None:
            soft = self.max_alt - 0.3
            if soft < tel.alt_m < self.max_alt and out["throttle"] > 0:
                out["throttle"] *= (self.max_alt - tel.alt_m) / 0.3  # fade climbing out near the ceiling
            if tel.alt_m >= self.max_alt and out["throttle"] > 0:
                out["throttle"] = min(0.0, out["throttle"]) - 0.2
                notes.append("ceiling")
            if tel.flying and tel.alt_m <= self.min_alt and out["throttle"] < 0:
                out["throttle"] = 0.0
                notes.append("floor")
        if tel.x_m is not None and tel.y_m is not None:
            r = math.hypot(tel.x_m, tel.y_m)
            if r > self.fence and out["forward"] > 0:
                out["forward"] = 0.0
                notes.append("geofence")
        if tel.battery_pct is not None and tel.battery_pct < self.min_batt:
            self.land_requested = True
            notes.append("battery low -> land")
        if now - self._start > self.max_flight:
            self.land_requested = True
            notes.append("max flight time -> land")

        # slew-rate limit, but always allow moving toward zero instantly
        max_step = self.slew * max(dt, 1e-3)
        for a in AXES:
            prev = getattr(self._prev, a)
            v = out[a]
            if abs(v) > abs(prev) or math.copysign(1, v) != math.copysign(1, prev):
                v = max(prev - max_step, min(prev + max_step, v))
            out[a] = v
        safe = FlightCommand(**out, escape=cmd.escape, note="; ".join([cmd.note] + notes).strip("; "))
        for n in notes:
            self._event(n)
        self._prev = safe
        return safe
