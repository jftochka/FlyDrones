"""Radio Cognitive Fruit Fly: a station that never stops, played by a fly brain.

The compositor turns one flight into music. A station needs more than that: a
programme, so the piece changes; a fly that keeps flying, so it does not end
when the pack runs down or the drone lands; and something that says what is
going on, because the interesting part is inside the brain and nobody can see
it.

    flydrones radio                       # open the address it prints
    flydrones radio --to strudel,pd       # and send it to Pure Data as well

Each show sets the key, the tempo, what the hand in front of the camera does,
and whether the mushroom body is learning. The memory carries across shows: the
fly that spent ten minutes learning that the chair hurts arrives at the next
show knowing it, and forgets it slowly, on air.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from .brain import Brain, load_connectome
from .drones import SimDrone
from .learn import CHAIR, START
from .music import Compositor, make_sinks
from .music.conductor import PHRASES, improvisation
from .runtime import Pilot
from .senses import ScriptedGestures

STATION = "Radio Cognitive Fruit Fly"


@dataclass
class Show:
    """One programme item: a key, a tempo, and something for the fly to do."""

    name: str
    minutes: float
    blurb: str
    scale: str = "minor_pentatonic"
    root: str = "C3"
    cps: float = 0.5
    tempo_from: str = "fixed"
    cruise: float = 0.0
    learning: bool = True
    gestures: str = "conductor"  # conductor | quiet
    weights: dict[str, float] = field(default_factory=dict)  # conductor phrase weights
    laps: bool = False  # fly at the chair, again and again
    goal_every_s: float = 0.0  # set a new heading to hold, this often

    @property
    def seconds(self) -> float:
        return self.minutes * 60.0


# The programme. Order matters: The Chair teaches the fly something and
# Afterwards is what that sounds like once nothing is hurting it any more.
PROGRAMME: list[Show] = [
    Show("Dawn Chorus", 6, "hovering, hardly anything happening. DNg02 breathing, and the room going by.",
         scale="minor_pentatonic", root="C3", cps=0.42, weights={"rest": 2.5, "hold": 3.0, "loom": 0.05}),
    Show("The Chair", 9, "the fly flies at a chair until it wants nothing to do with it. Listen for the "
                         "dopamine note, then for the memory voice arriving and the avoidance turn taking over.",
         scale="dorian", root="A2", cps=0.55, cruise=0.9, laps=True, learning=True, gestures="quiet"),
    Show("Afterwards", 5, "the same room, nothing to be afraid of, and a memory quietly creeping back to "
                          "where it started. The memory voice thins out over five minutes.",
         scale="lydian", root="C3", cps=0.38, tempo_from="alt", weights={"rest": 3.0, "hold": 2.0, "loom": 0.0}),
    Show("Compass Rose", 7, "a new heading to hold every half minute. The compass is the fly's own; nothing "
                            "tells it which way it is pointing.",
         scale="whole_tone", root="D3", cps=0.5, goal_every_s=30.0, weights={"loom": 0.1}),
    Show("Swat Hour", 6, "something rushes at the drone every few seconds. Giant fibre, saccades, and the "
                         "looming riser that only exists while the danger does.",
         scale="blues", root="A2", cps=0.75, tempo_from="drive", weights={"loom": 4.0, "rest": 0.4, "hold": 1.0}),
    Show("Night Flight", 7, "slow, low, and mostly lift: the tempo follows the altitude.",
         scale="phrygian", root="F2", cps=0.33, tempo_from="alt", weights={"rest": 3.0, "hold": 3.0, "loom": 0.1}),
]


@dataclass
class LogEntry:
    t: float
    kind: str
    text: str

    def as_dict(self) -> dict:
        return {"t": round(self.t, 1), "kind": self.kind, "text": self.text}


class Station:
    """One fly, one drone, one programme, and whoever is listening."""

    def __init__(self, cfg: dict, targets: str = "print,strudel", seed: int = 0,
                 programme: list[Show] | None = None, log_size: int = 40):
        self.cfg = self._station_config(cfg)
        self.programme = list(programme or PROGRAMME)
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.brain = Brain(load_connectome(self.cfg["brain"]["source"]), self.cfg)
        self.pilot = Pilot(self.brain, SimDrone(start=(-1.6, 0.0, 0.0), seed=seed), self.cfg, name="on-air")
        self.compositor = Compositor(self.cfg, brain=self.brain)
        self.sinks = make_sinks(targets, self.cfg)
        for sink in self.sinks.sinks:  # the browser target gets the station's own front page
            if hasattr(sink, "server"):
                sink.server.page = "radio.html"
        self.hz = float(self.cfg["control"]["hz"])
        self.dt = 1.0 / self.hz
        self.t = 0.0
        self.frames = 0
        self.show_index = -1
        self.show_started = 0.0
        self.entries: list[LogEntry] = []
        self.log_size = int(log_size)
        self.counts = {"escapes": 0, "collisions": 0, "lessons": 0, "goals": 0, "laps": 0, "packs": 0, "shows": 0}
        self._collisions = 0
        self._escaping = False
        self._memory_mark = 0.0
        self._lap_started = 0.0
        self._goal_at = -1e9
        self._started = False
        self.last_info = None  # the most recent tick, for anything recording the flight

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _station_config(cfg: dict) -> dict:
        """A station flies for hours; the shipped limits are for a real drone."""
        cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
        cfg["safety"] = {**cfg.get("safety", {}), "max_flight_s": 1e9}
        cfg["decoder"] = {**cfg.get("decoder", {})}
        cfg["cognition"] = {**cfg.get("cognition", {})}
        cfg["cognition"]["mushroom_body"] = {**cfg["cognition"].get("mushroom_body", {})}
        return cfg

    def start(self) -> Station:
        self.pilot.drone.connect()
        self.pilot.warmup(self.pilot.decoder.settle_s + 0.1, self.dt)
        self.compositor.set_baselines(self.pilot.decoder.baseline)
        self.pilot.drone.takeoff()
        self._started = True
        self.log("station", f"{STATION} on air: {len(self.programme)} shows, {self.brain.n_neurons:,} neurons")
        self._begin_show(0)
        return self

    # -------------------------------------------------------------------- log
    def log(self, kind: str, text: str) -> None:
        self.entries.append(LogEntry(self.t, kind, text))
        del self.entries[: max(0, len(self.entries) - self.log_size)]

    # ------------------------------------------------------------------ shows
    @property
    def show(self) -> Show:
        return self.programme[max(0, self.show_index) % len(self.programme)]

    @property
    def next_show(self) -> Show:
        return self.programme[(max(0, self.show_index) + 1) % len(self.programme)]

    @property
    def show_left_s(self) -> float:
        return max(0.0, self.show.seconds - (self.t - self.show_started))

    def _begin_show(self, index: int) -> None:
        self.show_index = index % len(self.programme)
        self.show_started = self.t
        show = self.show
        self.counts["shows"] += 1

        self.compositor.scale = self.compositor.scale.parse(show.scale, show.root)
        for voice in self.compositor.voices:
            voice.scale = self.compositor.scale
        self.compositor.base_cps = self.compositor.cps = float(show.cps)
        self.compositor.tempo_from = show.tempo_from
        self.pilot.decoder.cruise = float(show.cruise)
        self.pilot.safety.max["forward"] = max(0.4, float(show.cruise))
        if self.pilot.cognition is not None:
            self.pilot.cognition.mb.enabled = bool(show.learning) and self.pilot.cognition.mb.pos.size > 0
            self.pilot.cognition.set_goal(None)
        self.pilot.gestures = self._gestures(show)
        self._lap_started = self.t
        self._goal_at = -1e9
        self.log("show", f"now playing: {show.name} — {show.blurb}")

    def _gestures(self, show: Show):
        if show.gestures == "quiet":
            return None
        weights = {k: v[1] for k, v in PHRASES.items()}
        weights.update(show.weights)
        saved = {k: PHRASES[k] for k in PHRASES}
        try:
            for k, w in weights.items():  # the conductor reads the weights from PHRASES
                if k in PHRASES:
                    g, _old, lo, hi = PHRASES[k]
                    PHRASES[k] = (g, float(w), lo, hi)
            timeline = improvisation(show.seconds + 30.0, seed=self.rng.randrange(1 << 30))
        finally:
            PHRASES.update(saved)
        return ScriptedGestures([(t + self.t, g) for t, g in timeline])

    # ------------------------------------------------------------------- loop
    def _keep_flying(self) -> None:
        """A station cannot land. Swap the pack, pick it up, put it back in the air."""
        d = self.pilot.drone
        if d.battery < 35.0:
            d.battery = 100.0
            self.counts["packs"] += 1
            self.log("pack", f"pack swap number {self.counts['packs']}: the fly keeps flying")
        if self.pilot.safety.land_requested or not d.flying:
            self.pilot.safety.land_requested = False
            self.pilot.safety.events.clear()
            self.pilot.safety._start = None
            d.pos[2] = max(0.9, float(d.pos[2]))
            d.takeoff()
            if self.pilot.cognition is not None:
                self.pilot.cognition.reset()  # the compass bump is planted again
            self.log("recovery", "back in the air (the compass starts again from here)")

    def _run_scenario(self, show: Show) -> None:
        d = self.pilot.drone
        if show.laps:
            past = float(d.pos[0]) > CHAIR[0] + 0.5
            if past or self.t - self._lap_started > 10.0:
                d.pos = np.array(START, dtype=float)
                d.vel[:] = 0.0
                d.yaw = 0.0
                d.yaw_rate = 0.0
                if self.pilot.cognition is not None:
                    self.pilot.cognition.reset()
                self._lap_started = self.t
                self.counts["laps"] += 1
                if self.counts["laps"] % 3 == 0:
                    memory = self.pilot.cognition.state.memory if self.pilot.cognition else 0.0
                    self.log("lap", f"approach {self.counts['laps']}: memory {memory:.2f}")
        if show.goal_every_s > 0 and self.t - self._goal_at > show.goal_every_s and self.pilot.cognition is not None:
            self._goal_at = self.t
            heading = self.pilot.cognition.state.heading_deg
            if not math.isnan(heading):
                goal = (heading + self.rng.choice([-135, -90, -60, 60, 90, 135])) % 360
                self.pilot.cognition.set_goal(goal)
                self.counts["goals"] += 1
                self.log("goal", f"new heading to hold: {goal:.0f}° (it is at {heading:.0f}°)")

    def _watch(self, info) -> None:
        """Everything worth saying out loud, from one tick of the flight."""
        cog = info.cognition
        if info.cmd.escape and not self._escaping:
            self.counts["escapes"] += 1
            self.log("escape", "the giant fibre fired")
        self._escaping = bool(info.cmd.escape)
        hits = int(getattr(self.pilot.drone, "collisions", 0))
        if hits > self._collisions:
            self.counts["collisions"] += 1
            self.log("bump", "it hit something, and the dopamine says so")
        self._collisions = hits
        if cog is None:
            return
        if cog.memory > self._memory_mark + 0.08:
            self._memory_mark = cog.memory
            self.counts["lessons"] += 1
            self.log("learning", f"it has learned something: memory {cog.memory:.2f}, MBON down to {cog.mbon_hz:.0f} Hz")
        elif cog.memory < self._memory_mark - 0.08:
            self._memory_mark = cog.memory
            self.log("forgetting", f"the memory is fading: {cog.memory:.2f}")
        if cog.goal_deg is not None and abs(cog.heading_error_deg) < 12 and self.t - self._goal_at > 4:
            self._goal_at = self.t  # reached: hold it until the next one is due
            self.log("goal", f"it is holding {cog.goal_deg:.0f}°")

    def tick(self):
        """One control tick: fly, compose, send. Returns the frame."""
        if not self._started:
            self.start()
        if self.t - self.show_started >= self.show.seconds:
            self._begin_show(self.show_index + 1)
        show = self.show
        self._keep_flying()
        self._run_scenario(show)
        info = self.pilot.tick(self.t, self.dt)
        for _ in range(4):
            self.pilot.drone.step(self.dt / 4)
        self.last_info = info
        self._watch(info)
        frame = self.compositor.tick(info)
        self.sinks.set_context(self.context())
        self.sinks.frame(frame)
        self.t += self.dt
        self.frames += 1
        return frame

    def run(self, seconds: float | None = None, on_tick=None) -> None:
        """Real time, until Ctrl+C or ``seconds``. The clock is the point."""
        if not self._started:
            self.start()
        t0 = time.monotonic()
        behind = 0
        try:
            while seconds is None or self.t < seconds:
                frame = self.tick()
                if on_tick:
                    on_tick(self, frame)
                slack = (t0 + self.t) - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
                elif slack < -1.0:
                    behind += 1
                    t0 = time.monotonic() - self.t
                    if behind in (1, 10, 100):
                        self.log("slow", f"the brain is behind real time ({self.brain.realtime_factor:.1f}x); the music drags")
        except KeyboardInterrupt:
            self.log("station", "off air")
        finally:
            self.close()

    def close(self) -> None:
        self.sinks.close()

    # ---------------------------------------------------------------- reading
    def context(self) -> dict:
        """What the page shows: who is on air, what is playing, what happened."""
        cog = self.pilot.cognition.state if self.pilot.cognition else None
        show = self.show
        return {
            "station": STATION,
            "show": {"name": show.name, "blurb": show.blurb, "left_s": round(self.show_left_s, 1),
                     "length_s": show.seconds, "scale": f"{show.scale} / {show.root}", "learning": show.learning},
            "next": self.next_show.name,
            "uptime_s": round(self.t, 1),
            "counts": dict(self.counts),
            "brain": {
                "neurons": int(self.brain.n_neurons),
                "memory": round(float(cog.memory), 3) if cog else 0.0,
                "heading": None if cog is None or math.isnan(cog.heading_deg) else round(cog.heading_deg, 1),
                "goal": None if cog is None or cog.goal_deg is None else round(cog.goal_deg, 1),
                "mbon_hz": round(float(cog.mbon_hz), 1) if cog else 0.0,
                "kc_active": round(float(cog.kc_active), 3) if cog else 0.0,
                "realtime": round(float(self.brain.realtime_factor), 2),
            },
            "log": [e.as_dict() for e in self.entries[-14:]],
        }

    def now_playing(self) -> str:
        cog = self.pilot.cognition.state if self.pilot.cognition else None
        bits = [f"{STATION} · {self.show.name}", f"{self.show_left_s / 60:.1f} min left"]
        if cog is not None:
            bits.append(f"memory {cog.memory:.2f}")
            if not math.isnan(cog.heading_deg):
                bits.append(f"heading {cog.heading_deg:.0f}°")
        bits.append(f"{self.counts['escapes']} escapes")
        return "  |  ".join(bits)
