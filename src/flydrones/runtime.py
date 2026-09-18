"""The closed loop: eyes -> brain -> decoder -> safety -> drone -> eyes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .brain import Brain
from .brain.cognition import Cognition, CognitiveState
from .drones.base import Drone
from .drones.sim import SimDrone
from .motor import FlightCommand, MotorDecoder
from .safety import SafetyGovernor, Telemetry
from .senses import GestureIllusion, InputEncoder, Retina
from .senses.gestures import GestureState


@dataclass
class TickInfo:
    t: float
    frame: np.ndarray | None
    rates: dict[str, float]
    raw: FlightCommand
    cmd: FlightCommand
    tel: Telemetry
    gesture: GestureState | None
    illusion: str
    raster: list = field(default_factory=list)
    rtf: float = float("nan")
    spikes: int = 0
    brain_ms: float = 0.0  # brain clock at the end of this tick
    cognition: CognitiveState | None = None  # what the memory and the compass say


class Pilot:
    """One brain flying one drone."""

    def __init__(self, brain: Brain, drone: Drone, cfg: dict, gestures=None, webcam=None, name: str = "fly-1"):
        self.name = name
        self.brain = brain
        self.drone = drone
        self.cfg = cfg
        self.gestures = gestures
        self.webcam = webcam
        self.retina = Retina.from_config(cfg)
        self.encoder = InputEncoder(brain.connectome, cfg)
        self.decoder = MotorDecoder(cfg)
        self.safety = SafetyGovernor(cfg)
        self.illusion = GestureIllusion()
        self.cognition = Cognition(brain, cfg) if cfg.get("cognition", {}).get("enabled", True) else None
        self.history: list[dict] = []
        self._t0 = None
        self._collisions = 0
        self._escaping = False

    def warmup(self, seconds: float, dt: float = 0.05) -> None:
        """Let the brain settle on the ground (still scene) and measure resting rates."""
        from .motor.command import FlightCommand as _FC

        t = -seconds
        while t < 0:
            frame = self.drone.frame() if self.drone.has_camera else None
            vision = self.retina.encode(frame)
            extra = self.cognition.extra(vision, dt) if self.cognition else None
            inputs = self.encoder.encode(vision, 0.0, extra=extra)
            rates = self.brain.tick(inputs, ms=dt * 1000.0)
            if self.cognition:  # the compass bump is planted while the quad is still on the ground
                self.cognition.update(self.brain.last_counts, dt * 1000.0, rates)
            self.decoder.update(rates, dt)
            t += dt
        self.drone.send(_FC.hover("warmup done"))

    def tick(self, t: float, dt: float) -> TickInfo:
        frame = self.drone.frame() if self.drone.has_camera else None
        cam = self.webcam.read() if self.webcam is not None else None
        vision = self.retina.encode(frame)
        g = None
        if self.gestures is not None:
            g = self.gestures.read(t, cam)
            vision = self.illusion.apply(vision, g, t)
        tel = self.drone.telemetry()
        hits = int(getattr(self.drone, "collisions", 0))
        if self.cognition is not None:
            if hits > self._collisions:  # something hurt: dopamine, and the scene it happened in
                self.cognition.punish(float(self.cfg.get("cognition", {}).get("collision_punishment", 1.0)))
            extra = self.cognition.extra(vision, dt)
        else:
            extra = None
        self._collisions = hits
        inputs = self.encoder.encode(vision, tel.yaw_rate_dps, extra=extra)
        rates = self.brain.tick(inputs, ms=dt * 1000.0)
        cog = self.cognition.update(self.brain.last_counts, dt * 1000.0, rates) if self.cognition else None
        raw = self.decoder.update(rates, dt)
        if self.cognition is not None:
            # A near miss teaches too. The giant fiber firing is the fly's own
            # report that something was about to hit it, and PPL1 dopaminergic
            # neurons carry threat as well as contact — so the escape is worth a
            # smaller dose of dopamine than a collision. Engineered, not measured
            # from a fly: cognition.punish_on_escape turns it off.
            if raw.escape and not self._escaping:
                self.cognition.punish(float(self.cfg.get("cognition", {}).get("punish_on_escape", 0.0)))
            self._escaping = bool(raw.escape)
        cmd = self.safety.filter(raw, tel, dt)
        if self.safety.land_requested:
            self.drone.land()
        else:
            self.drone.send(cmd)
        self.history.append({"t": t, "alt": tel.alt_m, "x": tel.x_m, "y": tel.y_m, "yaw": tel.yaw_deg,
                             **({"heading": cog.heading_deg, "memory": cog.memory, "mbon": cog.mbon_hz} if cog else {}), **{f"cmd_{k}": getattr(cmd, k) for k in ("throttle", "yaw", "forward")},
                             "escape": cmd.escape, **{f"hz_{k}": v for k, v in rates.items() if k.startswith("DN")}})
        return TickInfo(t, cam if cam is not None else frame, rates, raw, cmd, tel, g, self.illusion.mode if g is not None else "camera",
                        self.brain.last_raster, self.brain.realtime_factor, int(self.brain.last_counts.sum()),
                        self.brain.net.t_ms, cog)


def run_sim(pilots: list[Pilot], seconds: float, hz: float = 20.0, on_tick=None, physics_substeps: int = 4) -> list[list[TickInfo]]:
    """Run pilots whose drones are SimDrones in simulated time (no sleeping)."""
    dt = 1.0 / hz
    out: list[list[TickInfo]] = [[] for _ in pilots]
    for p in pilots:
        p.drone.connect()
        p.warmup(p.decoder.settle_s + 0.1, dt)
        if p.cfg.get("control", {}).get("takeoff", True):
            p.drone.takeoff()
    steps = int(seconds * hz)
    for k in range(steps):
        t = k * dt
        infos = []
        for i, p in enumerate(pilots):
            info = p.tick(t, dt)
            out[i].append(info)
            infos.append(info)
            assert isinstance(p.drone, SimDrone)
            for _ in range(physics_substeps):
                p.drone.step(dt / physics_substeps)
        if on_tick:
            on_tick(k, infos)
    return out


def run_realtime(pilot: Pilot, seconds: float | None = None, hz: float = 20.0, on_tick=None) -> None:
    """Fly real hardware. Ctrl+C lands."""
    dt_target = 1.0 / hz
    d = pilot.drone
    d.connect()
    try:
        print(f"warming up the brain for {pilot.decoder.settle_s:.1f} s (drone stays on the ground)...")
        pilot.warmup(pilot.decoder.settle_s + 0.1, dt_target)
        if pilot.cfg.get("control", {}).get("takeoff", True):
            d.takeoff()
        t0 = last = time.monotonic()
        while seconds is None or time.monotonic() - t0 < seconds:
            now = time.monotonic()
            dt = min(0.25, max(1e-3, now - last))
            last = now
            info = pilot.tick(now - t0, dt)
            if on_tick and on_tick(info) is False:
                break
            if pilot.safety.land_requested:
                print("safety: landing ->", "; ".join(pilot.safety.events[-3:]))
                break
            if info.rtf < 0.8:
                print(f"warning: brain runs at {info.rtf:.2f}x real time - try a sensorimotor core (build-brain --core-hops 3)")
            sleep = dt_target - (time.monotonic() - now)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nCtrl+C -> landing")
    finally:
        d.send(FlightCommand.hover("stop"))
        d.land()
        d.close()
