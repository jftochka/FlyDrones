"""The learning experiment: fly at the chair until the fly wants nothing to do with it.

One lap is one approach. The drone starts at the same place with the same
heading and cruises forward; the looming reflex gets it out of the way, or it
does not and the drone hits the chair. Either way the dopaminergic neuron
fires, the Kenyon cells that were active at the time lose their grip on the
MBON, and on the next lap the avoidance turn is released a little earlier.

Nothing here is a controller. The only thing that changes between lap one and
lap ten is a few hundred synapses inside the mushroom body.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .brain import Brain, load_connectome
from .drones import SimDrone
from .runtime import Pilot

# The chair in the bedroom (see drones/sim.py). The lap starts across the room
# from it, at the height of its back.
CHAIR = (1.45, 0.0)
START = (-2.2, 0.0, 0.9)


@dataclass
class Lap:
    lap: int
    collisions: int
    escapes: int
    closest_m: float
    memory: float
    mbon_hz: float
    turn_hz: float
    seconds: float

    def as_dict(self) -> dict:
        return asdict(self)

    def line(self) -> str:
        return (f"lap {self.lap:2d}  closest {self.closest_m:5.2f} m  collisions {self.collisions}  escapes {self.escapes}"
                f"  memory {self.memory:4.2f}  MBON {self.mbon_hz:5.1f} Hz  turn {self.turn_hz:4.1f} Hz")


def approach_config(cfg: dict, learning: bool = True, cruise: float = 0.9) -> dict:
    """The experiment flies faster than the default governor allows on purpose.

    At the shipped 0.4 m/s the looming reflex clears the chair every time and
    nothing ever hurts, so there is nothing to learn from. The cap is a safety
    policy for real hardware, and this is a simulator.
    """
    cfg = {**cfg}
    cfg["decoder"] = {**cfg.get("decoder", {}), "cruise": cruise}
    cfg["safety"] = {**cfg.get("safety", {}), "max_forward": max(cruise, cfg.get("safety", {}).get("max_forward", 0.4))}
    cfg["cognition"] = {**cfg.get("cognition", {})}
    cfg["cognition"]["mushroom_body"] = {**cfg["cognition"].get("mushroom_body", {}), "enabled": bool(learning)}
    return cfg


def make_pilot(cfg: dict, seed: int = 0) -> Pilot:
    brain = Brain(load_connectome(cfg["brain"]["source"]), cfg)
    pilot = Pilot(brain, SimDrone(start=(START[0], START[1], 0.0), seed=seed), cfg, name="learner")
    pilot.drone.connect()
    pilot.warmup(pilot.decoder.settle_s + 0.1, 1.0 / float(cfg["control"]["hz"]))
    pilot.drone.takeoff()
    return pilot


def fly_lap(pilot: Pilot, index: int, t0: float = 0.0, seconds: float = 9.0, hz: float = 20.0, on_tick=None) -> Lap:
    """One approach, from the same place, until it is past the chair or out of time."""
    dt = 1.0 / hz
    d = pilot.drone
    d.pos = np.array(START, dtype=float)
    d.vel[:] = 0.0
    d.yaw = 0.0
    d.yaw_rate = 0.0
    if pilot.cognition is not None:
        pilot.cognition.reset()  # the bump is planted again; the memory is not touched
    hits0, escapes, escaping = d.collisions, 0, False
    closest, t, n = 9.9, 0.0, 0
    memory = mbon = turn = 0.0
    while t < seconds:
        info = pilot.tick(t0 + t, dt)
        for _ in range(4):
            d.step(dt / 4)
        if on_tick:
            on_tick(pilot, info, index)
        closest = min(closest, float(math.dist(d.pos[:2], CHAIR)))
        if info.cmd.escape and not escaping:
            escapes += 1
        escaping = bool(info.cmd.escape)
        if info.cognition is not None:
            memory = info.cognition.memory
            mbon += info.cognition.mbon_hz
            turn += info.cognition.turn_hz
        n += 1
        t += dt
        if d.pos[0] > CHAIR[0] + 0.5:  # past it
            break
    return Lap(index, d.collisions - hits0, escapes, closest, memory, mbon / max(1, n), turn / max(1, n), t)


def approach_laps(cfg: dict, laps: int = 10, learning: bool = True, seed: int = 0, on_lap=None, on_tick=None) -> list[Lap]:
    """Fly ``laps`` approaches with one brain, learning between them."""
    cfg = approach_config(cfg, learning)
    pilot = make_pilot(cfg, seed=seed)
    out: list[Lap] = []
    t = 0.0
    for i in range(int(laps)):
        lap = fly_lap(pilot, i + 1, t0=t, hz=float(cfg["control"]["hz"]), on_tick=on_tick)
        t += lap.seconds + 0.5
        out.append(lap)
        if on_lap:
            on_lap(lap, pilot)
    return out


def summarise(laps: list[Lap], window: int = 3) -> dict:
    """First few laps against the last few: the whole experiment in six numbers."""
    if not laps:
        return {}
    window = max(1, min(int(window), len(laps) // 2))  # never compare a run against itself
    first, last = laps[:window], laps[-window:]
    mean = lambda rows, key: float(np.mean([getattr(r, key) for r in rows]))  # noqa: E731
    return {
        "laps": len(laps),
        "closest_first_m": mean(first, "closest_m"),
        "closest_last_m": mean(last, "closest_m"),
        "collisions_first": sum(r.collisions for r in first),
        "collisions_last": sum(r.collisions for r in last),
        "mbon_first_hz": mean(first, "mbon_hz"),
        "mbon_last_hz": mean(last, "mbon_hz"),
        "turn_first_hz": mean(first, "turn_hz"),
        "turn_last_hz": mean(last, "turn_hz"),
        "memory": laps[-1].memory,
    }
