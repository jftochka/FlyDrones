"""Measure the two cognitive circuits instead of arguing about them.

    python tools/bench_cognition.py            # everything
    python tools/bench_cognition.py compass    # heading gain, drift, goal steering
    python tools/bench_cognition.py memory     # learning, specificity, forgetting
    python tools/bench_cognition.py laps       # the flight experiment, with a control

Every number printed here is the reason a constant in `brain/synthetic.py` has
the value it has. The sweeps that chose them are one-liners away: change a key
in `MB` or `CX` at the top of that file and run this again.
"""

from __future__ import annotations

import sys

import numpy as np

from flydrones.brain import Brain, build_minifly
from flydrones.brain.cognition import Cognition
from flydrones.config import load_config
from flydrones.drones import SimDrone
from flydrones.learn import approach_laps, summarise
from flydrones.motor import FlightCommand
from flydrones.runtime import Pilot

COLUMNS = 16


def _unwrap(a) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(np.asarray(a, dtype=float))))


def _pilot(cfg, seed=0, start=(0.0, 0.0, 0.0)):
    p = Pilot(Brain(build_minifly(), cfg), SimDrone(start=start, seed=seed), cfg)
    p.drone.connect()
    p.warmup(p.decoder.settle_s + 0.1, 0.05)
    p.drone.takeoff()
    return p


# --------------------------------------------------------------------- compass
def compass(seconds: float = 13.0, dt: float = 0.05) -> None:
    print("\n== compass ==  the bump against the body it is riding on")
    print(f"{'yaw command':>11} {'body dps':>8} {'gain':>6} {'drift deg/s':>11} {'bump':>5}")
    for cmd in (0.15, 0.4, -0.4, 0.6):
        p = _pilot(load_config())
        rows = []
        for k in range(int(seconds / dt)):
            t = k * dt
            info = p.tick(t, dt)
            p.drone.send(FlightCommand(yaw=(cmd if 3 < t < 11 else 0.0)))
            for _ in range(4):
                p.drone.step(dt / 4)
            tel = p.drone.telemetry()
            rows.append((t, tel.yaw_deg, info.cognition.heading_deg, info.cognition.heading_strength, tel.yaw_rate_dps))
        t, yaw, head, strength, rate = (np.array(x) for x in zip(*rows))
        yaw_u, head_u = _unwrap(yaw), _unwrap(head)
        turning, still = (t > 3.5) & (t < 10.8), t < 2.8
        gain = (head_u[turning][-1] - head_u[turning][0]) / (yaw_u[turning][-1] - yaw_u[turning][0] + 1e-9)
        drift = (head_u[still][-1] - head_u[still][0]) / (t[still][-1] - t[still][0])
        print(f"{cmd:11.2f} {np.abs(rate[turning]).mean():8.1f} {gain:6.2f} {drift:11.2f} {strength.mean():5.2f}")
    print("  gain 1.0 would be a protractor; this is a heading, and it is the same one either way round.")


def goal(seconds: float = 30.0, dt: float = 0.05) -> None:
    print("\n== goal steering ==  PFL3 turning the body until the bump reaches the goal")
    print(f"{'goal':>6} {'|error| start':>13} {'|error| end':>11} {'body turned':>11}")
    for offset in (90.0, -90.0, 150.0):
        p = _pilot(load_config())
        goal_deg, rows = None, []
        for k in range(int(seconds / dt)):
            t = k * dt
            info = p.tick(t, dt)
            for _ in range(4):
                p.drone.step(dt / 4)
            c = info.cognition
            if goal_deg is None and t > 3.0 and not np.isnan(c.heading_deg):
                goal_deg = (c.heading_deg + offset) % 360.0
                p.cognition.set_goal(goal_deg)
            rows.append((t, p.drone.telemetry().yaw_deg, c.heading_error_deg))
        t, yaw, err = (np.array(x) for x in zip(*rows))
        e = np.abs(err[~np.isnan(err)])
        yaw_u = _unwrap(yaw)
        print(f"{offset:6.0f} {e[5]:13.1f} {e[-40:].mean():11.1f} {yaw_u[-1] - yaw_u[0]:11.1f}")


# ---------------------------------------------------------------------- memory
def memory(ms: float = 400.0) -> None:
    print("\n== mushroom body ==  one scene punished, another one left alone")
    cfg = load_config()
    b = Brain(build_minifly(), cfg, seed=0)
    cog = Cognition(b, cfg)

    def a_scene(seed: int, n: int = 24, fraction: float = 0.35) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = np.zeros(n, np.float32)
        v[rng.choice(n, int(n * fraction), replace=False)] = 1.0
        return v

    A, B = a_scene(1), a_scene(2)

    def present(scene, punish=0.0, repeats=4):
        rows = []
        for _ in range(repeats):
            r = b.tick({"VPN_L": 70 * scene, "VPN_R": 70 * scene,
                        "PPL1_L": 140 * punish, "PPL1_R": 140 * punish}, ms)
            cog.update(b.last_counts, ms, r)
            rows.append((cog.state.kc_active, r["MBON_L"], r["TURN_L"], r["DNp03_L"], r["DNp01_L"]))
        return np.array(rows[1:]).mean(0)

    b.tick({}, 400)
    print(f"{'':22} {'KCs firing':>10} {'MBON':>6} {'turn':>6} {'DNp03':>6} {'DNp01':>6} {'memory':>7}")

    def line(tag, v):
        print(f"{tag:22} {v[0]:10.2f} {v[1]:6.1f} {v[2]:6.1f} {v[3]:6.1f} {v[4]:6.1f} {cog.mb.strength:7.2f}")

    line("naive, scene A", present(A))
    line("naive, scene B", present(B))
    for i in range(3):
        line(f"scene A + dopamine {i + 1}", present(A, punish=1.0))
    line("learned, scene A", present(A))
    line("learned, scene B", present(B))
    loom = {"LPLC2_L": 150.0, "LPLC2_R": 150.0, "LC4_L": 150.0, "LC4_R": 150.0}
    rows = []
    for _ in range(3):
        r = b.tick({"VPN_L": 70 * B, "VPN_R": 70 * B, **loom}, ms)
        cog.update(b.last_counts, ms, r)
        rows.append((cog.state.kc_active, r["MBON_L"], r["TURN_L"], r["DNp03_L"], r["DNp01_L"]))
    line("reflex (looming)", np.array(rows[1:]).mean(0))
    print("  the turn is released for the scene that was punished, and for that scene only;")
    print("  the giant fiber never learned anything and does not need to.")


# ------------------------------------------------------------------------ laps
def laps(n: int = 10) -> None:
    print(f"\n== the experiment ==  {n} approaches to the chair, with and without the plasticity")
    cfg = load_config()
    for label, learning in (("learning", True), ("control", False)):
        print(f"\n  --- {label} ---")
        rows = approach_laps(cfg, laps=n, learning=learning, on_lap=lambda lap, _p: print("    " + lap.line()))
        s = summarise(rows)
        print(f"    closest {s['closest_first_m']:.2f} -> {s['closest_last_m']:.2f} m | "
              f"collisions {s['collisions_first']} -> {s['collisions_last']} | "
              f"MBON {s['mbon_first_hz']:.1f} -> {s['mbon_last_hz']:.1f} Hz | "
              f"turn {s['turn_first_hz']:.1f} -> {s['turn_last_hz']:.1f} Hz | memory {s['memory']:.2f}")


def main(argv: list[str]) -> int:
    which = argv[1:] or ["compass", "goal", "memory", "laps"]
    for name in which:
        {"compass": compass, "goal": goal, "memory": memory, "laps": laps}[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
