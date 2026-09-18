"""The mushroom body and the central complex, at the neuron and at the flight level."""

from __future__ import annotations

import numpy as np
import pytest

from flydrones.brain import Brain, build_minifly
from flydrones.brain.cognition import Cognition, SceneCode, bump_vector
from flydrones.config import load_config
from flydrones.drones import SimDrone
from flydrones.learn import approach_laps, summarise
from flydrones.motor import FlightCommand
from flydrones.runtime import Pilot
from flydrones.senses.retina import EyeFeatures, VisualFrame

COLUMNS = 16


def brain_and_cognition(**over):
    cfg = load_config()
    for key, value in over.items():
        cfg["cognition"]["mushroom_body"][key] = value
    b = Brain(build_minifly(), cfg, seed=0)
    return b, Cognition(b, cfg), cfg


def a_scene(seed: int, n: int = 24, fraction: float = 0.35) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.zeros(n, np.float32)
    v[rng.choice(n, int(n * fraction), replace=False)] = 1.0
    return v


def present(b, cog, scene, punish=0.0, repeats=4, ms=400.0, extra=None):
    rows = []
    for _ in range(repeats):
        inputs = {"VPN_L": 70 * scene, "VPN_R": 70 * scene, "PPL1_L": 140 * punish, "PPL1_R": 140 * punish}
        inputs.update(extra or {})
        rates = b.tick(inputs, ms)
        cog.update(b.last_counts, ms, rates)
        rows.append(rates)
    return {k: float(np.mean([r[k] for r in rows[1:]])) for k in rows[0]}


# ------------------------------------------------------------------ the pieces
def test_bump_vector_is_a_bump_around_a_ring():
    v = bump_vector(COLUMNS, 0)
    assert v.argmax() == 0
    assert v[1] == pytest.approx(v[-1])  # the ring wraps: both neighbours are equal
    assert bump_vector(COLUMNS, 8).argmax() == 8
    assert 0.0 <= v.min() and v.max() == pytest.approx(1.0)


def test_scene_code_is_contrast_not_brightness():
    def frame(bright: float, pattern: np.ndarray) -> VisualFrame:
        grids = {"brightness": (bright + pattern).astype(np.float32)}
        return VisualFrame({e: EyeFeatures(dict(grids)) for e in "LR"})

    pattern = np.zeros((6, 8), np.float32)
    pattern[2:4, 3:6] = 0.4
    code = SceneCode()
    dark, light = code(frame(0.1, pattern))["scene_L"], code(frame(0.6, pattern))["scene_L"]
    assert np.allclose(dark, light)  # the same room at two exposures is the same memory
    assert 0.0 < (dark > 0).mean() < 0.6  # and it is sparse
    flat = code(frame(0.3, np.zeros((6, 8), np.float32)))["scene_L"]
    assert flat.max() <= 1.0


def test_synapse_access_keeps_both_copies_of_the_weights():
    b, _, _ = brain_and_cognition()
    kc = b.connectome.group("KC_L")
    mbon = b.connectome.group("MBON_L")
    pos, owner = b.net.synapses_between(kc, mbon)
    assert pos.size > 50 and owner.size == pos.size
    b.net.set_synapses(pos, b.net.get_synapses(pos) * 0.5)
    assert np.allclose(b.net._data[pos], b.net.W.data[pos] * b.net.p.w_syn)


# ------------------------------------------------------------ mushroom body
def test_the_kenyon_cell_code_is_sparse():
    b, cog, _ = brain_and_cognition()
    b.tick({}, 400)
    present(b, cog, a_scene(1))
    assert 0.01 < cog.state.kc_active < 0.30, cog.state.kc_active


def test_punishment_is_specific_to_the_scene_that_hurt():
    b, cog, _ = brain_and_cognition()
    b.tick({}, 400)
    A, B = a_scene(1), a_scene(2)
    naive_a = present(b, cog, A)["MBON_L"]
    naive_b = present(b, cog, B)["MBON_L"]
    assert naive_a > 10 and naive_b > 10
    for _ in range(3):
        present(b, cog, A, punish=1.0)
    learned_a, learned_b = present(b, cog, A), present(b, cog, B)
    assert learned_a["MBON_L"] < 0.3 * naive_a  # the MBON lets go of the punished scene
    assert learned_b["MBON_L"] > 0.6 * naive_b  # and keeps holding the other one
    assert learned_a["TURN_L"] > 5.0  # the avoidance turn is released
    assert learned_b["TURN_L"] < 1.0
    assert learned_a["DNp03_L"] > 5.0  # and reaches the descending neuron that steers
    assert cog.mb.strength > 0.05


def test_the_reflex_does_not_learn_and_does_not_need_to():
    b, cog, _ = brain_and_cognition()
    b.tick({}, 400)
    A = a_scene(1)
    for _ in range(3):
        present(b, cog, A, punish=1.0)
    loom = {"LPLC2_L": 150.0, "LPLC2_R": 150.0, "LC4_L": 150.0, "LC4_R": 150.0}
    r = present(b, cog, a_scene(2), extra=loom, repeats=3)
    assert r["DNp01_L"] > 20 and r["DNp03_L"] > 20  # giant fiber and saccade, untouched


def test_a_memory_fades():
    b, cog, _ = brain_and_cognition(forget_s=2.0)
    b.tick({}, 400)
    A = a_scene(1)
    for _ in range(3):
        present(b, cog, A, punish=1.0)
    hot = cog.mb.strength
    assert hot > 0.05
    for _ in range(6):  # 4.8 s of quiet, more than two time constants
        present(b, cog, A * 0, repeats=2)
    assert cog.mb.strength < 0.5 * hot


def test_learning_can_be_switched_off():
    b, cog, _ = brain_and_cognition(enabled=False)
    b.tick({}, 400)
    A = a_scene(1)
    naive = present(b, cog, A)["MBON_L"]
    for _ in range(3):
        present(b, cog, A, punish=1.0)
    assert cog.mb.strength == 0.0
    assert present(b, cog, A)["MBON_L"] > 0.7 * naive


# ------------------------------------------------------------ central complex
def seed_bump(b, column: int = 4, ms: float = 400.0):
    per = b.connectome.group("EPG").size // COLUMNS
    b.tick({"EPG_SET": 120.0 * np.repeat(bump_vector(COLUMNS, column), per)}, ms)


def test_the_ring_holds_one_bump_and_keeps_it():
    b, cog, _ = brain_and_cognition()
    b.tick({}, 400)
    seed_bump(b, 4)
    b.tick({}, 600)
    heading, strength = cog.compass.read(b.last_counts, 600)
    assert strength > 0.5, "no bump"
    assert abs(((heading - 90.0) + 180) % 360 - 180) < 40  # column 4 of 16 is 90 degrees
    for _ in range(4):  # two more seconds with nothing driving it
        b.tick({}, 500)
        held, strength = cog.compass.read(b.last_counts, 500)
    assert strength > 0.5
    assert abs(((held - heading) + 180) % 360 - 180) < 30, "the bump wandered off"


def test_the_halteres_push_the_bump_both_ways():
    """Yaw in, rotation out. Accumulated, because the bump goes all the way round."""
    b, cog, _ = brain_and_cognition()
    b.tick({}, 400)
    seed_bump(b, 8)
    b.tick({}, 400)

    def rotate(drive: str, ticks: int, hz: float = 25.0) -> float:
        # Half drive and short windows on purpose: at full haltere rate the bump
        # can cross half the ring inside one window, and then the accumulation
        # below cannot tell which way round it went.
        last = cog.compass.read(b.last_counts, 300)[0]
        total = 0.0
        for _ in range(ticks):
            b.tick({drive: hz} if drive else {}, 300)
            now = cog.compass.read(b.last_counts, 300)[0]
            total += ((now - last) + 180) % 360 - 180
            last = now
        return total

    right, left = rotate("HAL_R", 8), rotate("HAL_L", 10)
    assert right < -40, f"a right turn rotated the bump {right:.0f} deg"
    assert left > 40, f"a left turn rotated it {left:.0f} deg"
    assert abs(rotate("", 4)) < 40, "the bump kept going after the halteres went quiet"


def test_the_compass_follows_the_body_and_holds_still_when_it_does():
    """A flight, not a poke: the bump has to track the drone it is riding on."""
    cfg = load_config()
    p = Pilot(Brain(build_minifly(), cfg), SimDrone(start=(0, 0, 0)), cfg)
    p.drone.connect()
    p.warmup(1.6, 0.05)
    p.drone.takeoff()
    still, turned = [], []
    for k in range(int(11 / 0.05)):
        t = k * 0.05
        info = p.tick(t, 0.05)
        p.drone.send(FlightCommand(yaw=0.4 if t > 3.0 else 0.0))
        for _ in range(4):
            p.drone.step(0.05 / 4)
        (turned if t > 3.0 else still).append((p.drone.telemetry().yaw_deg, info.cognition.heading_deg))
    unwrap = lambda a: np.degrees(np.unwrap(np.radians(np.array(a, dtype=float))))  # noqa: E731
    head_still = unwrap([r[1] for r in still])
    yaw_turn, head_turn = unwrap([r[0] for r in turned]), unwrap([r[1] for r in turned])
    assert abs(head_still[-1] - head_still[0]) < 15, "the heading drifted while the drone sat still"
    body = yaw_turn[-1] - yaw_turn[0]
    bump = head_turn[-1] - head_turn[0]
    assert abs(body) > 90
    assert np.sign(bump) == np.sign(body), "the compass turned the wrong way"
    assert 0.4 < abs(bump / body) < 4.0, f"gain {bump / body:.2f} is not a heading"


def test_a_goal_turns_the_drone_until_the_bump_reaches_it():
    cfg = load_config()
    p = Pilot(Brain(build_minifly(), cfg), SimDrone(start=(0, 0, 0)), cfg)
    p.drone.connect()
    p.warmup(1.6, 0.05)
    p.drone.takeoff()
    errors = []
    for k in range(int(26 / 0.05)):
        info = p.tick(k * 0.05, 0.05)
        for _ in range(4):
            p.drone.step(0.05 / 4)
        c = info.cognition
        if p.cognition.goal_deg is None and k * 0.05 > 3.0 and not np.isnan(c.heading_deg):
            p.cognition.set_goal((c.heading_deg - 90.0) % 360)  # a quarter turn to starboard
        elif p.cognition.goal_deg is not None:
            errors.append(abs(c.heading_error_deg))
    errors = [e for e in errors if not np.isnan(e)]
    assert errors[0] > 60
    assert min(errors) < 30, f"the fly never reached its goal (best {min(errors):.0f} deg)"
    assert np.mean(errors[-40:]) < np.mean(errors[:40])


# -------------------------------------------------------------------- in flight
def test_learning_keeps_the_drone_away_from_the_chair():
    cfg = load_config()
    laps = approach_laps(cfg, laps=5, learning=True)
    s = summarise(laps, window=2)
    assert laps[-1].memory > 0.1, "nothing was learned"
    assert s["mbon_last_hz"] < s["mbon_first_hz"], "the MBON did not let go"
    assert s["turn_last_hz"] > s["turn_first_hz"] + 1.0, "the avoidance turn was never released"
    assert s["closest_last_m"] > s["closest_first_m"], "it did not keep further away"


def test_cognition_switches_off_cleanly():
    cfg = load_config()
    cfg["cognition"]["enabled"] = False
    p = Pilot(Brain(build_minifly(), cfg), SimDrone(start=(0, 0, 0)), cfg)
    p.drone.connect()
    p.warmup(0.3, 0.05)
    info = p.tick(0.0, 0.05)
    assert p.cognition is None and info.cognition is None
