"""What the fly knows: a memory of what hurt, and a compass.

Two circuits sit beside the reflex pathways, and this module is the part of
them that is not wiring: the read-outs, the sparse code the mushroom body is
given, and the one rule that changes a synapse.

**The memory.** Visual projection neurons drive Kenyon cells through a random,
sparse expansion; APL feedback keeps only a few percent of them firing, so the
scene in front of the drone becomes a sparse code. Those Kenyon cells drive
MBON-g1pedc, which is GABAergic and holds an avoidance turn *off*. When
something hurts, PPL1 fires dopamine, and the Kenyon-cell synapses that were
active at that moment are depressed (Hige et al. 2015; Aso & Rubin 2016). The
next time that scene appears the MBON stays quiet, the turn is released, and
the fly steers away from something it has only ever seen — the reflex needs the
thing to be already rushing at it.

Depression is specific to the active code, which is the whole point: another
scene drives other Kenyon cells, whose synapses were never depressed.

**The compass.** EPG columns form a ring: each column excites itself and its
neighbours, Delta7 inhibits everything, and what survives is a single bump.
PEN cells, driven by the halteres, push the bump one column at a time, so it
integrates turns into a heading (Seelig & Jayaraman 2015; Green et al. 2017;
Turner-Evans et al. 2017). The heading here is the population vector of the
bump, in the ring's own frame: it is set when the bump is planted at take-off
and drifts slowly afterwards, exactly like a fly in the dark.

PFL3 compares the bump against the goal column held by FC2 and steers until
the two line up (Hulse et al. 2021; Westeinde et al. 2024). Because the
comparison happens in the ring's frame, an uncalibrated compass still holds a
heading: the fly flies the direction the goal was set to, whatever that is in
the world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _wrap180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def bump_vector(columns: int, column: float, width: float = 1.0, amplitude: float = 1.0) -> np.ndarray:
    """A smooth bump of drive centred on ``column``, around a ring."""
    d = (np.arange(columns) - float(column) + columns / 2) % columns - columns / 2
    return (amplitude * np.exp(-0.5 * (d / max(1e-6, width)) ** 2)).astype(np.float32)


@dataclass
class CognitiveState:
    """What the two circuits think, once per control tick."""

    heading_deg: float = float("nan")  # bump position, in the ring's own frame
    heading_strength: float = 0.0  # 0 = no bump, 1 = all the activity in one column
    goal_deg: float | None = None
    heading_error_deg: float = float("nan")
    memory: float = 0.0  # 0 = naive, 1 = the KC->MBON synapses are fully depressed
    dopamine_hz: float = 0.0
    mbon_hz: float = 0.0
    turn_hz: float = 0.0  # the learned avoidance turn, once the MBON lets go
    kc_active: float = 0.0  # fraction of Kenyon cells firing: the code's sparseness
    punished: bool = False

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["goal_deg"] = None if self.goal_deg is None else round(self.goal_deg, 1)
        return d


class SceneCode:
    """Brightness grids -> the drive that lands on the visual projection neurons.

    Contrast, not absolute brightness: a code that tracked the lamp would make
    every dark corner of the room the same memory. Each eye is normalised on its
    own and the quietest cells are dropped, which is the first sparsening step —
    the Kenyon cells and APL do the rest.
    """

    def __init__(self, keep: float = 0.45, gain: float = 1.6):
        self.keep = float(keep)
        self.gain = float(gain)

    def __call__(self, vision) -> dict[str, np.ndarray]:
        out = {}
        for eye in ("L", "R"):
            grid = None if vision is None else vision.eyes.get(eye)
            if grid is None:
                out[f"scene_{eye}"] = np.zeros(1, dtype=np.float32)
                continue
            b = np.asarray(grid.grids.get("brightness"), dtype=np.float32).reshape(-1)
            if b.size == 0 or not np.isfinite(b).all():
                out[f"scene_{eye}"] = np.zeros(1, dtype=np.float32)
                continue
            z = (b - b.mean()) / (b.std() + 1e-3)
            code = np.clip(z * self.gain, 0.0, 1.0)
            if code.size:
                cut = np.quantile(code, 1.0 - self.keep)
                code = np.where(code >= cut, code, 0.0)
            out[f"scene_{eye}"] = code.astype(np.float32)
        return out


class MushroomBody:
    """Dopamine-gated depression of the Kenyon-cell synapses onto the MBON."""

    def __init__(self, brain, cfg: dict | None = None):
        c = dict((cfg or {}).get("mushroom_body", {}) or {})
        self.brain = brain
        self.learning_rate = float(c.get("learning_rate", 0.55))
        self.forget_s = float(c.get("forget_s", 90.0))
        self.floor = float(c.get("floor", 0.08))  # how far a synapse may be depressed
        self.dan_ref_hz = float(c.get("dan_ref_hz", 60.0))
        self.kc_ref_hz = float(c.get("kc_ref_hz", 25.0))
        self.kc_groups = list(c.get("kc_groups", ["KC_L", "KC_R"]))
        self.mbon_groups = list(c.get("mbon_groups", ["MBON_L", "MBON_R"]))
        self.dan_groups = list(c.get("dan_groups", ["PPL1_L", "PPL1_R"]))
        self.kc_idx = np.concatenate([brain.connectome.group(g) for g in self.kc_groups] or [np.zeros(0, np.int64)])
        mbon = np.concatenate([brain.connectome.group(g) for g in self.mbon_groups] or [np.zeros(0, np.int64)])
        self.pos, self.owner = brain.net.synapses_between(self.kc_idx, mbon)
        self.w0 = brain.net.get_synapses(self.pos)
        self.w = self.w0.copy()
        self.enabled = bool(c.get("enabled", True)) and self.pos.size > 0

    # ------------------------------------------------------------------
    @property
    def strength(self) -> float:
        """0 while the synapses are untouched, 1 when they are fully depressed."""
        if not self.pos.size:
            return 0.0
        return float(1.0 - self.w.sum() / (self.w0.sum() + 1e-9))

    def reset(self) -> None:
        self.w = self.w0.copy()
        self.brain.net.set_synapses(self.pos, self.w)

    def activity(self, counts: np.ndarray, ms: float) -> np.ndarray:
        """Per-Kenyon-cell rate this tick, normalised to 0..1."""
        if not self.kc_idx.size:
            return np.zeros(0, np.float32)
        hz = counts[self.kc_idx] * 1000.0 / max(1e-6, ms)
        return np.clip(hz / max(1e-6, self.kc_ref_hz), 0.0, 1.0).astype(np.float32)

    def update(self, counts: np.ndarray, ms: float, dopamine_hz: float) -> float:
        """One tick of learning and of forgetting. Returns the memory strength."""
        if not self.enabled:
            return 0.0
        dop = min(1.0, max(0.0, float(dopamine_hz) / max(1e-6, self.dan_ref_hz)))
        changed = False
        if dop > 0.02:
            act = np.zeros(self.brain.connectome.n, dtype=np.float32)
            act[self.kc_idx] = self.activity(counts, ms)
            self.w *= 1.0 - self.learning_rate * dop * act[self.owner]
            np.maximum(self.w, self.w0 * self.floor, out=self.w)
            changed = True
        if self.forget_s > 0:
            gap = self.w0 - self.w
            if gap.any():
                self.w += gap * (1.0 - math.exp(-max(0.0, ms / 1000.0) / self.forget_s))
                changed = True
        if changed:
            self.brain.net.set_synapses(self.pos, self.w)
        return self.strength


class Compass:
    """Reads the heading bump out of the EPG columns."""

    def __init__(self, brain, group: str = "EPG", columns: int = 16, smoothing: float = 0.4):
        self.brain = brain
        self.group = group
        self.columns = int(columns)
        self.idx = brain.connectome.group(group)
        self.per_column = max(1, self.idx.size // self.columns) if self.idx.size else 0
        # A control tick is fifty milliseconds and the ring fires in the tens of
        # hertz, so some ticks hold no EPG spike at all. The read-out is smoothed
        # over ticks rather than reporting that the fly lost its heading.
        self.smoothing = float(smoothing)
        self.level = np.zeros(self.columns, dtype=np.float32)

    @property
    def available(self) -> bool:
        return self.idx.size >= self.columns

    def column_rates(self, counts: np.ndarray, ms: float) -> np.ndarray:
        if not self.available:
            return np.zeros(self.columns, np.float32)
        n = self.columns * self.per_column
        return (counts[self.idx[:n]].reshape(self.columns, self.per_column).sum(1) * 1000.0 / max(1e-6, ms)).astype(np.float32)

    def reset(self) -> None:
        self.level = np.zeros(self.columns, dtype=np.float32)

    def read(self, counts: np.ndarray, ms: float) -> tuple[float, float]:
        """(heading in degrees, how concentrated the bump is)."""
        a = min(1.0, max(0.0, self.smoothing))
        self.level = (1 - a) * self.level + a * self.column_rates(counts, ms)
        c = self.level
        total = float(c.sum())
        if total <= 0:
            return float("nan"), 0.0
        ang = np.arange(self.columns) * 2 * np.pi / self.columns
        v = complex((c * np.cos(ang)).sum(), (c * np.sin(ang)).sum()) / total
        return float(np.degrees(np.angle(v)) % 360.0), float(abs(v))


class Cognition:
    """The memory, the compass, and the inputs the two of them need."""

    def __init__(self, brain, cfg: dict):
        c = dict(cfg.get("cognition", {}) or {})
        self.cfg = c
        self.brain = brain
        self.columns = int(c.get("columns", 16))
        self.scene = SceneCode(float(c.get("scene_keep", 0.45)), float(c.get("scene_gain", 1.6)))
        self.mb = MushroomBody(brain, c)
        self.compass = Compass(brain, c.get("compass_group", "EPG"), self.columns, float(c.get("compass_smoothing", 0.4)))
        self.goal_width = float(c.get("goal_width", 1.2))
        self.seed_s = float(c.get("compass_seed_s", 0.6))
        self.seed_column = float(c.get("compass_seed_column", 0))
        self.punish_decay_s = float(c.get("punish_decay_s", 0.35))
        self.goal_deg: float | None = None
        self.state = CognitiveState()
        self._seed_left = self.seed_s
        self._punish = 0.0

    # ------------------------------------------------------------------
    def reset(self, forget: bool = False) -> None:
        """A crash, a teleport or a new flight: plant the bump again."""
        self._seed_left = self.seed_s
        self._punish = 0.0
        self.compass.reset()
        if forget:
            self.mb.reset()

    def punish(self, amount: float = 1.0) -> None:
        """Something hurt. The dopaminergic neuron carries it to the memory."""
        self._punish = min(1.0, max(self._punish, float(amount)))

    def set_goal(self, degrees: float | None) -> None:
        """Steer to this heading, in the compass's own frame; None to stop."""
        self.goal_deg = None if degrees is None else float(degrees) % 360.0

    def hold_current_heading(self) -> float | None:
        """Take the heading the fly has right now as the goal."""
        if not math.isnan(self.state.heading_deg):
            self.set_goal(self.state.heading_deg)
        return self.goal_deg

    # ------------------------------------------------------------------
    def extra(self, vision, dt: float) -> dict[str, np.ndarray | float]:
        """The non-grid inputs for this tick: scene code, dopamine, goal, seed."""
        out: dict[str, np.ndarray | float] = dict(self.scene(vision))
        out["punish"] = self._punish
        if self.goal_deg is None:
            out["goal"] = 0.0
        else:
            out["goal"] = bump_vector(self.columns, self.goal_deg / 360.0 * self.columns, self.goal_width)
        if self._seed_left > 0:
            out["compass_set"] = bump_vector(self.columns, self.seed_column, 1.0)
            self._seed_left -= max(0.0, dt)
        else:
            out["compass_set"] = 0.0
        return out

    def update(self, counts: np.ndarray, ms: float, rates: dict[str, float]) -> CognitiveState:
        """After the brain has run: learn, read the compass, report."""
        dop = max(rates.get(g, 0.0) for g in self.mb.dan_groups) if self.mb.dan_groups else 0.0
        memory = self.mb.update(counts, ms, dop)
        heading, strength = self.compass.read(counts, ms)
        kc_hz = self.mb.activity(counts, ms)
        error = float("nan")
        if self.goal_deg is not None and not math.isnan(heading):
            error = _wrap180(self.goal_deg - heading)
        self.state = CognitiveState(
            heading_deg=heading,
            heading_strength=strength,
            goal_deg=self.goal_deg,
            heading_error_deg=error,
            memory=memory,
            dopamine_hz=float(dop),
            mbon_hz=float(np.mean([rates.get(g, 0.0) for g in self.mb.mbon_groups])) if self.mb.mbon_groups else 0.0,
            turn_hz=float(np.mean([rates.get(g, 0.0) for g in ("TURN_L", "TURN_R")])),
            kc_active=float((kc_hz > 0.1).mean()) if kc_hz.size else 0.0,
            punished=self._punish > 0.02,
        )
        # dopamine is a pulse, not a state: it fades within a few control ticks
        self._punish *= math.exp(-max(0.0, ms / 1000.0) / max(1e-6, self.punish_decay_s))
        if self._punish < 0.02:
            self._punish = 0.0
        return self.state
