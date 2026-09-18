"""MiniFly: a small, hand-wired stand-in connectome for demos and tests.

MiniFly is NOT real fly data. It is a ~1,000-neuron circuit whose cell-type
names and pathways follow published fly flight circuitry, so the whole
FlyDrones pipeline (camera -> neurons -> drone) runs in seconds without the
1.1 GB MaleCNS download:

* T4/T5 motion cells  -> HS / VS lobula-plate cells  -> DNg02 (wing-stroke
  amplitude; Namiki et al. 2022: left/right DNg02 act independently, rightward
  motion raises right DNg02 and lowers left)
* LPLC2 + LC4 looming detectors -> giant fiber DNp01 (escape takeoff;
  Ache et al. 2019) and DNp03 (evasive flight saccades; Current Biology 2025)
* haltere afferents (body rotation) -> DNg02 (yaw damping)

and, since the fly is meant to be cognitive rather than only reflexive, two
learning and navigation circuits:

* visual projection neurons -> Kenyon cells (sparse expansion, APL feedback
  inhibition) -> MBON-g1pedc, with PPL1 dopamine depressing the Kenyon-cell
  synapses that were active when something hurt (Hige et al. 2015, Aso &
  Rubin 2016, Vogt et al. 2016 for the visual pathway). The MBON is GABAergic
  and holds an avoidance turn *off*; learning releases it for that scene.
* a central-complex ring: EPG columns with Delta7 global inhibition, shifted
  by PEN cells driven by the halteres, so the bump integrates turns into a
  heading. PFL3 compares the bump against a goal column held by FC2 and steers
  (Seelig & Jayaraman 2015, Green et al. 2017, Hulse et al. 2021,
  Westeinde et al. 2024).

Weights are synapse counts with signs, exactly like the real connectome, so
``flydrones`` treats MiniFly and MaleCNS identically.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .connectome import Connectome

GRID_ROWS, GRID_COLS = 6, 8  # ommatidia-like grid per eye
COLUMNS = 16  # columns around the central-complex ring: 22.5 degrees each

# Synapse counts for the two cognitive circuits, named so they can be measured
# rather than argued about: tools/bench_cognition.py sweeps them.
MB = {"vpn_kc": 30, "kc_apl": 120, "apl_kc": 25, "kc_mbon": 100, "ppl1_mbon": 2,
      "vpn_turn": 30, "mbon_turn": 220, "turn_dnp03": 60}
CX = {"epg_self": 35, "epg_near": 8, "epg_d7": 20, "d7_epg": 36, "epg_pen": 14,
      "hal_pen": 4, "pen_epg": 48, "fc2_pfl3": 22, "epg_pfl3": 7, "pfl3_dn": 28, "pfl3_lal": 34}


def _pop(spec: list, name: str, n: int, side: str, sign: float) -> None:
    spec.append((name, n, side, sign))


def build_minifly(seed: int = 7) -> Connectome:
    rng = np.random.default_rng(seed)
    cells = GRID_ROWS * GRID_COLS
    pops: list = []
    for s in ("L", "R"):
        _pop(pops, "R1-R6", 2 * cells, s, -1.0)  # histaminergic
        for sub in ("T4a", "T4b", "T4c", "T4d"):
            _pop(pops, sub, cells, s, +1.0)
        _pop(pops, "LPLC2", 24, s, +1.0)
        _pop(pops, "LC4", 12, s, +1.0)
        _pop(pops, "haltere", 16, s, +1.0)
        _pop(pops, "LPi_h", 10, s, -1.0)  # glutamatergic lobula plate intrinsic (horizontal)
        _pop(pops, "LPi_v", 10, s, -1.0)  # glutamatergic lobula plate intrinsic (vertical)
        _pop(pops, "HS", 3, s, +1.0)
        _pop(pops, "VS", 6, s, +1.0)
        _pop(pops, "PVLP", 20, s, +1.0)  # looming integrators
        _pop(pops, "LAL_inh", 12, s, -1.0)  # steering inhibition
        _pop(pops, "PVLP_inh", 6, s, -1.0)  # left/right competition for saccade direction
        _pop(pops, "DNg02", 15, s, +1.0)
        _pop(pops, "DNp03", 2, s, +1.0)
        _pop(pops, "DNp01", 1, s, +1.0)

    # --- cognition. Appended after the reflex populations on purpose: neuron
    # indices of everything above stay exactly where they were.
    for s_ in ("L", "R"):
        _pop(pops, "VPN-MB", 24, s_, +1.0)    # visual projection neurons into the calyx
        _pop(pops, "KCg-d", 180, s_, +1.0)    # visual Kenyon cells (gamma-d)
        _pop(pops, "APL", 1, s_, -1.0)        # the giant feedback inhibitor that keeps KCs sparse
        _pop(pops, "MBON-g1", 4, s_, -1.0)    # MBON-g1pedc>a/b, GABAergic
        _pop(pops, "PPL1-g1", 2, s_, +1.0)    # punishment dopamine
        _pop(pops, "LAL-turn", 8, s_, +1.0)   # premotor turn, held off by the MBON
        _pop(pops, "PEN", 2 * COLUMNS, s_, +1.0)  # shifts the heading bump
        _pop(pops, "PFL3", 2 * COLUMNS, s_, +1.0)  # heading vs goal -> steering
    _pop(pops, "EPG", 8 * COLUMNS, "C", +1.0)  # the compass ring itself
    _pop(pops, "Delta7", 8, "C", -1.0)         # global inhibition around the ring
    _pop(pops, "FC2", 2 * COLUMNS, "C", +1.0)      # the goal column

    types, sides, sign = [], [], []
    index: dict[tuple[str, str], np.ndarray] = {}
    start = 0
    for name, n, side, sg in pops:
        index[(name, side)] = np.arange(start, start + n)
        types += [name] * n
        sides += [side] * n
        sign += [sg] * n
        start += n
    N = start
    sign = np.asarray(sign, dtype=np.float32)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    def connect(pre: np.ndarray, post: np.ndarray, syn: float, p: float = 1.0, jitter: float = 0.3) -> None:
        pre, post = np.asarray(pre), np.asarray(post)
        P, Q = np.meshgrid(pre, post)
        mask = rng.random(P.shape) < p
        w = syn * (1 + jitter * rng.standard_normal(P.shape))
        w = np.clip(np.round(w), 1, None)
        rows.append(Q[mask])
        cols.append(P[mask])
        vals.append(w[mask])

    def g(name: str, side: str) -> np.ndarray:
        return index[(name, side)]

    other = {"L": "R", "R": "L"}
    for s in ("L", "R"):
        o = other[s]
        # --- optic flow -> lobula plate tangential cells
        # HS_s is driven by image motion toward side s (T4a in its own eye is
        # front-to-back; for the right eye that is rightward image motion).
        connect(g("T4a", s), g("HS", s), 6, p=0.8)
        connect(g("T4b", s), g("LPi_h", s), 5, p=0.5)
        connect(g("LPi_h", s), g("HS", s), 6, p=0.8)
        # VS = "scene moves up" = drone is sinking
        connect(g("T4c", s), g("VS", s), 5, p=0.8)
        connect(g("T4d", s), g("LPi_v", s), 5, p=0.6)
        connect(g("LPi_v", s), g("VS", s), 4, p=0.6)

        # --- HS -> DNg02: motion toward side s raises DNg02_s, lowers DNg02_o
        connect(g("HS", s), g("DNg02", s), 11, p=0.9)
        connect(g("HS", s), g("LAL_inh", o), 14, p=0.9)
        connect(g("LAL_inh", o), g("DNg02", o), 12, p=0.9)

        # --- VS -> DNg02 on both sides: sinking -> more lift
        connect(g("VS", s), g("DNg02", s), 10, p=0.9)
        connect(g("VS", s), g("DNg02", o), 6, p=0.6)
        # downward scene motion (rising) -> inhibit lift through LPi
        connect(g("LPi_v", s), g("DNg02", s), 7, p=0.7)

        # --- haltere rotation feedback: rotating toward s damps turning toward s
        connect(g("haltere", s), g("DNg02", o), 4, p=0.7)
        connect(g("haltere", s), g("LAL_inh", s), 5, p=0.7)
        connect(g("LAL_inh", s), g("DNg02", s), 4, p=0.5)

        # --- looming -> escape
        connect(g("LPLC2", s), g("PVLP", s), 6, p=0.7)
        connect(g("LC4", s), g("PVLP", s), 5, p=0.6)
        connect(g("LPLC2", s), g("DNp01", s), 3, p=1.0)
        connect(g("LC4", s), g("DNp01", s), 3, p=1.0)
        connect(g("PVLP", s), g("DNp03", s), 12, p=0.9)
        connect(g("PVLP", s), g("LAL_inh", s), 2, p=0.4)

        # --- photoreceptor brightness: dorsal light gives a weak lift bias
        connect(g("R1-R6", s)[: 2 * GRID_COLS], g("PVLP", s), 1, p=0.05)

        # DNp03 on the threatened side suppresses same-side DNg02 -> turn away
        connect(g("DNp03", s), g("LAL_inh", s), 20, p=1.0)
        # winner-take-all: a head-on threat still produces a turn to one side
        connect(g("DNp03", s), g("PVLP_inh", o), 25, p=1.0)
        connect(g("PVLP_inh", o), g("DNp03", o), 40, p=1.0)
        connect(g("PVLP_inh", o), g("PVLP", o), 3, p=0.5)

    # ------------------------------------------------------------------ cognition
    def column(name: str, side: str, c: int) -> np.ndarray:
        """Neurons of ring column ``c`` (columns are laid out in index order)."""
        idx = index[(name, side)]
        per = len(idx) // COLUMNS
        return idx[c * per : (c + 1) * per]

    def ring(pre: str, pre_side: str, post: str, post_side: str, offset: int, syn: float, p: float = 1.0) -> None:
        """Column c of ``pre`` onto column c + offset of ``post``, around the ring.

        No weight jitter here, unlike everywhere else. A ring attractor is a
        competition between columns, so a lucky column with heavier synapses
        wins every time and the bump sticks to it instead of following the
        halteres — which is exactly what the first version of this did.
        """
        for c in range(COLUMNS):
            connect(column(pre, pre_side, c), column(post, post_side, (c + offset) % COLUMNS), syn, p=p, jitter=0.0)

    for s in ("L", "R"):
        o = other[s]
        # --- mushroom body. A sparse visual code, an output neuron that holds a
        # turn off, and a dopaminergic neuron that teaches which scene hurts.
        connect(g("VPN-MB", s), g("KCg-d", s), MB["vpn_kc"], p=4 / 24, jitter=0.2)  # each KC samples ~4 VPNs
        connect(g("KCg-d", s), g("APL", s), MB["kc_apl"], p=1.0)                    # KCs excite the inhibitor
        connect(g("APL", s), g("KCg-d", s), MB["apl_kc"], p=1.0)                    # which keeps them sparse
        connect(g("KCg-d", s), g("MBON-g1", s), MB["kc_mbon"], p=0.5)                # the plastic synapses
        connect(g("PPL1-g1", s), g("MBON-g1", s), MB["ppl1_mbon"], p=0.5)              # dopamine also drives it a little
        connect(g("VPN-MB", s), g("LAL-turn", s), MB["vpn_turn"], p=0.35)             # the drive the MBON gates
        connect(g("MBON-g1", s), g("LAL-turn", s), MB["mbon_turn"], p=1.0)            # GABAergic: turn held off
        connect(g("LAL-turn", s), g("DNp03", s), MB["turn_dnp03"], p=0.8)               # learned avoidance saccade

        # --- central complex. PEN cells on each side shift the bump when the
        # halteres say the body turned; the ring itself is wired once, below.
        ring("EPG", "C", "PEN", s, 0, CX["epg_pen"])
        connect(g("haltere", s), g("PEN", s), CX["hal_pen"], p=1.0, jitter=0.0)
        ring("PEN", s, "EPG", "C", 1 if s == "R" else -1, CX["pen_epg"])

        # --- goal steering. PFL3 sees the goal column and the bump a quarter
        # turn away; whichever side gets both fires, and steers that way.
        ring("FC2", "C", "PFL3", s, 0, CX["fc2_pfl3"])
        # The quarter-turn offset decides which way the fly turns to close the
        # gap, and the wrong sign is a loop that runs *away* from the goal and
        # settles 180 degrees from it. Measured, not reasoned: tools/bench_cognition.py.
        ring("EPG", "C", "PFL3", s, -COLUMNS // 4 if s == "R" else COLUMNS // 4, CX["epg_pfl3"])
        connect(g("PFL3", s), g("DNg02", s), CX["pfl3_dn"], p=1.0, jitter=0.0)
        connect(g("PFL3", s), g("LAL_inh", o), CX["pfl3_lal"], p=1.0, jitter=0.0)

    # The ring is one structure across the midline, so it is wired once: local
    # excitation between neighbouring columns, and Delta7 inhibition everywhere
    # else, which is what leaves a single bump.
    ring("EPG", "C", "EPG", "C", 0, CX["epg_self"])
    ring("EPG", "C", "EPG", "C", 1, CX["epg_near"])
    ring("EPG", "C", "EPG", "C", -1, CX["epg_near"])
    connect(g("EPG", "C"), g("Delta7", "C"), CX["epg_d7"], p=1.0, jitter=0.0)
    connect(g("Delta7", "C"), g("EPG", "C"), CX["d7_epg"], p=1.0, jitter=0.0)

    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(vals).astype(np.float32) * sign[c]
    W = sparse.csc_matrix((v, (r, c)), shape=(N, N), dtype=np.float32)

    return Connectome(
        name="minifly-synthetic",
        weights=W,
        types=np.asarray(types),
        sides=np.asarray(sides),
        superclass=None,
        body_ids=np.arange(N, dtype=np.int64),
        meta={"synthetic": True, "grid": [GRID_ROWS, GRID_COLS], "seed": seed},
    )
