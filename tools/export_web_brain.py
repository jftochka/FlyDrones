"""Export MiniFly + the default config to JSON for the browser demo (docs/live/minifly.json).

    python tools/export_web_brain.py

The browser gets the whole connectome, cognitive neurons included, but not the
inputs that drive them: engine.js has no scene code, no dopamine and no goal,
so the mushroom body and the compass are exported silent rather than exported
half-driven. The browser demo is the reflex fly; cognition runs in Python.
"""

import copy
import json
from pathlib import Path

import numpy as np

from flydrones.brain import Brain, build_minifly
from flydrones.config import load_config

OUT = Path(__file__).resolve().parents[1] / "docs" / "live" / "minifly.json"


# Features engine.js knows how to compute. Anything else is a cognition input.
WEB_FEATURES = {"brightness", "ftb", "btf", "up", "down", "loom", "loom_speed", "yaw_pos", "yaw_neg"}
WEB_SKIP_BIAS = ("EPG", "PEN_L", "PEN_R")  # no compass in the browser, so no tonic drive for one


def _browser_config(cfg: dict) -> dict:
    web = {k: copy.deepcopy(cfg[k]) for k in ("brain", "inputs", "outputs", "vision", "decoder", "safety", "imu")}
    web["inputs"] = {k: v for k, v in web["inputs"].items() if v.get("feature") in WEB_FEATURES}
    for group in WEB_SKIP_BIAS:
        web["brain"].get("bias", {}).pop(group, None)
    return web


def export(out: Path = OUT) -> dict:
    cfg = load_config()
    brain = Brain(build_minifly(), cfg)
    c = brain.connectome
    W = c.weights.tocsc()
    # population runs: consecutive neurons with the same (type, side)
    pops = []
    start = 0
    for i in range(1, c.n + 1):
        if i == c.n or c.types[i] != c.types[start] or c.sides[i] != c.sides[start]:
            pops.append([str(c.types[start]), str(c.sides[start]), start, i - start])
            start = i
    data = {
        "name": c.name,
        "n": int(c.n),
        "nnz": int(W.nnz),
        "indptr": W.indptr.astype(int).tolist(),
        "indices": W.indices.astype(int).tolist(),
        "data": W.data.astype(int).tolist(),
        "pops": pops,
        "groups": {k: v.astype(int).tolist() for k, v in c.groups.items()},
        "config": _browser_config(cfg),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return data


if __name__ == "__main__":
    d = export()
    print(f"exported {d['n']} neurons, {d['nnz']} connections -> {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    assert np.all(np.diff(d["indptr"]) >= 0)
