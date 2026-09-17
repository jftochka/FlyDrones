"""A stand-in for FlyPV's bridge: one JSON request per line in, one out.

Enough of the protocol for the client to be tested against — and deliberately
not a simulator. It echoes the control back inside the telemetry, so a test can
assert which FlyPV axis each FlightCommand axis reached, which is the part of
this backend that can silently be wrong.
"""

from __future__ import annotations

import base64
import json
import sys

PROTOCOL = int(sys.argv[1]) if len(sys.argv) > 1 else 1
STATE = {"time": 0.0, "phase": "grounded", "control": {}, "camera": {"width": 4, "height": 3}}


def telemetry() -> dict:
    return {
        "time": STATE["time"],
        "altitude": 1.25,
        "east": 2.0,
        "north": -3.0,
        "verticalSpeed": 0.5,
        "forwardSpeed": 0.25,
        "lateralSpeed": 0.0,
        "groundSpeed": 0.25,
        "heading": 90.0,
        "yawRate": 12.0,
        "roll": 1.0,
        "pitch": -2.0,
        "batteryPercent": 88.0,
        "batteryVoltage": 16.2,
        "armed": STATE["phase"] in ("takeoff", "flying", "landing"),
        "flying": STATE["phase"] == "flying",
        "crashed": False,
        "crashes": 2,
        "phase": STATE["phase"],
        "armingRefusal": None,
        "nearest": 3.5,
        "control": STATE["control"],
    }


def frame() -> dict:
    w, h = STATE["camera"]["width"], STATE["camera"]["height"]
    pixels = bytes(range(w * h))
    return {"width": w, "height": h, "pixels": base64.urlsafe_b64encode(pixels).decode().rstrip("=")}


PLAN = {
    "name": "The Warehouse",
    "bounds": 60,
    "indoors": True,
    "footprints": [
        {"east": -4.0, "north": 2.0, "halfEast": 1.0, "halfNorth": 3.0, "top": 4.0},
        {"east": 3.0, "north": -5.0, "halfEast": 2.0, "halfNorth": 0.5, "top": 2.5},
        {"east": 200.0, "north": 0.0, "halfEast": 1.0, "halfNorth": 1.0, "top": 1.0},
    ],
}


def handle(request: dict) -> dict:
    op = request.get("op")
    if op == "hello":
        return {"ok": True, "op": "hello", "protocol": PROTOCOL, "config": {}, "world": PLAN,
                "worlds": ["warehouse", "valley"], "airframes": ["cinewhoop3"], "rates": ["cinematic"]}
    if op == "reset":
        config = request.get("config") or {}
        if config.get("world") not in (None, "warehouse", "valley"):
            return {"ok": False, "error": f"this build does not have location \"{config.get('world')}\""}
        if config.get("camera"):
            STATE["camera"] = config["camera"]
        STATE["time"] = 0.0
        STATE["phase"] = "grounded"
        return {"ok": True, "op": "reset", "telemetry": telemetry(), "world": PLAN}
    if op in ("takeoff", "land", "stop"):
        STATE["phase"] = {"takeoff": "flying", "land": "landing", "stop": "grounded"}[op]
        return {"ok": True, "op": op, "telemetry": telemetry()}
    if op == "step":
        dt = request.get("dt", 0)
        if not isinstance(dt, (int, float)) or dt <= 0:
            return {"ok": False, "error": "step needs a positive dt"}
        STATE["time"] += dt
        STATE["control"] = request.get("control", {})
        out = {"ok": True, "op": "step", "telemetry": telemetry(), "steps": int(dt * 1000)}
        if request.get("camera"):
            out["camera"] = frame()
        return out
    if op == "look":
        return {"ok": True, "op": "look", "telemetry": telemetry(), "camera": frame()}
    if op == "recording":
        return {"ok": True, "op": "recording", "recording": {"version": 18, "spans": [], "setup": {"world": "warehouse"}}}
    return {"ok": False, "error": f"unknown op: {op!r}"}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        print(json.dumps(handle(json.loads(line))), flush=True)


if __name__ == "__main__":
    main()
