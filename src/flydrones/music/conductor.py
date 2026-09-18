"""The hand in front of the camera, written as a score.

``compose`` can run for minutes, and the twenty-second demo timeline loops into
the same phrase over and over. The conductor writes a longer one instead: a
seeded sequence of the gestures the fly reacts to, with rests between them, so
a session has some shape and two runs of the same seed are the same piece.

It is still only an illusion generator — nothing here tells the drone to climb.
"""

from __future__ import annotations

import random

from ..senses.gestures import GestureState

# name -> (gesture, weight, shortest, longest). Sizes and openness are the ones
# the demo timeline uses, because those are the values the retina was tuned on.
PHRASES: dict[str, tuple[GestureState, float, float, float]] = {
    "rest": (GestureState(), 1.2, 1.5, 4.0),
    "hold": (GestureState(True, 0.10, 0.0, 0.0, 0.15, "fist"), 2.0, 2.0, 6.0),
    "climb": (GestureState(True, 0.95, 0.0, 0.0, 0.20, "open palm"), 1.6, 1.5, 4.0),
    "turn_left": (GestureState(True, 0.10, -0.85, 0.0, 0.15, "fist, moved left"), 1.0, 1.5, 3.5),
    "turn_right": (GestureState(True, 0.10, 0.85, 0.0, 0.15, "fist, moved right"), 1.0, 1.5, 3.5),
    "descend": (GestureState(False, 0.0, 0.0, 1.0, 0.0, "hand dropped"), 1.0, 1.5, 3.5),
    "loom": (GestureState(True, 0.10, 0.0, 0.0, 0.55, "hand rushes at camera"), 0.5, 0.6, 1.2),
}


# How a piece is shaped rather than merely filled: quiet at the edges, busy in
# the middle, and the thing that rushes the drone saved for the last third of
# the build. The fly still plays it; this only decides what it is shown, and when.
ARC = {
    "rest": (2.6, 0.5, 2.4),      # (weight at the start, in the middle, at the end)
    "hold": (2.4, 1.0, 2.6),
    "climb": (0.8, 2.2, 0.9),
    "turn_left": (0.4, 1.6, 0.5),
    "turn_right": (0.4, 1.6, 0.5),
    "descend": (0.6, 1.2, 1.0),
    "loom": (0.0, 1.4, 0.1),
}


def _shaped(weights: dict[str, float], shape: str, position: float) -> list[float]:
    """The phrase weights at one point through the piece, 0 at the start, 1 at the end."""
    if shape != "arc":
        return [weights[n] for n in weights]
    out = []
    for name in weights:
        a, b, c = ARC.get(name, (1.0, 1.0, 1.0))
        k = position * 2.0
        curve = a + (b - a) * k if k <= 1 else b + (c - b) * (k - 1)
        out.append(max(0.0, weights[name] * curve))
    return out


def improvisation(seconds: float = 120.0, seed: int = 0, settle: float = 2.5,
                  shape: str = "flat") -> list[tuple[float, GestureState]]:
    """A timeline of gestures for ``seconds`` of flight.

    ``shape="arc"`` gives the session a beginning, a middle and an end: it
    starts with the fly mostly left alone, works up to climbs, turns and
    something rushing at it, and settles again.
    """
    rng = random.Random(seed)
    names = list(PHRASES)
    weights = [PHRASES[n][1] for n in names]
    timeline: list[tuple[float, GestureState]] = [(0.0, GestureState())]
    t, last = float(settle), "rest"
    base = dict(zip(names, weights))
    while t < seconds:
        name = rng.choices(names, _shaped(base, shape, min(1.0, t / max(1e-6, seconds))))[0]
        if name == last:
            continue  # no gesture twice in a row: the brain would not notice the second
        gesture, _w, lo, hi = PHRASES[name]
        timeline.append((round(t, 3), gesture))
        t += rng.uniform(lo, hi)
        if name == "loom":  # a scare, then let the escape play out before the next one
            timeline.append((round(t, 3), PHRASES["hold"][0]))
            t += rng.uniform(1.5, 3.0)
            name = "hold"
        last = name
    return timeline


def describe(timeline: list[tuple[float, GestureState]]) -> str:
    return " · ".join(f"{t:.0f}s {g.label or 'no hand'}" for t, g in timeline[:8]) + (" …" if len(timeline) > 8 else "")
