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


def improvisation(seconds: float = 120.0, seed: int = 0, settle: float = 2.5) -> list[tuple[float, GestureState]]:
    """A timeline of gestures for ``seconds`` of flight."""
    rng = random.Random(seed)
    names = list(PHRASES)
    weights = [PHRASES[n][1] for n in names]
    timeline: list[tuple[float, GestureState]] = [(0.0, GestureState())]
    t, last = float(settle), "rest"
    while t < seconds:
        name = rng.choices(names, weights)[0]
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
