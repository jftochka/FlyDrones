"""Writing the flight down: mini-notation for TidalCycles and Strudel.

Streaming notes at a synthesiser is one half of live coding; the other half is
having something to edit. These functions turn a flight into source you can
paste into Tidal or Strudel — the transcription of what the fly played, and the
companion file that binds this configuration's voices to live ``cF`` values or
Strudel signals so you can keep patterning *while* it flies.

Pitch is written as a number in both, never as a note name: Strudel's ``note``
counts MIDI numbers from C-1 and Tidal's counts semitones from middle C, and a
name means different pitches in the two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .events import Frame

# Sounds that exist without installing anything: SuperDirt's bundled synths and
# Strudel's built-in waveforms.
TIDAL_SOUNDS = {"flybass": "supersaw", "flylead": "superpwm", "flyperc": "superhat",
                "flycrash": "superhoover", "flyloom": "supersquare", "flyair": "supersine"}
STRUDEL_SOUNDS = {"flybass": "sawtooth", "flylead": "triangle", "flyperc": "square",
                  "flycrash": "sawtooth", "flyloom": "square", "flyair": "sine"}


@dataclass
class Score:
    """What was played, on a grid: ``voices[name][cycle][step]`` is a MIDI note."""

    cps: float = 0.5
    steps: int = 16
    voices: dict[str, list[list[int | None]]] = field(default_factory=dict)
    sounds: dict[str, str] = field(default_factory=dict)
    pans: dict[str, float] = field(default_factory=dict)
    cycles: int = 0
    notes: int = 0
    dropped: int = 0  # notes that fell on a step already taken

    def voice_names(self) -> list[str]:
        return [v for v in self.voices if any(any(s is not None for s in c) for c in self.voices[v])]


def transcribe(frames: list[Frame], steps: int = 16, max_cycles: int = 64) -> Score:
    """Quantise a flight onto ``steps`` slots per cycle.

    The compositor already placed every note at the millisecond its neurons
    fired; a grid is what makes it editable, and is the one place in the chain
    that throws timing away on purpose.
    """
    frames = [f for f in frames if f is not None]
    if not frames:
        return Score(steps=steps)
    cps = frames[len(frames) // 2].cps or 0.5
    first = math.floor(frames[0].cycle)
    score = Score(cps=cps, steps=steps)
    for f in frames:
        for n in f.notes:
            pos = f.cycle + (n.t - f.t) * (f.cps or cps)
            c = int(math.floor(pos)) - first
            if c < 0 or c >= max_cycles:
                continue
            slot = int(round((pos - math.floor(pos)) * steps)) % steps
            cycles = score.voices.setdefault(n.voice, [])
            while len(cycles) <= c:
                cycles.append([None] * steps)
            row = cycles[c]
            if row[slot] is None:
                row[slot] = int(n.note)
                score.notes += 1
            else:
                score.dropped += 1
            score.sounds[n.voice] = n.sound
            score.pans[n.voice] = n.pan
    score.cycles = max((len(v) for v in score.voices.values()), default=0)
    for cycles in score.voices.values():  # pad, so every voice spans the whole flight
        while len(cycles) < score.cycles:
            cycles.append([None] * steps)
    return score


def _cycle_string(row: list[int | None], offset: int = 0) -> str:
    return " ".join("~" if n is None else str(n + offset) for n in row)


def _voice_pattern(cycles: list[list[int | None]], offset: int = 0) -> str:
    """One cycle per ``<>`` slot: the flight plays back in order, then repeats."""
    bodies = [_cycle_string(row, offset) for row in cycles]
    if len(bodies) == 1:
        return bodies[0]
    return "<" + " ".join(f"[{b}]" for b in bodies) + ">"


def to_strudel(score: Score, sounds: dict[str, str] | None = None, title: str = "a flight") -> str:
    sounds = {**STRUDEL_SOUNDS, **(sounds or {})}
    names = score.voice_names()
    if not names:
        return "// nothing was played\nsilence\n"
    lines = [f"// FlyDrones · {title}: {score.notes} notes from a fly brain, {score.cycles} cycles.",
             "// Pitch is MIDI. Each <...> slot is one cycle of the flight, in order.",
             f"setcps({score.cps:.4f})", "stack("]
    for i, v in enumerate(names):
        sound = sounds.get(score.sounds.get(v, ""), "triangle")
        comma = "," if i < len(names) - 1 else ""
        lines.append(f'  note("{_voice_pattern(score.voices[v])}").s("{sound}").pan({score.pans.get(v, 0.5):.2f})'
                     f"{comma}  // {v}")
    lines.append(")")
    return "\n".join(lines) + "\n"


def to_tidal(score: Score, sounds: dict[str, str] | None = None, title: str = "a flight", orbit_base: int = 0) -> str:
    sounds = {**TIDAL_SOUNDS, **(sounds or {})}
    names = score.voice_names()
    if not names:
        return "-- nothing was played\nd1 $ silence\n"
    lines = [f"-- FlyDrones · {title}: {score.notes} notes from a fly brain, {score.cycles} cycles.",
             "-- Pitch is semitones from middle C (Tidal's note numbering).",
             f"setcps {score.cps:.4f}", "", "d1 $ stack ["]
    for i, v in enumerate(names):
        sound = sounds.get(score.sounds.get(v, ""), "superpwm")
        comma = "," if i < len(names) - 1 else ""
        lines.append(f'  note "{_voice_pattern(score.voices[v], offset=-60)}" # s "{sound}" '
                     f"# pan {score.pans.get(v, 0.5):.2f} # orbit {orbit_base + i}{comma}  -- {v}")
    lines.append("  ]")
    return "\n".join(lines) + "\n"


def tidal_live_file(voices: list[str], controls: list[str], port: int = 6010) -> str:
    """A ``.tidal`` file that reads the live flight instead of replaying it.

    ``flydrones compose --to tidal`` sends ``/ctrl`` messages; every name below
    is one of them, so ``cF`` picks it up inside any pattern you write.
    """
    out = [
        "-- FlyDrones · live control bus. Start `flydrones compose --to tidal` and",
        f"-- evaluate this file. Tidal already listens for /ctrl on {port}.",
        "--",
        "-- Continuous values from the flight (0..1 unless noted):",
        "--   " + ", ".join(controls),
        "-- Per voice: <voice>_n last note (MIDI), <voice>_v velocity, <voice>_g 1 on the tick it fired.",
        "",
        "let flyF name = cF 0 name                 -- any control above",
        '    flyN v    = cN 60 (v ++ "_n") - 60     -- MIDI on the wire, middle C = 0 here',
        '    flyV v    = cF 0.8 (v ++ "_v")',
        '    flyG v    = cF 0 (v ++ "_g")',
        "",
        "-- The wings carry the melody, the eyes open the filter, looming adds noise,",
        "-- and the giant fiber only sounds on the tick it fired.",
        "d1 $ stack [",
        '  note (flyN "wing_left")  # s "superpwm" # pan 0.2 # struct "t*4"',
        '    # gain (0.5 + 0.4 * flyF "drive") # lpf (200 + 4000 * flyF "bright"),',
        '  note (flyN "wing_right") # s "superpwm" # pan 0.8 # struct "~ t ~ t"',
        '    # gain (0.5 + 0.4 * flyF "drive"),',
        '  note (flyN "lift")       # s "supersaw" # struct "t ~ ~ t" # gain 0.8,',
        '  note (flyN "loom")       # s "supersquare" # struct "t*8" # gain (0.7 * flyF "loom"),',
        '  note (flyN "giant_fiber") # s "superhoover" # struct "t*2" # gain (flyG "giant_fiber")',
        "  ]",
        "",
        "-- Voices in this configuration: " + ", ".join(voices),
    ]
    return "\n".join(out) + "\n"


def strudel_live_snippet(voices: list[str], controls: list[str], url: str = "http://127.0.0.1:8765/events") -> str:
    """A self-contained paste for strudel.cc: connect, then pattern the values.

    The served player page imports ``flybrain.mjs`` instead; this is the copy
    for the public REPL, which cannot import from your laptop.
    """
    return f"""// FlyDrones · paste into https://strudel.cc while `flydrones compose --to strudel` runs.
// It opens one EventSource to your own machine and keeps the latest frame in `fly`.
// Controls: {", ".join(controls)}
// Voices:   {", ".join(voices)}
globalThis.fly ??= (() => {{
  const f = {{ c: {{}}, last: {{}}, at: {{}}, section: 'ground', cps: 0.5, frames: 0 }};
  const es = new EventSource({url!r});
  es.onmessage = (m) => {{
    const d = JSON.parse(m.data);
    Object.assign(f.c, d.controls || {{}});
    f.section = d.section; f.cps = d.cps; f.frames++;
    for (const n of d.notes || []) {{ f.last[n.voice] = n; f.at[n.voice] = performance.now(); }}
  }};
  f.ctl = (k, d = 0) => (Number.isFinite(f.c[k]) ? f.c[k] : d);
  f.n = (v, d = 60) => (f.last[v] ? f.last[v].note : d);
  f.gate = (v, s = 0.15) => ((performance.now() - (f.at[v] ?? -1e9)) / 1000 < s ? 1 : 0);
  return f;
}})();

stack(
  note(signal(() => fly.n('wing_left', 60))).s("triangle").pan(.2).struct("t*4"),
  note(signal(() => fly.n('wing_right', 60))).s("triangle").pan(.8).struct("~ t ~ t"),
  note(signal(() => fly.n('lift', 36))).s("sawtooth").struct("t ~ ~ t")
    .lpf(signal(() => 250 + 3500 * fly.ctl('bright'))),
  note(signal(() => fly.n('loom', 84))).s("square").struct("t*8")
    .gain(signal(() => .55 * fly.ctl('loom'))),
  note(signal(() => fly.n('giant_fiber', 28))).s("sawtooth").struct("t*2")
    .gain(signal(() => fly.gate('giant_fiber', .45)))
).gain(signal(() => .35 + .5 * fly.ctl('drive')))
"""
