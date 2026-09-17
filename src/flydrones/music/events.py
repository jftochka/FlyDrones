"""The musical vocabulary: notes, controls and the scale they are quantised to.

Deliberately tiny and free of I/O. Everything downstream — Pure Data, Max/MSP,
TidalCycles, Strudel, a JSONL file — is a rendering of this event stream, so
the mapping from a fly brain to music is decided in exactly one place and can
be tested without a socket or a synthesiser.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Scale degrees in semitones. A rate loop does not care, but a listener does:
# unquantised pitch from a firing rate sounds like a fault, not like a fly.
SCALES: dict[str, tuple[int, ...]] = {
    "chromatic": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
    "whole_tone": (0, 2, 4, 6, 8, 10),
}

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[int(midi) % 12]}{int(midi) // 12 - 1}"


def parse_root(root: str | int) -> int:
    """``"C3"`` / ``"f#2"`` / ``"Bb4"`` / ``48`` -> MIDI number."""
    if isinstance(root, (int, float)):
        return int(root)
    s = str(root).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    semis = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    letter, rest = s[:1].upper(), s[1:]
    accidental = 0
    if rest[:1] in ("#", "b"):
        accidental = 1 if rest[0] == "#" else -1
        rest = rest[1:]
    if letter not in semis or not rest.lstrip("-").isdigit():
        raise ValueError(f"not a note name: {root!r} (try C3, F#2, Bb4 or a MIDI number)")
    return semis[letter] + accidental + (int(rest) + 1) * 12


@dataclass(frozen=True)
class Scale:
    """A set of allowed pitches. ``quantise`` is the only thing anyone calls."""

    name: str = "minor_pentatonic"
    root: int = 48  # C3

    @staticmethod
    def parse(name: str = "minor_pentatonic", root: str | int = "C3") -> Scale:
        key = str(name).lower().replace("-", "_").replace(" ", "_")
        if key not in SCALES:
            raise ValueError(f"unknown scale {name!r}; try one of {', '.join(sorted(SCALES))}")
        return Scale(key, parse_root(root))

    @property
    def degrees(self) -> tuple[int, ...]:
        return SCALES[self.name]

    def notes_between(self, low: int, high: int) -> list[int]:
        """Every pitch of the scale in ``[low, high]``, lowest first."""
        if high < low:
            low, high = high, low
        out = [n for n in range(low, high + 1) if (n - self.root) % 12 in self.degrees]
        return out or [low]

    def quantise(self, x: float, low: int, high: int) -> int:
        """Map ``x`` in 0..1 onto the scale pitches inside ``[low, high]``."""
        ladder = self.notes_between(low, high)
        x = 0.0 if x != x else min(1.0, max(0.0, float(x)))  # NaN -> bottom of the ladder
        return ladder[int(round(x * (len(ladder) - 1)))]


@dataclass(frozen=True)
class NoteEvent:
    """One sound. ``t`` is flight time in seconds, not wall clock."""

    t: float
    voice: str
    sound: str
    note: int
    velocity: float = 0.8
    duration: float = 0.25
    pan: float = 0.5
    orbit: int = 0
    channel: int = 1
    rate_hz: float = 0.0
    kind: str = "note"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ControlEvent:
    """A continuous value: cutoff, altitude, a firing rate, the flight command."""

    t: float
    name: str
    value: float
    kind: str = "control"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Frame:
    """Everything the compositor produced for one control tick."""

    t: float
    notes: list[NoteEvent] = field(default_factory=list)
    controls: list[ControlEvent] = field(default_factory=list)
    cycle: float = 0.0
    cps: float = 0.5
    section: str = ""

    def __iter__(self):
        return iter([*self.notes, *self.controls])

    def __len__(self) -> int:
        return len(self.notes) + len(self.controls)


def event_from_dict(d: dict) -> NoteEvent | ControlEvent:
    d = dict(d)
    kind = d.pop("kind", "note")
    if kind == "control":
        return ControlEvent(**d)
    return NoteEvent(**d)
