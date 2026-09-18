"""The arrangement: a form, a harmony and an orchestration over a flight.

The compositor decides *when* a note happens and roughly how high — that is the
fly, and it is not negotiable. This module decides everything a composer would
decide afterwards: what key the piece is in at that moment, which chord is
underneath it, who is playing and who is sitting out, and how loud.

Nothing here invents a note. Every note in the finished piece is still a spike;
the arranger only moves pitches onto the chord that is sounding, thins a texture
that would otherwise be thirteen voices from beginning to end, and shapes the
dynamics. That distinction matters and is the whole reason the module exists as
a separate pass over the frames rather than as more settings in the compositor:
you can render the same flight with and without it and hear what each one did.

    frames = [comp.tick(info) for ...]
    frames = Arranger(FORMS["daylight"]).apply(frames)

Three ideas do the work.

**Roles, not voices.** Thirteen neuron groups are too many things to write an
arrangement for, and they are named after neurons. They are mapped onto seven
musical roles — bass, lead, pad, colour, spark, percussion, event — and the
score is written for the roles.

**Harmony in cycles, not seconds.** Chord changes land on the beat, and the beat
here is elastic (the tempo rides the wing-stroke neurons). So the form is laid
out in cycles and scaled to however many cycles the flight turned out to be.

**Snapping, by role.** A bass note moves to the nearest chord tone and takes the
root on a chord change; a pad and the bells take chord tones only; the lead gets
the chord *and* the mode, minus any note a semitone above a chord tone — the
usual avoid-note rule, which is also what keeps a borrowed chord from sounding
like a mistake. Percussion is left alone, because its pitch is a click.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from .events import Frame, NoteEvent, parse_root

# ------------------------------------------------------------------- harmony


@dataclass(frozen=True)
class Chord:
    """A chord as a root above the key and a set of intervals above that root."""

    name: str
    root: int
    tones: tuple[int, ...]

    def pitch_classes(self, key: int) -> frozenset[int]:
        return frozenset((key + self.root + t) % 12 for t in self.tones)

    def root_classes(self, key: int) -> frozenset[int]:
        """Root and fifth: what a bass line and a low hit want to land on."""
        fifth = 7 if 7 in self.tones else (self.tones[1] if len(self.tones) > 1 else 0)
        return frozenset(((key + self.root) % 12, (key + self.root + fifth) % 12))


# D lydian and two borrowings. The raised fourth is the brightness; the two
# chords with a natural fourth in them (Cmaj7, Am7) are the only shade in the
# piece, and they are spent in one section.
LYDIAN_PALETTE: dict[str, Chord] = {
    "I": Chord("Imaj9#11", 0, (0, 4, 7, 11, 14, 18)),
    "II": Chord("II9", 2, (0, 4, 7, 10, 14)),
    "iii": Chord("iiim7", 4, (0, 3, 7, 10)),
    "V": Chord("Vmaj7", 7, (0, 4, 7, 11)),
    "vi": Chord("vim9", 9, (0, 3, 7, 10, 14)),
    "vii": Chord("viim7", 11, (0, 3, 7, 10)),
    "bVII": Chord("bVIImaj7", 10, (0, 4, 7, 11)),      # borrowed: the light goes out
    "v": Chord("vm7", 7, (0, 3, 7, 10)),               # borrowed: and stays out a bar
}

MODES: dict[str, tuple[int, ...]] = {
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
}

# ------------------------------------------------------------------- the cast

ROLES: dict[str, str] = {
    "lift": "bass",
    "wing_left": "lead",
    "wing_right": "lead",
    "rising": "pad",
    "falling": "pad",
    "memory": "colour",
    "avoid": "colour",
    "kenyon": "spark",
    "saccade_left": "perc",
    "saccade_right": "perc",
    "loom": "perc",
    "giant_fiber": "event",
    "dopamine": "event",
}

# Roles no section is allowed to silence, and the gain they keep when a section
# does not name them. The giant fibre firing is the loudest thing that happens
# to a fly; an arrangement that mutes it because the form did not expect one
# there is arranging a different flight from the one that was flown.
ALWAYS: dict[str, float] = {"event": 0.95}

# How each role is allowed to move onto the harmony.
SNAP: dict[str, str] = {
    "bass": "root",      # chord tones, and the root itself on a change
    "lead": "mode",      # chord plus the mode, minus the avoid notes
    "pad": "chord",
    "colour": "chord",
    "spark": "chord",
    "perc": "none",      # a click has no pitch worth arguing about
    "event": "root",
}

# ------------------------------------------------------------------ the score


@dataclass(frozen=True)
class Section:
    """One part of the piece: how long, on what harmony, played by whom."""

    name: str
    weight: float                       # relative length, in cycles
    chords: tuple[str, ...]
    roles: dict[str, float]             # role -> gain. A role not named here is silent.
    density: dict[str, float] = field(default_factory=dict)   # role -> share of notes kept
    octave: dict[str, int] = field(default_factory=dict)      # role -> semitones
    note: str = ""


#: how far an anchor may drag a section, as a share of the whole piece. A form
#: that will move anywhere is not a form; this one will bend, not fold.
ANCHOR_REACH = 0.18


def _anchor(plan: list[tuple[float, float, Section]], name: str, candidates: list[float],
            cycles: float) -> list[tuple[float, float, Section]]:
    """Slide the layout so the named section starts on one of ``candidates``.

    The candidate used is the one nearest where the form already put the
    section, and only if it is within :data:`ANCHOR_REACH` of it — otherwise the
    written form wins. An arrangement that rebuilds itself around whichever
    event happened first is not an arrangement.
    """
    hit = next((i for i, (_a, _b, s) in enumerate(plan) if s.name == name), None)
    if hit is None or cycles <= 0 or not candidates or hit in (0, len(plan) - 1):
        return plan
    a, b, s = plan[hit]
    length = b - a
    wants = [min(max(0.0, c - 0.2 * length), max(0.0, cycles - length)) for c in candidates]
    want = min(wants, key=lambda w: abs(w - a))
    if abs(want - a) > ANCHOR_REACH * cycles or abs(want - a) < 1e-9:
        return plan
    # everything before is scaled into [0, want], everything after into the rest
    head = a or 1e-9
    tail = max(1e-9, cycles - b)
    out = []
    for i, (x, y, sec) in enumerate(plan):
        if i < hit:
            out.append((x * want / head, y * want / head, sec))
        elif i == hit:
            out.append((want, want + length, sec))
        else:
            k = (cycles - want - length) / tail
            out.append((want + length + (x - b) * k, want + length + (y - b) * k, sec))
    return out


@dataclass(frozen=True)
class Form:
    """A whole arrangement, independent of how long the flight happened to be."""

    name: str
    key: str
    mode: str
    sections: tuple[Section, ...]
    palette: dict[str, Chord] = field(default_factory=lambda: dict(LYDIAN_PALETTE))

    @property
    def root(self) -> int:
        return parse_root(self.key)

    def plan(self, cycles: float, anchors: dict[str, float] | None = None
             ) -> list[tuple[float, float, Section]]:
        """Lay the form out over ``cycles``, as ``(from, to, section)`` in cycles.

        ``anchors`` maps a section name to the moments, in cycles, that the
        section would like to start on — the form bending to the flight rather
        than the other way round. The anchored section keeps its length;
        everything before and after it is squeezed or stretched to fit around
        it, and it will not move further than :data:`ANCHOR_REACH`.
        """
        total = sum(s.weight for s in self.sections) or 1.0
        out, at = [], 0.0
        for s in self.sections:
            end = at + cycles * s.weight / total
            out.append((at, end, s))
            at = end
        if out:  # the last section owns whatever rounding is left over
            a, _, s = out[-1]
            out[-1] = (a, max(cycles, a), s)
        for name, moments in (anchors or {}).items():
            out = _anchor(out, name, [float(m) for m in moments], cycles)
        return out


# The piece. Seven sections, a chord palette of eight, and one place where the
# harmony is allowed to go dark — which is where the giant fibre tends to fire.
DAYLIGHT = Form(
    name="daylight",
    key="D3",
    mode="lydian",
    sections=(
        Section("dawn", 0.75, ("I",), {"pad": 1.0, "colour": 0.6, "bass": 0.55},
                density={"bass": 0.5}, note="held, almost no pulse: the fly is only hovering"),
        Section("wings", 1.5, ("I", "II"), {"pad": 0.7, "bass": 0.75, "lead": 0.9, "colour": 0.4},
                density={"lead": 0.85}, note="the wings come in over a moving bass"),
        Section("current", 1.9, ("I", "II", "vi", "V"),
                {"pad": 0.55, "bass": 0.95, "lead": 1.0, "colour": 0.5, "spark": 0.7, "perc": 0.75},
                note="the whole band, harmony turning every bar"),
        Section("crowd", 2.1, ("II", "vi", "iii", "V"),
                {"pad": 0.5, "bass": 1.0, "lead": 1.0, "colour": 0.7, "spark": 0.85, "perc": 0.9,
                 "event": 1.0},
                octave={"lead": 12}, note="the busiest of it, the lead an octave up"),
        Section("escape", 1.4, ("bVII", "v", "bVII", "I"),
                {"pad": 0.6, "bass": 1.0, "lead": 0.9, "colour": 0.8, "spark": 0.5, "perc": 0.8,
                 "event": 1.0},
                note="the two borrowed chords: the only shade in the piece"),
        Section("settle", 1.6, ("vi", "V", "I"),
                {"pad": 0.7, "bass": 0.7, "lead": 0.8, "colour": 0.5},
                density={"lead": 0.75}, note="the bells and the drums stop"),
        Section("horizon", 1.1, ("I",), {"pad": 0.8, "lead": 0.5, "colour": 0.35},
                density={"lead": 0.5, "colour": 0.6}, octave={"lead": -12},
                note="one chord, thinning, the lead falling back an octave"),
    ),
)

FORMS: dict[str, Form] = {"daylight": DAYLIGHT}

# ---------------------------------------------------------------- the arranger


def _nearest(note: int, classes: frozenset[int], low: int = 21, high: int = 108) -> int:
    """The pitch closest to ``note`` whose pitch class is allowed. Ties go down."""
    if not classes:
        return note
    for step in range(0, 13):
        for cand in ((note - step), (note + step)):
            if low <= cand <= high and cand % 12 in classes:
                return cand
    return note


class Arranger:
    """Rewrites a flight's frames into an arrangement of it.

    Deterministic: the same frames and the same seed give the same score, which
    is what makes a rendered track reproducible and a test possible.
    """

    #: two notes on the same pitch this close together are a doubling, not a chord
    DOUBLE_S = 0.08

    def __init__(self, form: Form | str = DAYLIGHT, seed: int = 0, roles: dict[str, str] | None = None):
        self.form = FORMS[form] if isinstance(form, str) else form
        self.seed = int(seed)
        self.roles = dict(roles or ROLES)
        self.plan: list[tuple[float, float, Section]] = []

    # -- what is sounding at a given point
    def _at(self, cycle: float) -> tuple[Section, Chord]:
        section = self.plan[-1][2]
        start, end = self.plan[-1][0], self.plan[-1][1]
        for a, b, s in self.plan:
            if a <= cycle < b:
                section, start, end = s, a, b
                break
        span = max(1e-9, end - start)
        i = min(len(section.chords) - 1, int(len(section.chords) * (cycle - start) / span))
        return section, self.form.palette[section.chords[max(0, i)]]

    def _allowed(self, chord: Chord, how: str) -> frozenset[int]:
        key = self.form.root % 12
        tones = chord.pitch_classes(key)
        if how == "chord":
            return tones
        if how == "root":
            return chord.root_classes(key)
        if how == "mode":
            mode = frozenset((key + d) % 12 for d in MODES.get(self.form.mode, MODES["lydian"]))
            avoid = frozenset((t + 1) % 12 for t in tones)   # a semitone above a chord tone
            return tones | (mode - avoid - tones)
        return frozenset(range(12))

    def _bands(self, frames: list[Frame]) -> dict[str, tuple[int, int]]:
        """Each voice's register, taken from the flight and widened by an octave.

        Widened because the arrangement is allowed to move a voice — a section
        can put the lead an octave up — but not to move it anywhere.
        """
        seen: dict[str, tuple[int, int]] = {}
        for frame in frames:
            for note in frame.notes:
                lo, hi = seen.get(note.voice, (note.note, note.note))
                seen[note.voice] = (min(lo, note.note), max(hi, note.note))
        # only the shifts that apply to this voice's role widen its band: giving
        # every voice every section's octave move is a band wide enough to be no
        # band at all, which is how a bass line reaches the bottom of a piano.
        shifts: dict[str, list[int]] = {}
        for section in self.form.sections:
            for role, semis in section.octave.items():
                shifts.setdefault(role, []).append(semis)
        out = {}
        for voice, (lo, hi) in seen.items():
            moves = shifts.get(self.roles.get(voice, ""), [])
            head, foot = max([0, *moves]), min([0, *moves])
            out[voice] = (max(12, lo + foot - 6), min(120, hi + head + 6))
        return out

    def _anchors(self, frames: list[Frame]) -> dict[str, list[float]]:
        """Where the flight says a section belongs.

        One rule so far, and the only one that has earned itself: the section on
        the borrowed chords would like to start where the giant fibre fires.
        That is the moment the fly decided something was about to hit it, and it
        is the one moment in a flight a listener hears as an event whatever is
        playing underneath. If one of them is near enough to where the form
        already puts the shade, the two land together.
        """
        moments = [f.cycle for f in frames
                   if any(self.roles.get(n.voice) == "event" for n in f.notes)]
        return {"escape": moments} if moments else {}

    @staticmethod
    def _register(pitch: int, band: tuple[int, int]) -> int:
        """Fold a pitch back into the register this voice was written for.

        Voice leading chases the previous note, and a long enough chain of
        octave steps walks a voice out of its own range and into somebody
        else's — which is how a bass line ends up at A0 and a pad at the top of
        a piano. The band is the range the flight itself used, so the clamp
        needs no configuration and follows a re-registered voice automatically.
        """
        low, high = band
        while pitch > high and pitch - 12 >= low:
            pitch -= 12
        while pitch < low and pitch + 12 <= high:
            pitch += 12
        return pitch

    @staticmethod
    def _lead(pitch: int, previous: int | None, low: int = 21, high: int = 108) -> int:
        """Bring a note back inside an octave of the last one this voice played.

        Snapping a line onto a chord is what puts it *in* the harmony, and it is
        also what makes it leap: consecutive chord tones are three or four
        semitones apart where consecutive scale tones are two, and forcing the
        bass onto a root at every chord change can throw it into another octave
        entirely. Measured on the shipped flight, snapping alone took the lead
        voices from 70% stepwise motion to 49% and the bass to 10%.

        Moving by whole octaves fixes it without touching the harmony at all —
        the pitch class is what the chord cares about, and the register is what
        the melody cares about. This is voice leading written down as four lines.
        """
        if previous is None:
            return pitch
        while pitch - previous > 6 and pitch - 12 >= low:
            pitch -= 12
        while previous - pitch > 6 and pitch + 12 <= high:
            pitch += 12
        return pitch

    def _space(self, pitch: int, t: float, recent: dict[int, float], allowed: frozenset[int],
               band: tuple[int, int] = (21, 108)) -> int:
        """Move a pitch off anything another voice is holding right next to it.

        Two voices on the same note are one voice and a loss of level; two a
        semitone apart in the same octave are a beat, not a chord. Both happen
        constantly here, because several voices draw on the same available
        pitches and the wings are a pair by construction. The fix is the one an
        arranger writes by hand: keep the note, change the octave or take the
        neighbouring chord tone — whichever is nearer, and never outside ``band``.
        This is the last thing that touches a pitch: folding it back into
        register afterwards would undo the spacing it just did.
        """
        busy = {p for p, when in recent.items() if t - when < self.DOUBLE_S}
        if not any(abs(pitch - p) <= 1 for p in busy):
            return pitch
        for cand in sorted((c for c in range(pitch - 14, pitch + 15)
                            if band[0] <= c <= band[1] and c % 12 in allowed),
                           key=lambda c: abs(c - pitch)):
            if not any(abs(cand - p) <= 1 for p in busy):
                return cand
        return pitch

    def apply(self, frames: list[Frame]) -> list[Frame]:
        """A flight in, the same flight arranged out. The input is not modified."""
        if not frames:
            return []
        cycles = max((f.cycle for f in frames), default=0.0)
        self.plan = self.form.plan(max(cycles, 1e-6), self._anchors(frames))
        rng = random.Random(self.seed)
        chord_now: Chord | None = None
        bass_due = True          # the next bass note after a chord change takes the root
        recent: dict[int, float] = {}   # pitch -> the last time anything played it
        sang: dict[str, int] = {}       # voice -> the last pitch it sang, for the voice leading
        bands = self._bands(frames)     # voice -> the register the flight wrote it in
        out: list[Frame] = []

        for frame in frames:
            section, chord = self._at(frame.cycle)
            if chord is not chord_now:
                chord_now, bass_due = chord, True
            notes: list[NoteEvent] = []
            for note in frame.notes:
                role = self.roles.get(note.voice, "colour")
                gain = section.roles.get(role, ALWAYS.get(role))
                if gain is None:
                    continue                                  # this role is not in this section
                if rng.random() > section.density.get(role, 1.0):
                    continue                                  # thinned out
                how = SNAP.get(role, "chord")
                pitch = int(note.note) + int(section.octave.get(role, 0))
                if how != "none":
                    if role == "bass" and bass_due:
                        pitch = _nearest(pitch, frozenset({(self.form.root + chord.root) % 12}))
                        bass_due = False
                    else:
                        pitch = _nearest(pitch, self._allowed(chord, how))
                    band = bands[note.voice]
                    pitch = self._register(self._lead(pitch, sang.get(note.voice)), band)
                    pitch = self._space(pitch, note.t, recent, self._allowed(chord, how), band)
                    sang[note.voice] = pitch
                    recent[pitch] = note.t
                notes.append(replace(note, note=pitch,
                                     velocity=round(min(1.0, max(0.0, note.velocity * gain)), 4)))
            new = Frame(t=frame.t, cycle=frame.cycle, cps=frame.cps, section=section.name)
            new.notes = notes
            new.controls = list(frame.controls)
            out.append(new)
        return out

    # -- the score, for the terminal
    def describe(self, cycles: float | None = None, seconds: float | None = None) -> str:
        plan = self.plan or self.form.plan(cycles or 1.0)
        total = plan[-1][1] if plan else 1.0
        lines = [f"arrangement '{self.form.name}': {self.form.key} {self.form.mode}, "
                 f"{len(plan)} sections over {total:.0f} cycles"]
        for a, _b, s in plan:
            when = f"{a * (seconds or total) / max(total, 1e-9):5.0f}s" if seconds else f"{a:5.0f}c"
            cast = " ".join(sorted(s.roles))
            lines.append(f"  {when}  {s.name:9s} {'-'.join(s.chords):22s} {cast}")
        return "\n".join(lines)
