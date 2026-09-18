"""The arrangement: does it actually put the flight on the harmony, and does it
leave the flight recognisable afterwards?

The questions worth asking of an arranger are all comparisons — the same frames
with and without it — so most of these tests build a score, run it through, and
measure the difference rather than asserting a pitch.
"""

from __future__ import annotations

import math

import pytest

from flydrones.music.arrange import (
    ALWAYS,
    ANCHOR_REACH,
    DAYLIGHT,
    FORMS,
    LYDIAN_PALETTE,
    ROLES,
    SNAP,
    Arranger,
    Chord,
    Form,
    Section,
)
from flydrones.music.events import Frame, NoteEvent, parse_root

VOICES = ("lift", "wing_left", "wing_right", "rising", "kenyon", "memory", "saccade_left")


#: base pitch and how fast each voice walks, so the lines cross each other the
#: way a real flight's do instead of moving in parallel forever.
WALK = {"lift": (45, 1), "wing_left": (64, 2), "wing_right": (66, 3), "rising": (78, 1),
        "kenyon": (88, 5), "memory": (52, 4), "saccade_left": (70, 7)}


def score(seconds: float = 120.0, cps: float = 0.5, every: float = 0.25) -> list[Frame]:
    """A flight-shaped score: every voice, a walking pitch, a steady cycle."""
    frames, t, cycle, i = [], 0.0, 0.0, 0
    while t < seconds:
        f = Frame(t=round(t, 4), cycle=cycle, cps=cps, section="hover")
        for v in VOICES:
            base, step = WALK[v]
            f.notes.append(NoteEvent(t=f.t, voice=v, sound="flylead",
                                     note=base + (i * step) % 11, velocity=0.8, duration=0.3))
        frames.append(f)
        t += every
        cycle += every * cps
        i += 1
    return frames


def test_every_pitched_note_lands_on_the_chord_that_is_sounding():
    """The whole point: after arranging, nothing is off the harmony."""
    arr = Arranger(DAYLIGHT, seed=3)
    out = arr.apply(score())
    checked = 0
    for frame in out:
        _section, chord = arr._at(frame.cycle)
        for note in frame.notes:
            how = SNAP[ROLES[note.voice]]
            if how == "none":
                continue
            checked += 1
            assert note.note % 12 in arr._allowed(chord, how), f"{note.voice} {note.note} off {chord.name}"
    assert checked > 500


def test_it_removes_the_close_intervals_it_is_there_to_remove():
    """A minor second between two voices in the same octave is a beat, not a
    chord, and two voices on the same note are one voice at half the level."""
    before = score()
    after = Arranger(DAYLIGHT, seed=1).apply(before)

    def close(frames):
        n = 0
        for f in frames:
            pitches = [x.note for x in f.notes if ROLES[x.voice] != "perc"]
            for i, a in enumerate(pitches):
                for b in pitches[i + 1:]:
                    if abs(a - b) <= 1:
                        n += 1
        return n

    assert close(before) > 50
    assert close(after) == 0


def test_a_bass_note_takes_the_root_when_the_chord_changes():
    arr = Arranger(DAYLIGHT, seed=0)
    out = arr.apply(score())
    key = DAYLIGHT.root % 12
    seen, chord_before = 0, None
    for frame in out:
        _s, chord = arr._at(frame.cycle)
        changed = chord is not chord_before
        chord_before = chord
        if not changed:
            continue
        for note in frame.notes:
            if ROLES[note.voice] == "bass":
                assert note.note % 12 == (key + chord.root) % 12
                seen += 1
                break
    assert seen >= 3, "the test score never crossed a chord change with a bass note on it"


def test_the_orchestration_changes_from_section_to_section():
    """A section that names nobody new is not an arrangement, it is a setting."""
    out = Arranger(DAYLIGHT, seed=2).apply(score())
    cast: dict[str, set[str]] = {}
    for frame in out:
        cast.setdefault(frame.section, set()).update(ROLES[n.voice] for n in frame.notes)
    assert len(cast) == len(DAYLIGHT.sections)
    assert len({frozenset(v) for v in cast.values()}) >= 4
    assert cast["dawn"] < cast["current"]          # the opening is a subset of the middle


def test_the_dynamics_rise_and_fall_across_the_piece():
    out = Arranger(DAYLIGHT, seed=2).apply(score())
    level: dict[str, list[float]] = {}
    for frame in out:
        level.setdefault(frame.section, []).extend(n.velocity for n in frame.notes)
    mean = {k: sum(v) / len(v) for k, v in level.items() if v}
    assert mean["crowd"] > mean["dawn"] and mean["crowd"] > mean["horizon"]


def test_the_giant_fibre_is_never_silenced():
    """No section may mute the fly's own drama, whatever the form says."""
    assert "event" in ALWAYS
    frames = score(seconds=40)
    for i, f in enumerate(frames):
        f.notes.append(NoteEvent(t=f.t, voice="giant_fiber", sound="flycrash", note=36 + i % 5))
    out = Arranger(DAYLIGHT, seed=0).apply(frames)
    kept = sum(1 for f in out for n in f.notes if n.voice == "giant_fiber")
    assert kept == len(frames)


def test_the_shade_moves_to_an_escape_that_is_near_enough_and_not_to_one_that_is_not():
    frames = score(seconds=120)
    cycles = max(f.cycle for f in frames)
    plan = DAYLIGHT.plan(cycles)
    nominal = next(a for a, _b, s in plan if s.name == "escape")

    near = DAYLIGHT.plan(cycles, {"escape": [nominal + 0.05 * cycles]})
    moved = next(a for a, _b, s in near if s.name == "escape")
    assert moved != pytest.approx(nominal) and abs(moved - nominal) < ANCHOR_REACH * cycles

    far = DAYLIGHT.plan(cycles, {"escape": [0.02 * cycles]})
    assert next(a for a, _b, s in far if s.name == "escape") == pytest.approx(nominal)


def test_the_form_fits_whatever_length_the_flight_turned_out_to_be():
    for cycles in (4.0, 51.0, 900.0):
        plan = DAYLIGHT.plan(cycles)
        assert plan[0][0] == 0.0
        assert plan[-1][1] == pytest.approx(cycles)
        assert all(b >= a for a, b, _s in plan)
        assert all(plan[i][1] == pytest.approx(plan[i + 1][0]) for i in range(len(plan) - 1))


def test_arranging_is_deterministic_and_leaves_the_timing_alone():
    frames = score()
    a = Arranger(DAYLIGHT, seed=5).apply(frames)
    b = Arranger(DAYLIGHT, seed=5).apply(frames)
    assert [[(n.voice, n.note, n.velocity) for n in f.notes] for f in a] == \
           [[(n.voice, n.note, n.velocity) for n in f.notes] for f in b]
    assert [f.t for f in a] == [f.t for f in frames]          # not one note was moved in time
    for out_frame, in_frame in zip(a, frames):
        kept = {n.voice for n in out_frame.notes}
        for note in out_frame.notes:
            source = next(x for x in in_frame.notes if x.voice == note.voice)
            assert note.t == source.t
        assert kept <= {n.voice for n in in_frame.notes}      # and no note was invented


def test_it_moves_pitches_without_throwing_the_melody_away():
    """A snap that ignores the contour is a different tune, not an arrangement."""
    frames = score()
    out = Arranger(DAYLIGHT, seed=0).apply(frames)
    before = {f.t: n.note for f in frames for n in f.notes if n.voice == "wing_left"}
    after = {f.t: n.note for f in out for n in f.notes if n.voice == "wing_left"}
    assert len(after) >= len(before) * 0.5
    # Paired by the moment the note happened, because thinning means the two
    # lists are different lengths, and modulo the octave, because a section
    # moving the lead an octave is the arrangement doing its job. What is left
    # is how far the snap itself moved each note.
    moved = [min(abs(after[t] - before[t]) % 12, 12 - abs(after[t] - before[t]) % 12) for t in after]
    assert sum(moved) / len(moved) < 2.0                      # a whole tone, on average


def test_a_chord_knows_its_pitch_classes_and_its_root():
    chord = LYDIAN_PALETTE["II"]
    d = parse_root("D3") % 12
    assert chord.pitch_classes(d) == frozenset({(d + 2 + t) % 12 for t in chord.tones})
    assert (d + 2) % 12 in chord.root_classes(d) and len(chord.root_classes(d)) == 2


def test_the_avoid_note_rule_keeps_a_borrowed_chord_from_sounding_wrong():
    """Over the borrowed Cmaj7, the mode's G# and C# are a semitone above a
    chord tone, and the lead is not allowed to play them."""
    arr = Arranger(DAYLIGHT)
    allowed = arr._allowed(LYDIAN_PALETTE["bVII"], "mode")
    d = parse_root("D3") % 12
    assert (d + 6) % 12 not in allowed     # G#, a semitone above the chord's G
    assert (d + 11) % 12 not in allowed    # C#, a semitone above the chord's C
    assert (d + 2) % 12 in allowed         # E is in both the mode and the chord


def test_an_unknown_form_is_an_error_and_the_shipped_one_is_sound():
    with pytest.raises(KeyError):
        Arranger("nocturne")
    assert FORMS["daylight"] is DAYLIGHT
    for section in DAYLIGHT.sections:
        assert section.weight > 0 and section.chords
        assert all(c in DAYLIGHT.palette for c in section.chords)
        assert all(0.0 <= g <= 1.0 for g in section.roles.values())
        assert all(0.0 < d <= 1.0 for d in section.density.values())
    assert {r for s in DAYLIGHT.sections for r in s.roles} <= set(SNAP)


def test_a_form_of_one_section_still_works():
    form = Form(name="drone", key="A2", mode="dorian",
                sections=(Section("all", 1.0, ("I",), {r: 1.0 for r in SNAP}),),
                palette={"I": Chord("Im7", 0, (0, 3, 7, 10))})
    out = Arranger(form).apply(score(seconds=20))
    assert out and all(f.section == "all" for f in out)
    assert not math.isnan(sum(n.note for f in out for n in f.notes))


def test_an_empty_flight_arranges_to_nothing_rather_than_crashing():
    assert Arranger(DAYLIGHT).apply([]) == []


def test_the_arrangement_renders_and_masters_end_to_end():
    """The three modules meet here; a shape or a name that does not line up
    between them typechecks, passes their own tests and produces silence."""
    from flydrones.music.master import master
    from flydrones.music.synth import SynthConfig, describe, render

    frames = Arranger(DAYLIGHT, seed=0).apply(score(seconds=30))
    sr = 22050  # half rate: the same code path, a quarter of the test's time
    audio = render(frames, SynthConfig(sample_rate=sr, tail_s=1.0))
    out, report = master(audio, sr, fade_out_s=1.0)
    assert out.shape[0] == 2 and out.shape[1] > sr * 25
    assert report["lufs"] == pytest.approx(-14.0, abs=0.5)
    assert report["true_peak_dbtp"] <= -0.98
    assert describe(out, sr)["centroid_hz"] > 800


def _lead_stepwise(frames) -> tuple[float, int]:
    """(share of steps rather than leaps, number of jumps of an octave or more)."""
    steps = leaps = octaves = 0
    for voice in ("wing_left", "wing_right"):
        line = [n.note for f in frames for n in f.notes if n.voice == voice]
        for a, b in zip(line, line[1:]):
            if a == b:
                continue
            if abs(b - a) <= 4:
                steps += 1
            else:
                leaps += 1
            octaves += abs(b - a) >= 12
    return 100 * steps / max(1, steps + leaps), octaves


def test_voice_leading_is_what_keeps_the_melody(monkeypatch):
    """Snapping onto chords is what makes a line leap; voice leading is what
    stops it. Measured on the shipped flight, snapping alone took the lead from
    70% stepwise motion to 49% — which is the quality the piece was asked for,
    spent on the quality it was also asked for. This is the A/B."""
    frames = score()
    led, led_octaves = _lead_stepwise(Arranger(DAYLIGHT, seed=0).apply(frames))
    monkeypatch.setattr(Arranger, "_lead", staticmethod(lambda pitch, previous, **_: pitch))
    bare, bare_octaves = _lead_stepwise(Arranger(DAYLIGHT, seed=0).apply(frames))
    # The margin is small here and large on a real flight: the lines in `score`
    # walk by a tone and are already about as stepwise as a line can be, so
    # there is little for the leading to recover. The octaves are the tell.
    assert led > bare
    assert led_octaves == 0 < bare_octaves


def test_no_voice_is_moved_out_of_its_own_register():
    frames = score()
    out = Arranger(DAYLIGHT, seed=0).apply(frames)
    for voice in VOICES:
        was = [n.note for f in frames for n in f.notes if n.voice == voice]
        now = [n.note for f in out for n in f.notes if n.voice == voice]
        if not now:
            continue
        head = max(0, max((s.octave.get(ROLES[voice], 0) for s in DAYLIGHT.sections), default=0))
        foot = min(0, min((s.octave.get(ROLES[voice], 0) for s in DAYLIGHT.sections), default=0))
        assert min(now) >= min(was) + foot - 6
        assert max(now) <= max(was) + head + 6
