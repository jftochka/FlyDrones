"""The compositor, its wire formats and its receivers.

Nothing here needs Pure Data, Max, SuperCollider or a browser: the sinks are
pointed at real loopback sockets and the packets are decoded again, because a
music bridge that is only tested against itself is a music bridge that sends
nothing anybody can hear.
"""

from __future__ import annotations

import json
import math
import socket
import urllib.request

import pytest

from flydrones.brain import Brain, build_minifly
from flydrones.config import load_config
from flydrones.motor import FlightCommand
from flydrones.music import (
    Compositor,
    ControlEvent,
    EventServer,
    Frame,
    NoteEvent,
    Scale,
    make_sink,
    make_sinks,
    read_jsonl,
    to_strudel,
    to_tidal,
    transcribe,
)
from flydrones.music.conductor import improvisation
from flydrones.music.events import SCALES, note_name, parse_root
from flydrones.music.osc import decode, decode_message, encode_bundle, encode_message
from flydrones.music.patterns import strudel_live_snippet, tidal_live_file
from flydrones.runtime import TickInfo
from flydrones.safety import Telemetry


def cfg():
    return load_config()


def info(t: float, rates: dict, *, brain_ms: float | None = None, alt: float = 1.5, cmd: FlightCommand | None = None,
         raster=None, spikes: int = 0) -> TickInfo:
    c = cmd or FlightCommand()
    tel = Telemetry(t=t, alt_m=alt, yaw_deg=0.0, x_m=0.0, y_m=0.0, battery_pct=90.0, flying=True)
    return TickInfo(t=t, frame=None, rates=rates, raw=c, cmd=c, tel=tel, gesture=None, illusion="test",
                    raster=raster or [], rtf=10.0, spikes=spikes, brain_ms=(t + 0.05) * 1000 if brain_ms is None else brain_ms)


def udp_listener() -> tuple[socket.socket, int]:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(2.0)
    return s, s.getsockname()[1]


# ------------------------------------------------------------------- scales
def test_scale_quantises_into_the_scale_and_stays_in_range():
    scale = Scale.parse("minor_pentatonic", "C3")
    ladder = [scale.quantise(x / 20, 48, 72) for x in range(21)]
    assert ladder == sorted(ladder)  # more excitement is never a lower note
    assert min(ladder) == 48 and max(ladder) == 72
    assert all((n - scale.root) % 12 in scale.degrees for n in ladder)


def test_scale_survives_nonsense():
    scale = Scale.parse("dorian", 50)
    assert scale.quantise(float("nan"), 40, 60) == scale.notes_between(40, 60)[0]
    assert scale.quantise(5.0, 40, 60) == scale.notes_between(40, 60)[-1]
    assert scale.quantise(-3.0, 40, 60) == scale.notes_between(40, 60)[0]
    with pytest.raises(ValueError):
        Scale.parse("lipogram", "C3")


def test_note_names_round_trip():
    assert parse_root("C3") == 48 and parse_root("F#2") == 42 and parse_root("Bb4") == 70
    assert note_name(60) == "C4" and note_name(48) == "C3"
    assert all(len(v) >= 5 for v in SCALES.values())


# ---------------------------------------------------------------------- osc
def test_osc_message_round_trip_and_padding():
    packet = encode_message("/fly/note", ["wing_left", "flylead", 67, 0.8, True, None])
    assert len(packet) % 4 == 0
    address, args = decode_message(packet)
    assert address == "/fly/note"
    assert args[:3] == ["wing_left", "flylead", 67]
    assert args[3] == pytest.approx(0.8, rel=1e-6) and args[4] is True and args[5] is None
    with pytest.raises(ValueError):
        encode_message("fly/note", [])


def test_osc_bundle_carries_its_timetag():
    when = 1_800_000_000.5
    flat = decode(encode_bundle([encode_message("/a", [1]), encode_message("/b", [2.0])], when))
    assert [m[1] for m in flat] == ["/a", "/b"]
    assert flat[0][0] == pytest.approx(when, abs=1e-6)
    assert decode(encode_bundle([encode_message("/a", [1])]))[0][0] is None  # immediately


# --------------------------------------------------------------- compositor
def rising_rates(hz: float) -> dict:
    return {"DNg02_L": 31.0 + hz, "DNg02_R": 31.0 + hz, "R16_L": 14.0, "R16_R": 14.0}


def baselines() -> dict:
    return {"DNg02_L": 31.0, "DNg02_R": 31.0, "R16_L": 14.0, "R16_R": 14.0}


def test_a_quiet_brain_makes_no_music():
    comp = Compositor(cfg(), baselines=baselines())
    for k in range(20):
        frame = comp.tick(info(k * 0.05, rising_rates(0.0)))
        assert frame.notes == []
    assert frame.section == "hover"


def test_excitement_opens_a_voice_and_hysteresis_keeps_it_open():
    comp = Compositor(cfg(), baselines=baselines())
    comp.smoothing = 1.0  # no EMA: this test is about the threshold, not the filter
    for k in range(40):
        comp.tick(info(k * 0.05, rising_rates(20.0)))
    lift = next(v for v in comp.voices if v.spec.name == "lift")
    assert lift.notes > 2 and lift.open
    # on_hz 4, off_hz 2: between them the voice stays open rather than chattering
    comp.tick(info(3.0, rising_rates(3.0)))
    assert lift.open
    comp.tick(info(3.05, rising_rates(1.0)))
    assert not lift.open


def test_notes_stay_inside_the_range_they_were_given():
    comp = Compositor(cfg(), baselines=baselines())
    spec = next(v.spec for v in comp.voices if v.spec.name == "wing_left")
    seen = []
    for k in range(200):
        seen += [n for n in comp.tick(info(k * 0.05, rising_rates(k % 40))).notes if n.voice == "wing_left"]
    assert seen, "the melody voice never fired"
    assert all(spec.low <= n.note <= spec.high for n in seen)
    assert all(0.0 <= n.velocity <= 1.0 for n in seen)


def test_a_trigger_voice_fires_once_per_burst():
    comp = Compositor(cfg(), baselines={})
    gf = next(v for v in comp.voices if v.spec.name == "giant_fiber")
    comp.smoothing = 1.0
    for k in range(20):  # one long burst of the giant fiber
        comp.tick(info(k * 0.05, {"DNp01_L": 60.0, "DNp01_R": 60.0}))
    assert gf.notes == 1
    for k in range(20, 40):  # it goes quiet
        comp.tick(info(k * 0.05, {"DNp01_L": 0.0, "DNp01_R": 0.0}))
    for k in range(40, 50):  # and fires again
        comp.tick(info(k * 0.05, {"DNp01_L": 60.0, "DNp01_R": 60.0}))
    assert gf.notes == 2


def test_sections_follow_the_flight():
    comp = Compositor(cfg(), baselines=baselines())
    assert comp.tick(info(0.0, {}, cmd=FlightCommand(throttle=0.4))).section == "climb"
    assert comp.tick(info(0.1, {}, cmd=FlightCommand(throttle=-0.4))).section == "descend"
    assert comp.tick(info(0.2, {}, cmd=FlightCommand(yaw=0.5))).section == "turn"
    assert comp.tick(info(0.3, {}, cmd=FlightCommand(escape=True))).section == "escape"
    grounded = info(0.4, {})
    grounded.tel.flying = False
    assert comp.tick(grounded).section == "ground"


def test_controls_are_finite_even_when_the_flight_is_not():
    comp = Compositor(cfg(), baselines=baselines())
    broken = info(0.0, {"DNg02_L": float("nan"), "LPLC2_L": float("inf")}, alt=float("nan"))
    broken.tel.battery_pct = None
    for c in comp.tick(broken).controls:
        assert math.isfinite(c.value), c.name


def test_every_configured_voice_listens_to_a_group_the_brain_has():
    """A mistyped group name is a voice that silently never plays."""
    c = cfg()
    brain = Brain(build_minifly(), c)
    comp = Compositor(c, brain=brain)
    assert len(comp.voices) >= 5
    for voice in comp.voices:
        assert voice.spec.groups, voice.spec.name
        for group in voice.spec.groups:
            known = {**c["inputs"], **c["outputs"], **c.get("monitors", {})}
            assert group in known, f"{voice.spec.name}: {group}"
            assert brain.connectome.group(group).size > 0, group


def test_spike_times_place_notes_inside_the_tick():
    c = cfg()
    brain = Brain(build_minifly(), c)
    comp = Compositor(c, baselines=baselines(), brain=brain)
    comp.smoothing = 1.0
    positions = [i for i, n in enumerate(brain.record) if n in set(brain.connectome.group("DNg02_L").tolist())]
    assert positions, "no DNg02_L neuron is in the recorded set"
    # each tick covers 50 ms of brain time and its group fires 12.5 ms in
    frames = [comp.tick(info(k * 0.05, rising_rates(25.0), brain_ms=1000.0 + (k + 1) * 50,
                             raster=[(1000.0 + k * 50 + 12.5, positions[:2])])) for k in range(4)]
    notes = [n for f in frames for n in f.notes if n.voice in ("lift", "wing_left")]
    assert notes
    assert any(n.t > f.t for f in frames for n in f.notes), "spike times never moved a note"
    for f in frames:
        for n in f.notes:
            assert f.t <= n.t <= f.t + 0.05


# -------------------------------------------------------------------- sinks
def test_osc_sink_sends_the_documented_schema():
    sock, port = udp_listener()
    sink = make_sink(f"pd:{port}", cfg())
    frame = Frame(t=2.0, cycle=1.0, cps=0.5, section="climb")
    frame.notes = [NoteEvent(t=2.02, voice="lift", sound="flybass", note=48, velocity=0.9, duration=0.35,
                             pan=0.5, orbit=0, channel=1, rate_hz=12.5)]
    frame.controls = [ControlEvent(t=2.0, name="drive", value=0.42)]
    sink.frame(frame)
    messages = {addr: args for _when, addr, args in decode(sock.recv(65535))}
    sink.close()
    sock.close()
    assert messages["/fly/frame"][:3] == pytest.approx([2.0, 1.0, 0.5])
    assert messages["/fly/section"] == ["climb"]
    assert messages["/fly/ctrl"] == ["drive", pytest.approx(0.42, rel=1e-5)]
    note = messages["/fly/note"]
    assert note[:3] == ["lift", "flybass", 48]
    assert note[-1] == pytest.approx(220.0, abs=1.0)  # 0.2 s latency + 20 ms into the tick


def test_fudi_sends_one_datagram_per_message():
    sock, port = udp_listener()
    sink = make_sink(f"pd-fudi:{port}", cfg())
    frame = Frame(t=0.0, cycle=0.0, cps=0.5, section="hover")
    frame.notes = [NoteEvent(t=0.0, voice="wing right", sound="flylead", note=60)]
    sink.frame(frame)
    seen = [sock.recv(4096).decode() for _ in range(3)]
    sink.close()
    sock.close()
    assert all(line.endswith(";\n") for line in seen)
    assert seen[0].startswith("frame ") and seen[1].startswith("section ")
    assert "wing_right" in seen[2]  # spaces would end the atom, so they are not sent


def test_tidal_sends_ctrl_pairs_for_voices_and_controls():
    sock, port = udp_listener()
    sink = make_sink(f"tidal:{port}", cfg())
    frame = Frame(t=0.0, cycle=3.0, cps=0.5, section="escape")
    frame.notes = [NoteEvent(t=0.0, voice="giant_fiber", sound="flycrash", note=28, velocity=1.0)]
    frame.controls = [ControlEvent(t=0.0, name="loom", value=0.9)]
    sink.frame(frame)
    pairs = {args[0]: args[1] for _w, addr, args in decode(sock.recv(65535)) if addr == "/ctrl"}
    sink.close()
    sock.close()
    assert pairs["loom"] == pytest.approx(0.9, rel=1e-5)
    assert pairs["section"] == "escape"
    assert pairs["giant_fiber_n"] == pytest.approx(28.0)
    assert pairs["giant_fiber_g"] == pytest.approx(1.0)


def test_superdirt_message_is_a_timestamped_dirt_play():
    import time

    sock, port = udp_listener()
    sink = make_sink(f"superdirt:{port}", cfg())
    frame = Frame(t=1.0, cycle=2.0, cps=0.5)
    frame.notes = [NoteEvent(t=1.0, voice="lift", sound="flybass", note=72, velocity=0.7, duration=0.25, orbit=2)]
    before = time.time()
    sink.frame(frame)
    when, address, args = decode(sock.recv(65535))[0]
    sink.close()
    sock.close()
    assert address == "/dirt/play"
    params = dict(zip(args[::2], args[1::2]))
    assert params["s"] == "supersaw"  # the generic name is mapped to a synth SuperDirt has
    assert params["note"] == pytest.approx(12.0)  # semitones from middle C
    assert params["orbit"] == 2 and params["cycle"] == pytest.approx(2.0)
    assert before + 0.1 < when < before + 1.0  # scheduled, not immediate


def test_make_sink_parses_targets_and_refuses_nonsense():
    assert make_sink("max", cfg()).out.port == 7400
    assert make_sink("maxmsp:7401", cfg()).out.port == 7401
    s = make_sink("pd:192.168.1.9:9001", cfg())
    assert (s.out.host, s.out.port) == ("192.168.1.9", 9001)
    with pytest.raises(ValueError):
        make_sink("ableton", cfg())


def test_jsonl_round_trips_a_score(tmp_path):
    path = tmp_path / "score.jsonl"
    sinks = make_sinks([f"jsonl:{path}"], cfg())
    frame = Frame(t=0.5, cycle=0.25, cps=0.5, section="hover")
    frame.notes = [NoteEvent(t=0.52, voice="lift", sound="flybass", note=50, velocity=0.6)]
    frame.controls = [ControlEvent(t=0.5, name="alt", value=0.75)]
    sinks.frame(frame)
    sinks.close()
    back = read_jsonl(path)
    assert len(back) == 1 and back[0].section == "hover"
    assert back[0].notes[0].note == 50 and back[0].notes[0].t == pytest.approx(0.52)
    assert {c.name: c.value for c in back[0].controls}["alt"] == pytest.approx(0.75)


def test_event_server_streams_frames_to_a_browser():
    hub = EventServer("127.0.0.1", 0).start()
    try:
        stream = urllib.request.urlopen(hub.url + "events", timeout=5)
        assert stream.readline() == b": flydrones\n"
        hub.publish({"t": 1.0, "section": "climb", "notes": [], "controls": {"drive": 0.5}})
        line = stream.readline()
        while line in (b"\n", b": keepalive\n"):
            line = stream.readline()
        assert line.startswith(b"data: ")
        assert json.loads(line[6:])["section"] == "climb"
        stream.close()
        page = urllib.request.urlopen(hub.url, timeout=5).read().decode()
        assert "flybrain.mjs" in page and "initStrudel" in page
        assert json.loads(urllib.request.urlopen(hub.url + "state.json", timeout=5).read())["t"] == 1.0
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(hub.url + "nope", timeout=5)
    finally:
        hub.close()


# ----------------------------------------------------------------- patterns
def score_frames() -> list[Frame]:
    frames = []
    for k in range(32):  # 1.25 cps at 20 Hz: one tick is exactly one 16th step
        f = Frame(t=k * 0.05, cycle=k / 16, cps=1.25, section="hover")
        if k % 4 == 0:
            f.notes = [NoteEvent(t=k * 0.05, voice="lift", sound="flybass", note=48 + k // 4, pan=0.5)]
        frames.append(f)
    return frames


def test_transcription_lands_on_the_grid():
    score = transcribe(score_frames(), steps=16)
    assert score.notes == 8 and score.dropped == 0
    row = score.voices["lift"][0]
    assert [i for i, n in enumerate(row) if n is not None] == [0, 4, 8, 12]
    assert score.cycles == 2
    assert row[0] == 48


def test_transcription_writes_playable_mini_notation():
    score = transcribe(score_frames(), steps=16)
    tidal, strudel = to_tidal(score), to_strudel(score)
    assert "d1 $ stack [" in tidal and "setcps 1.25" in tidal
    assert '# s "supersaw"' in tidal
    assert "-12" in tidal  # 48 is twelve semitones below Tidal's middle C
    assert "stack(" in strudel and 'note("' in strudel and "48" in strudel and "setcps(1.2500)" in strudel
    assert "<[" in strudel  # one <> slot per cycle of the flight
    assert to_tidal(transcribe([], steps=16)).strip().endswith("silence")


def test_live_files_name_the_voices_they_were_built_for():
    voices = ["lift", "wing_left", "giant_fiber"]
    controls = ["drive", "loom"]
    tidal = tidal_live_file(voices, controls)
    assert "cF" in tidal and "giant_fiber" in tidal and "drive, loom" in tidal
    snippet = strudel_live_snippet(voices, controls, "http://127.0.0.1:9999/events")
    assert "EventSource('http://127.0.0.1:9999/events')" in snippet and "stack(" in snippet


def test_the_conductor_is_deterministic():
    assert improvisation(40, seed=7) == improvisation(40, seed=7)
    assert improvisation(40, seed=7) != improvisation(40, seed=8)
    timeline = improvisation(60, seed=1)
    assert timeline[0][0] == 0.0 and timeline[-1][0] < 60
    assert all(b[0] >= a[0] for a, b in zip(timeline, timeline[1:]))
    assert all(a[1].label != b[1].label for a, b in zip(timeline, timeline[1:]))


# ------------------------------------------------------------- the real loop
def test_a_real_flight_plays_and_replays_the_same_music():
    """The whole chain on MiniFly, twice: same seed, same notes."""
    from flydrones.drones import SimDrone
    from flydrones.runtime import Pilot, run_sim
    from flydrones.senses import ScriptedGestures, demo_timeline

    def fly() -> list[Frame]:
        c = cfg()
        brain = Brain(build_minifly(), c)
        pilot = Pilot(brain, SimDrone(start=(-1.5, 0.0, 0.0), seed=3), c, gestures=ScriptedGestures(demo_timeline()))
        comp = Compositor(c, brain=brain)
        out: list[Frame] = []

        def on_tick(_k, infos):
            if not comp.baselines:
                comp.set_baselines(pilot.decoder.baseline)
            out.append(comp.tick(infos[0]))

        run_sim([pilot], 8.0, hz=c["control"]["hz"], on_tick=on_tick)
        return out

    a, b = fly(), fly()
    notes = [n for f in a for n in f.notes]
    assert len(notes) > 10, "a whole flight and nothing to play"
    assert {n.voice for n in notes} >= {"lift", "wing_left", "wing_right"}
    assert [(n.t, n.voice, n.note, n.velocity) for n in notes] == [(n.t, n.voice, n.note, n.velocity)
                                                                   for f in b for n in f.notes]
    assert all(f.cps > 0 for f in a)
