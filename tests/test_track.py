"""Flight tracks: what the 3D replay page reads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flydrones.config import load_config
from flydrones.learn import approach_config, fly_lap, make_pilot
from flydrones.track import FIELDS, VERSION, FlightNarrator, TrackRecorder, load

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "docs" / "live" / "tracks" / "learn.json"


def a_flight(seconds: float = 4.0, hz: float = 20.0):
    cfg = approach_config(load_config(), learning=True)
    pilot = make_pilot(cfg)
    rec = TrackRecorder(title="test flight", subtitle="four seconds of it", hz=25.0)
    narrator = FlightNarrator(rec)
    fly_lap(pilot, 1, seconds=seconds, hz=hz,
            on_tick=lambda p, info, _lap: rec.add(info, hit=narrator.watch(info, p.drone)))
    return rec, narrator, pilot


def test_a_track_records_where_the_drone_was_and_what_the_brain_knew():
    rec, _, _ = a_flight()
    assert 40 < len(rec.frames) < 120  # 4 s at 25 Hz, from a 20 Hz control loop
    assert rec.seconds == pytest.approx(4.0, abs=0.3)
    rows = [dict(zip(FIELDS, row)) for row in rec.frames]
    assert all(len(row) == len(FIELDS) for row in rec.frames)
    assert all(0.0 <= r["z"] < 3.0 for r in rows), "the drone left the room"
    assert all(-4 < r["x"] < 4 and -4 < r["y"] < 4 for r in rows)
    assert all(0.0 <= r["heading"] <= 360.0 or r["heading"] == -1 for r in rows)
    assert any(r["kc"] > 0 for r in rows), "the mushroom body never saw anything"
    assert rows[-1]["x"] > rows[0]["x"], "it never went anywhere"


def test_a_track_round_trips_through_json(tmp_path):
    rec, _, _ = a_flight(2.0)
    rec.chapter(0.0, "approach 1", "memory 0.00")
    rec.event(1.0, "escape", "the giant fibre fired")
    path = rec.save(tmp_path / "t.json")
    data = load(path)
    assert data["version"] == VERSION and data["fields"] == list(FIELDS)
    assert data["title"] == "test flight" and data["chapters"][0]["name"] == "approach 1"
    assert data["events"][0]["kind"] == "escape"
    assert len(data["rows"]) == len(rec.frames)
    assert data["rows"][0]["t"] == rec.frames[0][0]
    assert path.stat().st_size < 200_000


def test_the_recorder_decimates_to_its_own_rate():
    rec = TrackRecorder(hz=5.0)

    class Tick:  # the smallest thing that looks like a TickInfo
        def __init__(self, t):
            self.t = t
            self.rates = {}
            self.cmd = type("C", (), {"throttle": 0.0, "yaw": 0.0, "forward": 0.0, "escape": False})()
            self.tel = type("T", (), {"x_m": 0.0, "y_m": 0.0, "alt_m": 1.0, "yaw_deg": 0.0})()
            self.cognition = None

    kept = [rec.add(Tick(i * 0.05)) for i in range(40)]  # 2 s at 20 Hz
    assert sum(kept) == 10  # 5 Hz
    assert len(rec.frames) == 10


def test_the_narrator_says_each_thing_once():
    rec = TrackRecorder()
    narrator = FlightNarrator(rec, memory_step=0.05, min_gap_s=2.0)

    class Tick:
        def __init__(self, t, escape=False, memory=0.0):
            self.t = t
            self.cmd = type("C", (), {"escape": escape})()
            self.cognition = type("G", (), {"memory": memory, "goal_deg": None})()

    for i in range(20):  # one long escape, not twenty
        narrator.watch(Tick(i * 0.05, escape=True, memory=0.0))
    assert narrator.counts["escapes"] == 1
    narrator.watch(Tick(2.0, memory=0.3))
    narrator.watch(Tick(2.2, memory=0.6))  # inside the gap: no second caption
    assert len([e for e in rec.events if e["kind"] == "learning"]) == 1


def test_the_shipped_track_is_the_one_the_replay_page_expects():
    """docs/live/tracks/learn.json is what replay.html loads by default."""
    assert SHIPPED.exists(), "regenerate with: flydrones learn --laps 8 --track docs/live/tracks/learn.json"
    data = json.loads(SHIPPED.read_text(encoding="utf-8"))
    assert data["fields"] == list(FIELDS), "the track format moved; re-record the shipped track"
    assert data["version"] == VERSION
    assert data["seconds"] > 30 and len(data["frames"]) > 500
    assert data["chapters"] and data["events"]
    assert any(e["kind"] == "learning" for e in data["events"]), "a learning flight with nothing learned"
    assert SHIPPED.stat().st_size < 400_000, "the shipped track should stay small enough to serve from Pages"


def test_the_replay_page_is_wired_to_a_track_and_to_the_recorder():
    """CI has no browser, so this is the cheap guard on the page's wiring."""
    html = (ROOT / "docs" / "live" / "replay.html").read_text(encoding="utf-8")
    js = (ROOT / "docs" / "live" / "replay.js").read_text(encoding="utf-8")
    assert './replay.js' in html and 'id="scene"' in html
    for element in ("caption", "chapter", "v-memory", "needle", "seek"):
        assert f'id="{element}"' in html, element
    for hook in ("caption", "chapter", "needle", "seek", "v-${name}"):
        assert hook in js, hook
    for hook in ("window.FDR", "seek(t)", "frame(dt", "ready: true", "duration"):
        assert hook in js, hook
    assert "tracks/learn.json" in js, "the page has to have a track to show by default"
    assert "./models.js" in js and "./room.js" in js, "the replay should reuse the demo's 3D models"
