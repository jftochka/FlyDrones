"""Radio Cognitive Fruit Fly: the programme, and a station that will not stop."""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pytest

from flydrones.config import load_config
from flydrones.radio import PROGRAMME, STATION, Show, Station


def short(show: Show, minutes: float = 0.05) -> Show:
    return Show(show.name, minutes, show.blurb, show.scale, show.root, show.cps, show.tempo_from,
                show.cruise, show.learning, show.gestures, show.weights, show.laps, show.goal_every_s)


def station(targets: str = "jsonl:/dev/null", shows=None, minutes: float = 0.05, **kw) -> Station:
    programme = [short(s, minutes) for s in (shows or PROGRAMME[:3])]
    return Station(load_config(), targets=targets, programme=programme, **kw)


def test_the_programme_is_a_programme():
    assert len(PROGRAMME) >= 4
    names = [s.name for s in PROGRAMME]
    assert len(set(names)) == len(names)
    for s in PROGRAMME:
        assert s.minutes > 0 and s.blurb and s.cps > 0
        assert s.scale in __import__("flydrones.music.events", fromlist=["SCALES"]).SCALES
    assert any(s.laps for s in PROGRAMME), "nothing in the programme teaches the fly anything"
    assert any(s.goal_every_s for s in PROGRAMME), "nothing in the programme uses the compass"


def test_a_station_flies_composes_and_says_what_it_is_doing(tmp_path):
    st = station(targets=f"jsonl:{tmp_path / 'radio.jsonl'}")
    st.start()
    for _ in range(80):  # four seconds of air time
        st.tick()
    ctx = st.context()
    assert ctx["station"] == STATION
    assert ctx["show"]["name"] == st.show.name and ctx["next"] == st.next_show.name
    assert ctx["brain"]["neurons"] > 500
    assert any(e["kind"] == "show" for e in ctx["log"])
    assert st.frames == 80 and st.t == pytest.approx(4.0, abs=0.1)
    st.close()
    rows = [json.loads(line) for line in (tmp_path / "radio.jsonl").read_text().splitlines()]
    assert len(rows) == 80
    assert sum(len(r["notes"]) for r in rows) > 5, "the station went out silent"


def test_shows_change_on_time_and_take_the_key_with_them():
    st = station(minutes=0.05)  # three seconds each
    st.start()
    first = st.show.name
    assert st.compositor.scale.name == st.show.scale
    for _ in range(int(3.0 * st.hz) + 2):
        st.tick()
    assert st.show.name != first, "the programme never moved on"
    assert st.compositor.scale.name == st.show.scale
    assert st.compositor.base_cps == pytest.approx(st.show.cps)
    assert st.pilot.decoder.cruise == pytest.approx(st.show.cruise)
    assert st.counts["shows"] >= 2
    st.close()


def test_the_station_puts_the_fly_back_in_the_air():
    st = station()
    st.start()
    for _ in range(10):
        st.tick()
    st.pilot.safety.land_requested = True
    st.pilot.drone.flying = False
    st.tick()
    assert st.pilot.drone.flying, "the station landed and stayed down"
    assert not st.pilot.safety.land_requested
    assert any(e.kind == "recovery" for e in st.entries)
    st.close()


def test_the_station_swaps_the_pack():
    st = station()
    st.start()
    st.pilot.drone.battery = 12.0
    st.tick()
    assert st.pilot.drone.battery > 90
    assert st.counts["packs"] == 1
    assert any(e.kind == "pack" for e in st.entries)
    st.close()


def test_the_chair_show_flies_laps_and_teaches():
    st = station(shows=[s for s in PROGRAMME if s.laps], minutes=0.6)
    st.start()
    for _ in range(int(30 * st.hz)):
        st.tick()
    assert st.counts["laps"] >= 2, "it never went back for another approach"
    assert st.pilot.cognition.state.memory > 0.0, "thirty seconds of flying at a chair taught it nothing"
    st.close()


def test_the_compass_show_sets_goals():
    st = station(shows=[s for s in PROGRAMME if s.goal_every_s], minutes=0.5)
    st.start()
    for _ in range(int(12 * st.hz)):
        st.tick()
    assert st.counts["goals"] >= 1
    assert st.pilot.cognition.goal_deg is not None
    assert any(e.kind == "goal" for e in st.entries)
    st.close()


def test_the_log_stays_short_and_the_line_reads():
    st = station(log_size=5)
    st.start()
    for i in range(12):
        st.log("test", f"entry {i}")
    assert len(st.entries) == 5
    assert "Radio Cognitive Fruit Fly" in st.now_playing()
    st.close()


def test_a_listener_gets_the_station_page_and_the_station_state():
    st = station(targets="strudel:127.0.0.1:0")
    st.start()
    hub = st.sinks.sinks[0].server
    try:
        for _ in range(5):
            st.tick()
        page = urllib.request.urlopen(hub.url, timeout=5).read().decode()
        assert "Radio Cognitive Fruit Fly" in page and "flybrain.mjs" in page
        state = json.loads(urllib.request.urlopen(hub.url + "state.json", timeout=5).read())
        assert state["station"] == STATION
        assert state["show"]["name"] and "controls" in state and "log" in state
        assert state["listeners"] == 0
    finally:
        st.close()


def test_two_stations_with_the_same_seed_play_the_same_show():
    a, b = station(seed=5), station(seed=5)
    a.start()
    b.start()
    for _ in range(40):
        a.tick()
        b.tick()
    assert [e.text for e in a.entries] == [e.text for e in b.entries]
    assert np.isclose(a.pilot.drone.pos, b.pilot.drone.pos).all()
    a.close()
    b.close()
