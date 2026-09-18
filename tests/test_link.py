"""The Orange LTE router, the prober, and what the governor does about them."""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from flydrones.config import load_config
from flydrones.link import (
    ORANGE_ROUTERS,
    THRESHOLDS,
    LinkError,
    LinkProbe,
    LinkQuality,
    MockRouter,
    Pinger,
    Signal,
    grade,
    interpolate,
    make_router,
    parse_huawei_plmn,
    parse_huawei_signal,
    parse_huawei_status,
    parse_huawei_traffic,
    parse_zte,
    score_signal,
)
from flydrones.motor import FlightCommand
from flydrones.safety import SafetyGovernor, Telemetry

# Real shapes, from the APIs those boxes actually serve.
HUAWEI_SIGNAL = """<?xml version="1.0" encoding="UTF-8"?><response><pci>231</pci><sc></sc>
<cell_id>21102341</cell_id><rsrq>-11dB</rsrq><rsrp>-95dBm</rsrp><rssi>-70dBm</rssi><sinr>12dB</sinr>
<rscp></rscp><ecio></ecio><mode>7</mode><ulbandwidth>20MHz</ulbandwidth><dlbandwidth>20MHz</dlbandwidth>
<band>3</band><earfcn>DL:1650 UL:19650</earfcn></response>"""
HUAWEI_STATUS = """<response><ConnectionStatus>901</ConnectionStatus><SignalStrength></SignalStrength>
<CurrentNetworkType>19</CurrentNetworkType><CurrentNetworkTypeEx>1011</CurrentNetworkTypeEx>
<SignalIcon>4</SignalIcon><maxsignal>5</maxsignal></response>"""
HUAWEI_PLMN = "<response><State>0</State><FullName>Orange F</FullName><ShortName>Orange</ShortName><Numeric>20801</Numeric><Rat>6</Rat></response>"
HUAWEI_TRAFFIC = """<response><CurrentConnectTime>3612</CurrentConnectTime><CurrentUpload>12345</CurrentUpload>
<CurrentUploadRate>125000</CurrentUploadRate><CurrentDownloadRate>1250000</CurrentDownloadRate></response>"""
ZTE_JSON = json.dumps({"lte_rsrp": "-101", "lte_rsrq": "-14", "lte_snr": "5.2", "lte_band": "20",
                       "lte_pci": "77", "cell_id": "1A2B3C", "rssi": "-78", "network_type": "LTE",
                       "ppp_status": "ppp_connected", "network_provider": "Orange Polska",
                       "realtime_tx_thrpt": "62500", "realtime_rx_thrpt": "500000", "realtime_time": "900"})


# --------------------------------------------------------------------- parsing
def test_huawei_endpoints_parse():
    sig = parse_huawei_signal(HUAWEI_SIGNAL)
    assert sig["rsrp_dbm"] == -95 and sig["rsrq_db"] == -11 and sig["sinr_db"] == 12
    assert sig["band"] == "B3" and sig["cell_id"] == "21102341" and sig["pci"] == "231"
    assert parse_huawei_status(HUAWEI_STATUS) == {"connected": True, "network": "LTE+"}
    assert parse_huawei_plmn(HUAWEI_PLMN) == {"operator": "Orange F", "plmn": "20801"}
    traffic = parse_huawei_traffic(HUAWEI_TRAFFIC)
    assert traffic["up_kbps"] == pytest.approx(1000.0)  # 125 kB/s is a megabit
    assert traffic["down_kbps"] == pytest.approx(10000.0) and traffic["uptime_s"] == 3612


def test_a_disconnected_huawei_is_not_connected():
    assert parse_huawei_status("<response><ConnectionStatus>902</ConnectionStatus></response>")["connected"] is False
    assert parse_huawei_signal("<response><rsrp></rsrp><sinr></sinr></response>")["rsrp_dbm"] is None


def test_zte_parses():
    z = parse_zte(ZTE_JSON)
    assert z["rsrp_dbm"] == -101 and z["sinr_db"] == pytest.approx(5.2) and z["band"] == "B20"
    assert z["operator"] == "Orange Polska" and z["connected"] is True
    assert z["down_kbps"] == pytest.approx(4000.0)
    assert parse_zte('{"ppp_status": "ppp_disconnected"}')["connected"] is False
    assert parse_zte("{}")["rsrp_dbm"] is None


def test_the_orange_boxes_are_named_and_routed():
    assert {"orange-airbox", "orange-flybox", "orange-home-4g", "orange-flybox-zte"} <= set(ORANGE_ROUTERS)
    assert make_router("orange-airbox").host == "192.168.8.1"
    assert make_router("orange-flybox-zte").host == "192.168.0.1"
    assert make_router("orange-airbox", "10.0.0.1").host == "10.0.0.1"
    assert make_router("huawei").kind == "huawei" and make_router("zte").kind == "zte"
    with pytest.raises(ValueError):
        make_router("linksys")


def test_an_unreachable_router_is_a_signal_with_an_error_not_an_exception():
    router = make_router("huawei", "127.0.0.1:1")  # nothing listens there
    router.host = "127.0.0.1:1"
    signal = router.signal()
    assert signal.error and not signal.ok and signal.connected is False
    assert "unreachable" in signal.error or "HTTP" in signal.error


# --------------------------------------------------------------------- scoring
def test_the_thresholds_are_monotonic_and_bounded():
    for metric, table in THRESHOLDS.items():
        values = [v for v, _ in table]
        scores = [s for _, s in table]
        assert values == sorted(values), metric
        assert scores == sorted(scores) and 0.0 <= scores[0] and scores[-1] <= 1.0, metric
    assert interpolate(THRESHOLDS["rsrp_dbm"], -200) == 0.0  # clamped, not extrapolated
    assert interpolate(THRESHOLDS["rsrp_dbm"], 0) == 1.0
    assert 0.4 < interpolate(THRESHOLDS["rsrp_dbm"], -100) < 0.5


def test_score_follows_the_signal():
    good = Signal(connected=True, rsrp_dbm=-75, rsrq_db=-8, sinr_db=21)
    edge = Signal(connected=True, rsrp_dbm=-112, rsrq_db=-18, sinr_db=-2)
    assert score_signal(good)[0] > 0.9
    assert score_signal(edge)[0] < 0.15
    assert score_signal(edge)[1], "a bad link should say what is wrong with it"
    assert score_signal(Signal(connected=False))[0] == 0.0
    assert score_signal(Signal(connected=True))[0] == 0.0  # connected but reporting nothing
    assert grade(0.8) == "excellent" and grade(0.4) == "fair" and grade(0.0) == "unusable"


def test_a_slow_link_scores_below_a_fast_one_with_the_same_signal():
    class Fixed(MockRouter):
        def read(self):
            return {"rsrp_dbm": -80, "rsrq_db": -9, "sinr_db": 18, "connected": True, "band": "B3"}

    fast, slow = LinkProbe(Fixed()), LinkProbe(Fixed())
    fast.pinger = type("P", (), {"rtt_ms": lambda self: 30.0, "host": "x", "port": 1})()
    slow.pinger = type("P", (), {"rtt_ms": lambda self: 600.0, "host": "x", "port": 1})()
    a, b = fast.sample(), slow.sample()
    assert a.score > b.score
    assert any("RTT" in r for r in b.reasons)
    assert a.grade in ("excellent", "good") and b.score < a.score * 0.7


def test_loss_counts_against_the_link():
    class Fixed(MockRouter):
        def read(self):
            return {"rsrp_dbm": -80, "rsrq_db": -9, "sinr_db": 18, "connected": True}

    probe = LinkProbe(Fixed())
    answers = [40.0, None, 45.0, None]
    probe.pinger = type("P", (), {"rtt_ms": lambda self: answers.pop(0), "host": "x", "port": 1})()
    for _ in range(4):
        q = probe.sample()
    assert q.loss == pytest.approx(0.5)
    assert any("loss" in r for r in q.reasons)
    assert q.score < 0.6


# ----------------------------------------------------------------------- probe
def test_the_mock_router_drives_out_of_coverage():
    router = MockRouter(seconds_to_edge=60)
    probe = LinkProbe(router)
    router.now = router.t0
    start = probe.sample()
    router.now = router.t0 + 30
    middle = probe.sample()
    router.now = router.t0 + 59.9
    end = probe.sample()
    assert start.score > middle.score > end.score
    assert start.grade == "excellent" and end.grade in ("poor", "unusable")
    assert start.signal.operator == "Orange" and start.signal.band == "B3"
    assert end.signal.rsrp_dbm < start.signal.rsrp_dbm - 30


def test_the_probe_polls_in_the_background_and_stops():
    probe = LinkProbe(MockRouter(seconds_to_edge=5), period_s=0.05).start()
    try:
        assert probe.last is not None, "start() should take one reading before returning"
        time.sleep(0.3)
        assert probe.samples > 2
    finally:
        probe.stop()
    taken = probe.samples
    time.sleep(0.2)
    assert probe.samples == taken, "the thread kept going after stop()"


def test_the_pinger_times_a_real_socket():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    threading.Thread(target=lambda: server.accept(), daemon=True).start()
    try:
        rtt = Pinger("127.0.0.1", port, timeout=1.0).rtt_ms()
        assert rtt is not None and 0 <= rtt < 500
    finally:
        server.close()
    assert Pinger("127.0.0.1", 1, timeout=0.3).rtt_ms() is None


# ---------------------------------------------------------------------- safety
def quality(score: float, age: float = 0.0) -> LinkQuality:
    return LinkQuality(t=time.monotonic() - age, score=score, grade=grade(score))


def governor(**over) -> SafetyGovernor:
    cfg = load_config()
    cfg["safety"]["link"] = {**cfg["safety"]["link"], **over}
    return SafetyGovernor(cfg)


def test_one_bad_reading_is_not_a_lost_link():
    g = governor(grace_s=2.0)
    now = time.monotonic()
    assert g.check_link(quality(0.05), now)[:2] == (False, False)
    assert g.check_link(quality(0.9), now + 1)[:2] == (False, False)
    assert g.check_link(quality(0.05), now + 2)[:2] == (False, False), "the clock should have restarted"


def test_a_link_that_stays_bad_holds_and_then_lands():
    g = governor(grace_s=1.0)
    now = time.monotonic()
    g.check_link(quality(0.25), now)
    hold, land, why = g.check_link(quality(0.25), now + 1.5)
    assert hold and not land and "poor" in why
    g.check_link(quality(0.05), now + 2)
    hold, land, _ = g.check_link(quality(0.05), now + 4)
    assert hold and land


def test_a_silent_probe_counts_as_no_link():
    g = governor(grace_s=0.5, timeout_s=3.0)
    now = time.monotonic()
    stale = quality(0.95, age=10.0)  # a good reading, from ten seconds ago
    g.check_link(stale, now)
    hold, land, why = g.check_link(stale, now + 1)
    assert hold and land and "silent" in why


def test_the_governor_holds_the_drone_still_and_then_lands_it():
    g = governor(grace_s=0.0)
    tel = Telemetry(t=time.monotonic(), alt_m=1.5, flying=True)
    forward = FlightCommand(throttle=0.3, yaw=0.4, forward=0.4)
    out = g.filter(forward, tel, 0.05, link=quality(0.25))
    assert out.forward == 0.0 and out.yaw == 0.0 and "hold" in out.note
    assert not g.land_requested
    out = g.filter(forward, tel, 0.05, link=quality(0.02))
    assert g.land_requested and "land" in " ".join(g.events)


def test_no_probe_means_the_governor_behaves_exactly_as_before():
    g = governor()
    tel = Telemetry(t=time.monotonic(), alt_m=1.5, flying=True)
    out = g.filter(FlightCommand(forward=0.3), tel, 0.05)
    assert out.forward == pytest.approx(0.125, abs=0.13)  # slew-limited, not held
    assert not g.land_requested
    assert g.check_link(None) == (False, False, "")


def test_a_flight_can_carry_a_probe():
    from flydrones.brain import Brain, build_minifly
    from flydrones.drones import SimDrone
    from flydrones.runtime import Pilot

    cfg = load_config()
    probe = LinkProbe(MockRouter(seconds_to_edge=600), period_s=0.05).start()
    try:
        pilot = Pilot(Brain(build_minifly(), cfg), SimDrone(start=(0, 0, 0)), cfg, link=probe)
        pilot.drone.connect()
        pilot.warmup(0.3, 0.05)
        info = pilot.tick(0.0, 0.05)
        assert info.link is not None and info.link.score > 0.8
        assert info.cmd.note == "" or "lte" not in info.cmd.note
    finally:
        probe.stop()


def test_make_router_auto_says_something_useful_when_there_is_nothing():
    with pytest.raises(LinkError) as e:
        make_router("auto", "127.0.0.1:1", timeout=0.2)
    message = str(e.value)
    assert "127.0.0.1:1" in message, "it should name the address it actually tried"
    assert "huawei" in message and "zte" in message and "mock" in message
