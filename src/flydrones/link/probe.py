"""The LTE prober: what the link is doing, as one number the governor can act on.

A drone on a mobile link fails in a way a drone on Wi-Fi does not. Wi-Fi drops
at a wall and you know why; LTE degrades — RSRP slides as you fly away from the
mast, SINR collapses when the cell fills up at five o'clock, the round trip
goes from 40 ms to 600 ms — and the commands still arrive, late. So the prober
measures both halves of it:

* what the modem sees, read from the router (RSRP, RSRQ, SINR, band, cell);
* what the packets see, measured by opening a TCP connection to the far end and
  timing it — round trip, jitter, and how many attempts never landed.

Those become a **score from 0 to 1** with the reasons that dragged it down, and
the safety governor turns that into hold, or land, or nothing at all. The
thresholds are the usual LTE ones and every one of them is in ``THRESHOLDS``
below rather than buried in a comparison.

    probe = LinkProbe(make_router("orange-airbox"), target=("192.168.8.100", 8889))
    probe.start()
    ...
    q = probe.last           # never blocks the flight loop; None until the first poll
    print(q.line())
"""

from __future__ import annotations

import socket
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from .router import LteRouter, Signal

# metric -> [(value, score)], linearly interpolated between the points and
# clamped outside them. The numbers are the ones network engineers use:
# RSRP is coverage, RSRQ is interference and load, SINR is what throughput follows.
THRESHOLDS: dict[str, list[tuple[float, float]]] = {
    "rsrp_dbm": [(-120, 0.0), (-110, 0.15), (-100, 0.45), (-90, 0.7), (-80, 0.9), (-70, 1.0)],
    "rsrq_db": [(-20, 0.0), (-16, 0.3), (-12, 0.6), (-10, 0.85), (-8, 1.0)],
    "sinr_db": [(-5, 0.0), (0, 0.2), (5, 0.45), (13, 0.8), (20, 1.0)],
}
WEIGHTS = {"rsrp_dbm": 0.4, "rsrq_db": 0.2, "sinr_db": 0.4}
RTT_SCORE = [(40, 1.0), (120, 0.9), (250, 0.7), (400, 0.45), (700, 0.2), (1200, 0.0)]
GRADES = [(0.75, "excellent"), (0.55, "good"), (0.35, "fair"), (0.15, "poor"), (0.0, "unusable")]


def interpolate(table: list[tuple[float, float]], x: float) -> float:
    """Piecewise-linear lookup, clamped at both ends."""
    if x <= table[0][0]:
        return table[0][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x <= x1:
            span = x1 - x0
            return y0 if span <= 0 else y0 + (y1 - y0) * (x - x0) / span
    return table[-1][1]


def grade(score: float) -> str:
    for floor, name in GRADES:
        if score >= floor:
            return name
    return "unusable"


def score_signal(signal: Signal) -> tuple[float, list[str]]:
    """0..1 from the modem's own numbers, with the reasons it is not 1."""
    if not signal.connected:
        return 0.0, [signal.error or "the modem is not connected"]
    total = weight = 0.0
    reasons: list[str] = []
    for metric, table in THRESHOLDS.items():
        value = getattr(signal, metric)
        if value is None:
            continue
        s = interpolate(table, float(value))
        total += s * WEIGHTS[metric]
        weight += WEIGHTS[metric]
        if s < 0.45:
            unit = "dBm" if metric.endswith("dbm") else "dB"
            reasons.append(f"{metric[:4].upper()} {value:.0f} {unit}")
    if weight <= 0:
        return 0.0, ["the router reported no signal metrics"]
    return total / weight, reasons


@dataclass
class LinkQuality:
    """One verdict on the link: the modem, the packets, and a number for both."""

    t: float = field(default_factory=time.monotonic)
    signal: Signal = field(default_factory=Signal)
    rtt_ms: float | None = None
    jitter_ms: float | None = None
    loss: float = 0.0
    score: float = 0.0
    grade: str = "unusable"
    reasons: list[str] = field(default_factory=list)

    def age_s(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.t

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "signal"}
        d["signal"] = self.signal.as_dict()
        return d

    def line(self) -> str:
        rtt = "—" if self.rtt_ms is None else f"{self.rtt_ms:.0f} ms"
        loss = f"{self.loss * 100:.0f}%"
        tail = f"  ({', '.join(self.reasons)})" if self.reasons else ""
        return f"{self.score:.2f} {self.grade:<9s} {self.signal.line()}  RTT {rtt}  loss {loss}{tail}"


class Pinger:
    """Round trip to the far end, timed by opening a TCP connection to it.

    A TCP handshake, not ICMP: ping needs a raw socket (root on Linux, blocked
    in most containers) and the thing we actually care about is whether the
    drone's own endpoint is reachable through the operator's network.
    """

    def __init__(self, host: str, port: int = 80, timeout: float = 1.5):
        self.host, self.port, self.timeout = host, int(port), float(timeout)

    def rtt_ms(self) -> float | None:
        t0 = time.monotonic()
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout):
                return (time.monotonic() - t0) * 1000.0
        except OSError:
            return None

    def __repr__(self) -> str:
        return f"<Pinger {self.host}:{self.port}>"


class LinkProbe:
    """Polls a router (and optionally a far end) in the background.

    The flight loop runs at 20 Hz and an HTTP GET to a router takes tens to
    hundreds of milliseconds, so the polling happens on its own thread and the
    loop reads the last verdict. A probe that has stalled is visible as an age,
    which is what the governor's timeout is about.
    """

    def __init__(self, router: LteRouter, target: tuple[str, int] | None = None, period_s: float = 1.0,
                 window: int = 20, timeout: float = 1.5):
        self.router = router
        self.period_s = float(period_s)
        self.pinger = Pinger(target[0], target[1], timeout) if target else None
        self.rtts: deque[float | None] = deque(maxlen=int(window))
        self.last: LinkQuality | None = None
        self.samples = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    def sample(self) -> LinkQuality:
        """One poll. Blocking, and safe to call from a test or a script."""
        signal = self.router.signal()
        rtt = self.pinger.rtt_ms() if self.pinger else None
        if self.pinger:
            self.rtts.append(rtt)
        seen = [r for r in self.rtts if r is not None]
        loss = 1.0 - len(seen) / len(self.rtts) if self.rtts else 0.0
        jitter = statistics.pstdev(seen) if len(seen) > 2 else None

        base, reasons = score_signal(signal)
        score = base
        if self.pinger:
            if rtt is None and loss >= 1.0:
                reasons.append("nothing answered at the far end")
                score = 0.0
            else:
                latency = interpolate(RTT_SCORE, rtt if rtt is not None else 1200.0)
                score *= latency * (1.0 - loss) ** 1.5
                if latency < 0.7 and rtt is not None:
                    reasons.append(f"RTT {rtt:.0f} ms")
                if loss > 0.01:
                    reasons.append(f"{loss * 100:.0f}% loss")
        score = min(1.0, max(0.0, score))
        q = LinkQuality(signal=signal, rtt_ms=rtt, jitter_ms=jitter, loss=loss,
                        score=round(score, 3), grade=grade(score), reasons=reasons)
        self.last = q
        self.samples += 1
        return q

    # ------------------------------------------------------------------
    def start(self) -> LinkProbe:
        if self._thread is not None:
            return self
        self.sample()  # one reading before the flight starts, so nothing flies blind
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="flydrones-lte", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.period_s):
            try:
                self.sample()
            except Exception:  # a router that dies mid-flight must not take the loop with it
                self.last = LinkQuality(signal=Signal(router=self.router.label, error="probe failed"),
                                        score=0.0, grade="unusable", reasons=["the probe itself failed"])

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.router.close()

    close = stop

    # ------------------------------------------------------------------
    def summary(self) -> str:
        if self.last is None:
            return "no reading yet"
        return self.last.line()
