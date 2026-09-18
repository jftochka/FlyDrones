"""Orange LTE routers, read over the LAN.

Flying a drone on a mobile link means the link is part of the airframe: when it
goes, the drone is on its own. So FlyDrones reads the modem directly rather
than guessing from a ping, and the numbers it reads are the ones a cell
engineer would look at — RSRP, RSRQ, SINR, band, cell.

Orange ships two families of LTE box, both of which answer on the LAN with no
extra software:

* **Huawei HiLink** — Airbox, Flybox, Home 4G/4G+ (B310s-22, B525s-23, B818,
  E5576/E5783 pocket boxes). A small XML API on ``http://192.168.8.1/api/...``.
* **ZTE** — Flybox MF283/MF286 and relatives, a JSON API on
  ``http://192.168.0.1/goform/goform_get_cmd_process``.

Both are read-only here. Nothing in FlyDrones ever asks a router to change
anything: the worst this code can do to your connection is fetch a status page
once a second.

    from flydrones.link import make_router
    router = make_router("orange-airbox")      # or "auto", "huawei", "zte", "mock"
    print(router.signal())

The parsers are pure functions over the bodies those endpoints return, so the
tests do not need a router (``tests/test_link.py`` carries real-shaped samples),
and ``MockRouter`` drives the whole chain — probe, scoring, safety — with a
scripted drive out of coverage.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from http.cookiejar import CookieJar

# Orange's own names for the boxes, and where each one answers by default.
ORANGE_ROUTERS: dict[str, tuple[str, str, str]] = {
    "orange-airbox": ("huawei", "192.168.8.1", "Orange Airbox 4G (Huawei E5783/E5576 pocket router)"),
    "orange-flybox": ("huawei", "192.168.8.1", "Orange Flybox 4G (Huawei B310s-22 / B525s-23)"),
    "orange-home-4g": ("huawei", "192.168.8.1", "Orange Home 4G+ (Huawei B818 / B535)"),
    "orange-flybox-zte": ("zte", "192.168.0.1", "Orange Flybox (ZTE MF283 / MF286)"),
}

# Huawei's numbers for what the modem is doing. 901 is the one that matters.
HUAWEI_CONNECTION = {"900": "connecting", "901": "connected", "902": "disconnected",
                     "903": "disconnecting", "904": "failed"}
HUAWEI_NETWORK = {"19": "LTE", "101": "LTE", "1011": "LTE+", "41": "WCDMA", "46": "HSPA+",
                  "9": "HSPA+", "3": "EDGE", "0": "none"}


class LinkError(RuntimeError):
    """The router could not be read. Never raised into the flight loop."""


@dataclass
class Signal:
    """One reading of the modem, in the units the standards use."""

    t: float = field(default_factory=time.monotonic)
    router: str = ""
    connected: bool = False
    rsrp_dbm: float | None = None  # reference signal received power: coverage
    rsrq_db: float | None = None  # reference signal received quality: interference and load
    sinr_db: float | None = None  # signal to interference + noise: what the throughput follows
    rssi_dbm: float | None = None
    band: str | None = None  # "B3", "B7", "B20" — Orange's LTE bands in Europe
    cell_id: str | None = None
    pci: str | None = None
    operator: str | None = None  # "Orange F", "Orange Polska"
    plmn: str | None = None  # 20801, 26003 ...
    network: str | None = None  # LTE, LTE+, WCDMA
    uptime_s: float | None = None
    up_kbps: float | None = None
    down_kbps: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.connected

    def as_dict(self) -> dict:
        return asdict(self)

    def line(self) -> str:
        if self.error:
            return f"{self.router}: {self.error}"
        bits = [f"{self.operator or self.plmn or self.router}", self.network or "?",
                f"{self.band or '?'}", f"RSRP {self._n(self.rsrp_dbm)} dBm",
                f"RSRQ {self._n(self.rsrq_db)} dB", f"SINR {self._n(self.sinr_db)} dB"]
        if self.cell_id:
            bits.append(f"cell {self.cell_id}")
        return "  ".join(bits)

    @staticmethod
    def _n(v: float | None) -> str:
        if v is None:
            return "—"
        text = f"{v:.0f}"
        return "0" if text == "-0" else text


def _number(text: str | None) -> float | None:
    """``"-95dBm"`` / ``">=-20dB"`` / ``"12"`` -> a float, or None for a blank field."""
    if text is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(text))
    if not m:
        return None
    v = float(m.group())
    return v if math.isfinite(v) else None


# ------------------------------------------------------------------ parsing
def parse_huawei_signal(xml: str) -> dict:
    """``/api/device/signal`` -> the fields we care about."""
    tags = dict(re.findall(r"<(\w+)>([^<]*)</\1>", xml or ""))
    band = tags.get("band") or tags.get("lteband")
    return {
        "rsrp_dbm": _number(tags.get("rsrp")),
        "rsrq_db": _number(tags.get("rsrq")),
        "sinr_db": _number(tags.get("sinr")),
        "rssi_dbm": _number(tags.get("rssi")),
        "cell_id": tags.get("cell_id") or None,
        "pci": tags.get("pci") or None,
        "band": f"B{band}" if band and band.isdigit() else (band or None),
    }


def parse_huawei_status(xml: str) -> dict:
    tags = dict(re.findall(r"<(\w+)>([^<]*)</\1>", xml or ""))
    status = tags.get("ConnectionStatus", "")
    network = tags.get("CurrentNetworkTypeEx") or tags.get("CurrentNetworkType") or ""
    return {
        "connected": HUAWEI_CONNECTION.get(status) == "connected",
        "network": HUAWEI_NETWORK.get(network, None),
    }


def parse_huawei_plmn(xml: str) -> dict:
    tags = dict(re.findall(r"<(\w+)>([^<]*)</\1>", xml or ""))
    return {"operator": tags.get("FullName") or tags.get("ShortName") or None,
            "plmn": tags.get("Numeric") or None}


def parse_huawei_traffic(xml: str) -> dict:
    tags = dict(re.findall(r"<(\w+)>([^<]*)</\1>", xml or ""))
    kbps = lambda key: (None if _number(tags.get(key)) is None else _number(tags.get(key)) * 8 / 1000.0)  # noqa: E731
    return {"up_kbps": kbps("CurrentUploadRate"), "down_kbps": kbps("CurrentDownloadRate"),
            "uptime_s": _number(tags.get("CurrentConnectTime"))}


ZTE_FIELDS = ("lte_rsrp", "lte_rsrq", "lte_snr", "Z5g_snr", "lte_band", "lte_pci", "cell_id", "rssi",
              "network_type", "ppp_status", "network_provider", "realtime_tx_thrpt", "realtime_rx_thrpt",
              "realtime_time")


def parse_zte(payload: str | dict) -> dict:
    """``/goform/goform_get_cmd_process?...&multi_data=1`` -> the same fields."""
    data = payload if isinstance(payload, dict) else json.loads(payload or "{}")
    band = data.get("lte_band")
    status = str(data.get("ppp_status", "")).lower()
    return {
        "rsrp_dbm": _number(data.get("lte_rsrp")),
        "rsrq_db": _number(data.get("lte_rsrq")),
        "sinr_db": _number(data.get("lte_snr") or data.get("Z5g_snr")),
        "rssi_dbm": _number(data.get("rssi")),
        "cell_id": str(data.get("cell_id")) if data.get("cell_id") else None,
        "pci": str(data.get("lte_pci")) if data.get("lte_pci") else None,
        "band": f"B{band}" if band and str(band).isdigit() else (band or None),
        "operator": data.get("network_provider") or None,
        "network": (data.get("network_type") or None),
        "connected": status.startswith("ppp_connected") or status == "connected",
        "up_kbps": (None if _number(data.get("realtime_tx_thrpt")) is None
                    else _number(data.get("realtime_tx_thrpt")) * 8 / 1000.0),
        "down_kbps": (None if _number(data.get("realtime_rx_thrpt")) is None
                      else _number(data.get("realtime_rx_thrpt")) * 8 / 1000.0),
        "uptime_s": _number(data.get("realtime_time")),
    }


# ------------------------------------------------------------------ routers
class LteRouter(ABC):
    """Read-only access to one LTE box."""

    kind = "router"

    def __init__(self, host: str = "", label: str = "", timeout: float = 2.0):
        self.host = host
        self.label = label or f"{self.kind}@{host}"
        self.timeout = float(timeout)

    @abstractmethod
    def read(self) -> dict:
        """Fields for :class:`Signal`, or raise :class:`LinkError`."""

    def signal(self) -> Signal:
        """Never raises: a router that cannot be read is a signal with an error."""
        try:
            return Signal(router=self.label, **self.read())
        except LinkError as e:
            return Signal(router=self.label, error=str(e))
        except Exception as e:  # a router firmware we have not met
            return Signal(router=self.label, error=f"{type(e).__name__}: {e}")

    def close(self) -> None:  # noqa: B027 - optional hook, like Drone.close
        pass


class HttpRouter(LteRouter):
    """Shared HTTP plumbing: one opener with a cookie jar, short timeouts."""

    def __init__(self, host: str, label: str = "", timeout: float = 2.0, headers: dict | None = None):
        super().__init__(host, label, timeout)
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
        self.headers = {"User-Agent": "flydrones-link", **(headers or {})}

    def get(self, path: str) -> str:
        url = f"http://{self.host}{path}"
        try:
            req = urllib.request.Request(url, headers={**self.headers, "Referer": f"http://{self.host}/"})
            with self.opener.open(req, timeout=self.timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            raise LinkError(f"{url} -> HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LinkError(f"{url} unreachable ({getattr(e, 'reason', e)})") from e


class HuaweiHilinkRouter(HttpRouter):
    """Orange Airbox / Flybox / Home 4G+ and every other HiLink box."""

    kind = "huawei"

    def session(self) -> None:
        """HiLink hands out a session cookie; some firmwares refuse reads without one."""
        try:
            self.get("/api/webserver/SesTokInfo")
        except LinkError:
            pass  # older firmware serves the status endpoints unauthenticated

    def read(self) -> dict:
        self.session()
        out: dict = {"connected": False}
        out.update(parse_huawei_signal(self.get("/api/device/signal")))
        out.update(parse_huawei_status(self.get("/api/monitoring/status")))
        for path, parse in (("/api/net/current-plmn", parse_huawei_plmn),
                            ("/api/monitoring/traffic-statistics", parse_huawei_traffic)):
            try:
                out.update(parse(self.get(path)))
            except LinkError:
                pass  # the extras are nice to have; the signal is the point
        return out


class ZteRouter(HttpRouter):
    """Orange Flybox and other ZTE boxes: one JSON endpoint, many fields."""

    kind = "zte"

    def read(self) -> dict:
        query = ",".join(ZTE_FIELDS)
        body = self.get(f"/goform/goform_get_cmd_process?isTest=false&multi_data=1&cmd={query}")
        return parse_zte(body)


class MockRouter(LteRouter):
    """A box that is not there: a scripted drive from good coverage to none.

    The point of it is that the whole chain — prober, scoring, the safety rules
    that hold and then land — can be exercised, demonstrated and tested without
    an Orange router, the same way MiniFly stands in for the connectome.
    """

    kind = "mock"

    def __init__(self, host: str = "mock", label: str = "mock (a drive out of coverage)",
                 seconds_to_edge: float = 60.0, start: float | None = None, band: str = "B3"):
        super().__init__(host, label)
        self.seconds_to_edge = float(seconds_to_edge)
        self.t0 = time.monotonic() if start is None else start
        self.band = band
        self.now = None  # tests set this to step time by hand

    def elapsed(self) -> float:
        return (self.now if self.now is not None else time.monotonic()) - self.t0

    def read(self) -> dict:
        k = min(1.0, max(0.0, self.elapsed() / max(1e-6, self.seconds_to_edge)))
        wobble = math.sin(self.elapsed() * 1.7) * 1.5
        rsrp = -72.0 - 48.0 * k + wobble
        return {
            "rsrp_dbm": round(rsrp, 1),
            "rsrq_db": round(-7.0 - 12.0 * k + wobble * 0.3, 1),
            "sinr_db": round(22.0 - 26.0 * k + wobble * 0.5, 1),
            "rssi_dbm": round(rsrp + 18.0, 1),
            "band": self.band,
            "cell_id": "20801-4711",
            "pci": "231",
            "operator": "Orange",
            "plmn": "20801",
            "network": "LTE+" if k < 0.4 else "LTE",
            "connected": k < 0.995,
            "up_kbps": round(max(0.0, 9000 * (1 - k)), 1),
            "down_kbps": round(max(0.0, 42000 * (1 - k)), 1),
            "uptime_s": round(self.elapsed(), 1),
        }


def make_router(kind: str = "auto", host: str | None = None, timeout: float = 2.0, **kw) -> LteRouter:
    """``"orange-airbox"``, ``"huawei"``, ``"zte"``, ``"mock"`` or ``"auto"``."""
    key = (kind or "auto").strip().lower()
    label = ""
    if key in ORANGE_ROUTERS:
        key, default_host, label = ORANGE_ROUTERS[key]
        host = host or default_host
    if key == "mock":
        return MockRouter(**kw)
    if key == "huawei":
        return HuaweiHilinkRouter(host or "192.168.8.1", label, timeout)
    if key == "zte":
        return ZteRouter(host or "192.168.0.1", label, timeout)
    if key == "auto":
        tried = []
        for candidate in (HuaweiHilinkRouter(host or "192.168.8.1", "", timeout),
                          ZteRouter(host or "192.168.0.1", "", timeout)):
            if candidate.signal().error is None:
                return candidate
            tried.append(f"{candidate.host} ({candidate.kind})")
        raise LinkError("no LTE router answered on " + " or ".join(dict.fromkeys(tried))
                        + ". Pass --link-host, or --link mock to try it without one.")
    raise ValueError(f"unknown router {kind!r}; try one of: "
                     + ", ".join(["auto", "huawei", "zte", "mock", *ORANGE_ROUTERS]))
