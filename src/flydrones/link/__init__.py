"""The link the drone flies on: an Orange LTE router, and a probe that watches it.

    from flydrones.link import make_router, LinkProbe

    probe = LinkProbe(make_router("orange-airbox"), target=("8.8.8.8", 53)).start()
    probe.last.line()

See ``docs/LTE.md``. Reading a router needs nothing beyond the standard library,
and ``make_router("mock")`` runs the whole chain without one.
"""

from .probe import GRADES, RTT_SCORE, THRESHOLDS, WEIGHTS, LinkProbe, LinkQuality, Pinger, grade, interpolate, score_signal
from .router import (
    ORANGE_ROUTERS,
    HuaweiHilinkRouter,
    LinkError,
    LteRouter,
    MockRouter,
    Signal,
    ZteRouter,
    make_router,
    parse_huawei_plmn,
    parse_huawei_signal,
    parse_huawei_status,
    parse_huawei_traffic,
    parse_zte,
)

__all__ = [
    "GRADES", "ORANGE_ROUTERS", "RTT_SCORE", "THRESHOLDS", "WEIGHTS",
    "HuaweiHilinkRouter", "LinkError", "LinkProbe", "LinkQuality", "LteRouter", "MockRouter", "Pinger",
    "Signal", "ZteRouter", "grade", "interpolate", "make_router", "parse_huawei_plmn", "parse_huawei_signal",
    "parse_huawei_status", "parse_huawei_traffic", "parse_zte", "score_signal",
]
