"""Where the notes go: Pure Data, Max/MSP, TidalCycles, SuperDirt, Strudel, a file.

Every sink renders the same :class:`~flydrones.music.events.Frame`, so a target
is a translation and never a second opinion about the music. What differs is
only how each program likes to be spoken to:

===========  ==========================================================
Pure Data    OSC into vanilla ``[netreceive -u -b] -> [oscparse]``, or
             FUDI if you would rather read the messages in a Pd window
Max/MSP      OSC into ``[udpreceive 7400]``
TidalCycles  ``/ctrl`` messages on 6010; the fly becomes ``cF``/``cS``
             values inside patterns you write
SuperDirt    ``/dirt/play`` on 57120 — the compositor *is* the pattern
             engine, and no Haskell is involved
Strudel      a browser cannot take UDP, so: JSON over Server-Sent Events
===========  ==========================================================

Timing. A frame is composed at the end of a control tick, and its notes carry
spike times from inside that tick, so everything is sent with a latency
(200 ms by default, Tidal's own habit): SuperDirt gets an OSC timetag it can
schedule on, and Pd and Max get the same figure as a plain ``delay_ms`` they
can hand to ``[delay]``. Turn it down to 0 for the lowest latency and the
loosest timing.
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

from .events import Frame
from .osc import UdpSender, encode_bundle, encode_message
from .server import EventServer

DEFAULT_PORTS = {"pd": 9000, "pd_fudi": 3001, "max": 7400, "tidal": 6010, "superdirt": 57120, "strudel": 8765}

# SuperDirt has no idea what a "flylead" is. These are synths that ship with it.
SUPERDIRT_SOUNDS = {"flybass": "supersaw", "flylead": "superpwm", "flyperc": "superhat",
                    "flycrash": "superhoover", "flyloom": "supersquare", "flyair": "supersine"}


class Sink:
    """Render frames somewhere. Never raises at the drone: music is not flight."""

    kind = "sink"

    def frame(self, frame: Frame) -> None:
        raise NotImplementedError

    def set_context(self, context: dict) -> None:
        """Whatever surrounds the music — the station, the show, the log.

        Most targets have nowhere to put it: Pd wants notes, not a programme.
        The ones that can (a browser, a file) override this.
        """

    def describe(self) -> str:
        return self.kind

    def close(self) -> None:
        pass


class OscSink(Sink):
    """The shared shape: /fly/note, /fly/ctrl, /fly/section, /fly/frame."""

    kind = "osc"

    def __init__(self, host: str = "127.0.0.1", port: int = 9000, prefix: str = "/fly", latency: float = 0.2,
                 bundle: bool = True, controls: bool = True):
        self.out = UdpSender(host, port)
        self.prefix = prefix.rstrip("/")
        self.latency = float(latency)
        self.bundle = bool(bundle)
        self.controls = bool(controls)

    def messages(self, frame: Frame) -> list[tuple[str, list]]:
        p = self.prefix
        msgs: list[tuple[str, list]] = [(f"{p}/frame", [float(frame.t), float(frame.cycle), float(frame.cps), len(frame.notes)])]
        if frame.section:
            msgs.append((f"{p}/section", [frame.section]))
        for n in frame.notes:
            delay_ms = max(0.0, (n.t - frame.t + self.latency) * 1000.0)
            msgs.append((f"{p}/note", [n.voice, n.sound, int(n.note), float(n.velocity), float(n.duration),
                                       float(n.pan), int(n.channel), int(n.orbit), float(n.rate_hz), float(delay_ms)]))
        if self.controls:
            msgs += [(f"{p}/ctrl", [c.name, float(c.value)]) for c in frame.controls]
        return msgs

    def frame(self, frame: Frame) -> None:
        msgs = self.messages(frame)
        if self.bundle:
            self.out.send_bundle(msgs)
        else:
            for address, args in msgs:
                self.out.send(address, *args)

    def describe(self) -> str:
        return f"{self.kind} -> osc {self.out.host}:{self.out.port} {self.prefix}/*"

    def close(self) -> None:
        self.out.close()


class PureDataSink(OscSink):
    kind = "puredata"


class MaxSink(OscSink):
    kind = "maxmsp"


class FudiSink(Sink):
    """Pure Data's native protocol: ``note flylead 64 0.8 ... ;`` down a socket.

    Vanilla ``[netreceive -u 3001]`` reads this with no externals and no
    ``[oscparse]``, and you can see every message in the Pd console, which is
    the fastest way to find out whether the fly or the patch is at fault.
    """

    kind = "puredata-fudi"

    def __init__(self, host: str = "127.0.0.1", port: int = 3001, latency: float = 0.2, controls: bool = True):
        self.host, self.port = host, int(port)
        self.latency = float(latency)
        self.controls = bool(controls)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sent = 0
        self.dropped = 0

    @staticmethod
    def _atom(x) -> str:
        if isinstance(x, bool):
            return "1" if x else "0"
        if isinstance(x, float):
            return f"{x:.5g}"
        s = str(x)
        for bad, good in ((";", r"\;"), (",", "\\,"), ("$", "\\$"), (" ", "_")):
            s = s.replace(bad, good)
        return s

    def lines(self, frame: Frame) -> list[str]:
        out = [f"frame {frame.t:.4f} {frame.cycle:.4f} {frame.cps:.4f} {len(frame.notes)}"]
        if frame.section:
            out.append(f"section {frame.section}")
        for n in frame.notes:
            delay_ms = max(0.0, (n.t - frame.t + self.latency) * 1000.0)
            out.append("note " + " ".join(self._atom(v) for v in (n.voice, n.sound, n.note, n.velocity, n.duration,
                                                                  n.pan, n.channel, n.orbit, n.rate_hz, delay_ms)))
        if self.controls:
            out += [f"ctrl {self._atom(c.name)} {c.value:.5g}" for c in frame.controls]
        return out

    def frame(self, frame: Frame) -> None:
        # One datagram per message. Pd's UDP netreceive evaluates only the first
        # message in a packet and silently drops the rest — measured against
        # `pd -nogui`, which is how this is a comment and not a bug report.
        for line in self.lines(frame):
            try:
                self.sock.sendto((line + ";\n").encode("utf-8"), (self.host, self.port))
                self.sent += 1
            except OSError:
                self.dropped += 1

    def describe(self) -> str:
        return f"{self.kind} -> fudi {self.host}:{self.port}"

    def close(self) -> None:
        self.sock.close()


class TidalSink(Sink):
    """TidalCycles' control bus: ``/ctrl`` pairs that patterns read with ``cF``.

    Tidal keeps the cycle and you keep writing the patterns; the fly is a set
    of live values inside them. Voices arrive as ``<voice>_n`` (its last note)
    and ``<voice>_g`` (1 on the tick it fired), controls under their own names.
    """

    kind = "tidalcycles"

    def __init__(self, host: str = "127.0.0.1", port: int = 6010, bundle: bool = True):
        self.out = UdpSender(host, port)
        self.bundle = bool(bundle)

    def messages(self, frame: Frame) -> list[tuple[str, list]]:
        msgs: list[tuple[str, list]] = [("/ctrl", [c.name, float(c.value)]) for c in frame.controls]
        msgs.append(("/ctrl", ["section", frame.section]))
        gates = {n.voice: n for n in frame.notes}
        for voice, n in gates.items():
            msgs.append(("/ctrl", [f"{voice}_n", float(n.note)]))
            msgs.append(("/ctrl", [f"{voice}_v", float(n.velocity)]))
            msgs.append(("/ctrl", [f"{voice}_g", 1.0]))
        return msgs

    def frame(self, frame: Frame) -> None:
        msgs = self.messages(frame)
        if self.bundle:
            self.out.send_bundle(msgs)
        else:
            for address, args in msgs:
                self.out.send(address, *args)

    def describe(self) -> str:
        return f"{self.kind} -> /ctrl on {self.out.host}:{self.out.port}"

    def close(self) -> None:
        self.out.close()


class SuperDirtSink(Sink):
    """Straight to SuperDirt: ``/dirt/play``, one timestamped bundle per note.

    This is the target with no other program in it — the compositor takes the
    part Tidal would play, which is the point when the pattern is a fly.
    """

    kind = "superdirt"

    def __init__(self, host: str = "127.0.0.1", port: int = 57120, latency: float = 0.2,
                 sounds: dict[str, str] | None = None, extra: dict | None = None):
        self.out = UdpSender(host, port)
        self.latency = float(latency)
        self.sounds = dict(SUPERDIRT_SOUNDS if sounds is None else sounds)
        self.extra = dict(extra or {})

    def params(self, note, frame: Frame) -> list:
        args: list = [
            "s", self.sounds.get(note.sound, note.sound),
            "note", float(note.note - 60),  # SuperDirt counts semitones from middle C
            "gain", float(note.velocity),
            "pan", float(note.pan),
            "orbit", int(note.orbit),
            "sustain", float(note.duration),
            "delta", float(note.duration),
            "cps", float(frame.cps),
            "cycle", float(frame.cycle),
        ]
        for k, v in self.extra.items():
            args += [str(k), v]
        return args

    def frame(self, frame: Frame) -> None:
        now = time.time()
        for n in frame.notes:
            when = now + self.latency + max(0.0, n.t - frame.t)
            self.out.send_bytes(encode_bundle([encode_message("/dirt/play", self.params(n, frame))], when))

    def describe(self) -> str:
        return f"{self.kind} -> /dirt/play on {self.out.host}:{self.out.port}"

    def close(self) -> None:
        self.out.close()


class StrudelSink(Sink):
    """A browser: one JSON frame per tick over Server-Sent Events."""

    kind = "strudel"

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, server: EventServer | None = None):
        self.server = (server or EventServer(host, port)).start()
        self.context: dict = {}

    def set_context(self, context: dict) -> None:
        self.context = dict(context or {})

    @staticmethod
    def payload(frame: Frame) -> dict:
        return {"t": round(frame.t, 4), "cycle": round(frame.cycle, 4), "cps": round(frame.cps, 4),
                "section": frame.section,
                "notes": [{k: v for k, v in n.as_dict().items() if k != "kind"} for n in frame.notes],
                "controls": {c.name: c.value for c in frame.controls}}

    def frame(self, frame: Frame) -> None:
        self.server.publish({**self.payload(frame), **self.context, "listeners": self.server.clients})

    def describe(self) -> str:
        return f"{self.kind} -> open {self.server.url} ({self.server.clients} listening)"

    def close(self) -> None:
        self.server.close()


class JsonlSink(Sink):
    """One JSON frame per line: the score, kept. Replay it with ``--replay``."""

    kind = "jsonl"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")
        self.frames = 0

    def frame(self, frame: Frame) -> None:
        self.file.write(json.dumps(StrudelSink.payload(frame), separators=(",", ":")) + "\n")
        self.frames += 1

    def describe(self) -> str:
        return f"{self.kind} -> {self.path}"

    def close(self) -> None:
        self.file.close()


class NullSink(Sink):
    """Nowhere. For a run whose output is the rendered file, not a live target."""

    kind = "none"

    def frame(self, frame) -> None:
        pass


class PrintSink(Sink):
    """A meter in the terminal, once a second, so you can tell it is alive."""

    kind = "print"

    def __init__(self, every_s: float = 1.0, out=None):
        self.every = float(every_s)
        self.out = out
        self._next = 0.0
        self._notes = 0
        self.context: dict = {}

    def set_context(self, context: dict) -> None:
        self.context = dict(context or {})

    def frame(self, frame: Frame) -> None:
        self._notes += len(frame.notes)
        if frame.t < self._next:
            return
        self._next = frame.t + self.every
        drive = next((c.value for c in frame.controls if c.name == "drive"), 0.0)
        loom = next((c.value for c in frame.controls if c.name == "loom"), 0.0)
        bar = "█" * int(round(drive * 12)) + "·" * (12 - int(round(drive * 12)))
        show = (self.context.get("show") or {}).get("name")
        line = (f"t={frame.t:6.1f}s  {frame.section:<7s} lift [{bar}] loom {loom:4.2f}  "
                f"cycle {frame.cycle:6.2f}  {self._notes:4d} notes")
        if show:
            memory = (self.context.get("brain") or {}).get("memory", 0.0)
            line += f"  · {show} · memory {memory:4.2f}"
        print(line, file=self.out)

    def describe(self) -> str:
        return f"{self.kind} -> stdout"


class FanOut(Sink):
    """All of the above at once. One flight can play four programs."""

    kind = "fanout"

    def __init__(self, sinks: list[Sink]):
        self.sinks = list(sinks)

    def frame(self, frame: Frame) -> None:
        for s in self.sinks:
            s.frame(frame)

    def set_context(self, context: dict) -> None:
        for s in self.sinks:
            s.set_context(context)

    def describe(self) -> str:
        return "\n".join(f"  {s.describe()}" for s in self.sinks)

    def close(self) -> None:
        for s in self.sinks:
            s.close()


ALIASES = {"puredata": "pd", "pure-data": "pd", "pure_data": "pd", "maxmsp": "max", "max-msp": "max",
           "maxforlive": "max", "tidal": "tidal", "tidalcycles": "tidal", "dirt": "superdirt", "sc": "superdirt",
           "supercollider": "superdirt", "fudi": "pd-fudi", "file": "jsonl", "stdout": "print", "null": "none", "off": "none", "silent": "none"}


def make_sink(spec: str, cfg: dict | None = None) -> Sink:
    """``"pd"``, ``"max:7400"``, ``"strudel:0.0.0.0:8765"``, ``"jsonl:flight.jsonl"`` -> a sink."""
    cfg = dict((cfg or {}).get("music", {}) or {})
    latency = float(cfg.get("latency_s", 0.2))
    name, _, rest = spec.strip().partition(":")
    name = ALIASES.get(name.lower().strip(), name.lower().strip())
    host, port = "127.0.0.1", None
    if rest and name not in ("jsonl", "print"):
        bits = rest.split(":")
        if len(bits) == 1 and bits[0].isdigit():
            port = int(bits[0])
        else:
            host = bits[0] or host
            port = int(bits[1]) if len(bits) > 1 and bits[1] else None

    if name == "pd":
        return PureDataSink(host, port or DEFAULT_PORTS["pd"], latency=latency)
    if name == "pd-fudi":
        return FudiSink(host, port or DEFAULT_PORTS["pd_fudi"], latency=latency)
    if name == "max":
        return MaxSink(host, port or DEFAULT_PORTS["max"], latency=latency)
    if name == "tidal":
        return TidalSink(host, port or DEFAULT_PORTS["tidal"])
    if name == "superdirt":
        return SuperDirtSink(host, port or DEFAULT_PORTS["superdirt"], latency=latency,
                             sounds=(cfg.get("sounds") or {}).get("superdirt"))
    if name == "strudel":
        return StrudelSink(host, port or DEFAULT_PORTS["strudel"])
    if name == "jsonl":
        return JsonlSink(rest or "flight-score.jsonl")
    if name == "print":
        return PrintSink(float(rest) if rest else 1.0)
    if name == "none":
        return NullSink()
    raise ValueError(f"unknown music target {spec!r}; try pd, max, tidal, superdirt, strudel, jsonl:FILE, print or none")


def make_sinks(specs, cfg: dict | None = None) -> FanOut:
    if isinstance(specs, str):
        specs = [s for s in specs.replace(" ", ",").split(",") if s]
    return FanOut([make_sink(s, cfg) for s in specs])


def read_jsonl(path: str | Path):
    """Read a score written by :class:`JsonlSink` back into frames."""
    from .events import ControlEvent, NoteEvent

    frames: list[Frame] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            frames.append(Frame(
                t=float(d.get("t", 0.0)),
                notes=[NoteEvent(**n) for n in d.get("notes", [])],
                controls=[ControlEvent(t=float(d.get("t", 0.0)), name=k, value=float(v))
                          for k, v in (d.get("controls") or {}).items()],
                cycle=float(d.get("cycle", 0.0)), cps=float(d.get("cps", 0.5)), section=d.get("section", ""),
            ))
    return frames
