"""Open Sound Control 1.0, encoder and decoder, in one file with no dependencies.

Every target here speaks OSC: Pure Data through vanilla ``[oscparse]``, Max/MSP
through ``[udpreceive]``, SuperDirt on 57120 and TidalCycles' ``/ctrl`` bus on
6010. Bringing in a library for three hundred bytes of struct packing would
make the music an optional extra of the install; it is not.

Only the types this project sends are implemented (``i f s b`` and the tagless
``T F N``), and ``decode`` exists so the tests can read back what the sinks put
on the wire rather than trusting the encoder against itself.
"""

from __future__ import annotations

import socket
import struct
from datetime import datetime, timezone

# OSC timetags count seconds from 1900-01-01; unix time counts from 1970.
NTP_EPOCH_OFFSET = 2_208_988_800
IMMEDIATELY = b"\x00\x00\x00\x00\x00\x00\x00\x01"


def _pad(data: bytes) -> bytes:
    """OSC pads every chunk to a multiple of four bytes with nulls."""
    return data + b"\x00" * (-len(data) % 4)


def _string(s: str) -> bytes:
    return _pad(s.encode("utf-8") + b"\x00")


def timetag(unix_seconds: float) -> bytes:
    """Absolute OSC timetag for a unix timestamp (``time.time()``)."""
    ntp = unix_seconds + NTP_EPOCH_OFFSET
    seconds = int(ntp)
    fraction = int((ntp - seconds) * 2**32) & 0xFFFFFFFF
    return struct.pack(">II", seconds & 0xFFFFFFFF, fraction)


def encode_message(address: str, args=()) -> bytes:
    """``/fly/note`` + a list of ints, floats, strings, bools, bytes -> one packet."""
    if not address.startswith("/"):
        raise ValueError(f"OSC address must start with '/': {address!r}")
    tags, payload = [], []
    for a in args:
        if isinstance(a, bool):  # before int: bool is an int in Python, not in OSC
            tags.append("T" if a else "F")
        elif a is None:
            tags.append("N")
        elif isinstance(a, int):
            tags.append("i")
            payload.append(struct.pack(">i", max(-2**31, min(2**31 - 1, a))))
        elif isinstance(a, float):
            tags.append("f")
            payload.append(struct.pack(">f", a))
        elif isinstance(a, str):
            tags.append("s")
            payload.append(_string(a))
        elif isinstance(a, (bytes, bytearray)):
            tags.append("b")
            payload.append(struct.pack(">i", len(a)) + _pad(bytes(a)))
        else:
            raise TypeError(f"cannot put {type(a).__name__} in an OSC message")
    return _string(address) + _string("," + "".join(tags)) + b"".join(payload)


def encode_bundle(elements, when: float | None = None) -> bytes:
    """Bundle encoded messages so they arrive — and are scheduled — together."""
    head = b"#bundle\x00" + (IMMEDIATELY if when is None else timetag(when))
    return head + b"".join(struct.pack(">i", len(e)) + e for e in elements)


def decode_message(data: bytes) -> tuple[str, list]:
    address, i = _read_string(data, 0)
    if i >= len(data):
        return address, []
    tags, i = _read_string(data, i)
    args: list = []
    for tag in tags[1:]:
        if tag in "ir":
            args.append(struct.unpack_from(">i", data, i)[0])
            i += 4
        elif tag == "f":
            args.append(struct.unpack_from(">f", data, i)[0])
            i += 4
        elif tag == "d":
            args.append(struct.unpack_from(">d", data, i)[0])
            i += 8
        elif tag == "h":
            args.append(struct.unpack_from(">q", data, i)[0])
            i += 8
        elif tag == "s":
            s, i = _read_string(data, i)
            args.append(s)
        elif tag == "b":
            n = struct.unpack_from(">i", data, i)[0]
            i += 4
            args.append(data[i : i + n])
            i += n + (-n % 4)
        elif tag in "TF":
            args.append(tag == "T")
        elif tag == "N":
            args.append(None)
        else:
            raise ValueError(f"unsupported OSC type tag {tag!r}")
    return address, args


def decode(data: bytes) -> list[tuple[float | None, str, list]]:
    """Flatten a packet into ``(when, address, args)``; ``when`` is None for 'now'."""
    if data[:8] == b"#bundle\x00":
        seconds, fraction = struct.unpack_from(">II", data, 8)
        when = None if (seconds, fraction) == (0, 1) else seconds - NTP_EPOCH_OFFSET + fraction / 2**32
        out, i = [], 16
        while i < len(data):
            n = struct.unpack_from(">i", data, i)[0]
            i += 4
            for _when, addr, args in decode(data[i : i + n]):
                out.append((_when if _when is not None else when, addr, args))
            i += n
        return out
    addr, args = decode_message(data)
    return [(None, addr, args)]


def _read_string(data: bytes, i: int) -> tuple[str, int]:
    end = data.index(b"\x00", i)
    return data[i:end].decode("utf-8", "replace"), i + len(_pad(data[i:end] + b"\x00"))


def format_time(unix_seconds: float) -> str:
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]


class UdpSender:
    """Fire-and-forget UDP. A missed note is better than a stalled brain."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.host, self.port = host, int(port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sent = 0
        self.dropped = 0

    def send_bytes(self, packet: bytes) -> None:
        try:
            self.sock.sendto(packet, (self.host, self.port))
            self.sent += 1
        except OSError:  # nobody listening, buffer full, interface gone
            self.dropped += 1

    def send(self, address: str, *args) -> None:
        self.send_bytes(encode_message(address, args))

    def send_bundle(self, messages, when: float | None = None) -> None:
        if messages:
            self.send_bytes(encode_bundle([encode_message(a, g) for a, g in messages], when))

    def close(self) -> None:
        self.sock.close()

    def __repr__(self) -> str:
        return f"<UdpSender {self.host}:{self.port} sent={self.sent} dropped={self.dropped}>"
