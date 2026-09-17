"""FlyPV as a drone: a real quadcopter simulator on the other end of a pipe.

``SimDrone`` next door is a kinematic model. It moves because it was told to:
a first-order lag toward a commanded velocity, a sphere for a body, and a
camera that draws a striped room. That is the right thing for showing the loop
works, and it is the wrong thing for believing anything about *flying*, because
everything a flight controller is judged on has been left out of it — there is
no rate loop, no prop, no inflow, no battery sagging under the current the
climb is drawing.

`FlyPV <https://github.com/jftochka/FlyPV>`_ is an FPV simulator with all of
that: a 1 kHz rate-mode flight controller with real PID and filters, thrust
that depends on the air already moving through the disc, a gyro with noise and
prop vibration in it, and five airframes spanning a hundred to one in inertia.
It is TypeScript, and it is headless — its physics has no browser in it — so
this backend runs it as a child process and speaks one line of JSON per step.

What that buys, beyond honest physics:

* **The brain's flight comes back as a FlyPV recording.** Blackbox logs there
  store inputs and replay them exactly, so a flight flown by a connectome can
  be opened in the simulator's own browser UI, scrubbed, traced, and compared
  against a human's on the same airframe. ``recording()``, and
  ``flydrones fly --drone flypv --flypv-record flight.json``.
* **Lockstep.** Nothing here sleeps and nothing there reads a clock: the step
  is asked for and the answer comes back. A brain that thinks for a second per
  tick flies the same flight as one that runs faster than life, which is not
  true of any real drone and was not true of ``SimDrone`` either.
* **A camera that is really looking at the world.** Ray cast per pixel against
  the same geometry the physics collides with, at fly-eye resolution.

What it costs: Node.js, a checkout of FlyPV, and about a millisecond per
simulated step. Point this at one with ``--flypv-repo``, ``$FLYPV_REPO``, or a
sibling directory of this repository.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..motor.command import FlightCommand
from ..safety import Telemetry
from .base import Drone
from .sim import Box, Room

#: Protocol the bridge in FlyPV's ``src/core/sim/bridge.ts`` speaks.
PROTOCOL = 1

#: Where to look for a FlyPV checkout when nobody says.
SEARCH = ("FlyPV", "flypv", "../FlyPV", "../flypv")


class FlyPVError(RuntimeError):
    """The bridge refused, died, or was never there."""


def find_repo(explicit: str | None = None) -> Path:
    """Locate a FlyPV checkout: the argument, then ``$FLYPV_REPO``, then siblings."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("FLYPV_REPO"):
        candidates.append(Path(os.environ["FLYPV_REPO"]).expanduser())
    here = Path(__file__).resolve().parents[3]  # the repository root
    for name in SEARCH:
        candidates.append((here / name).resolve())
        candidates.append((Path.cwd() / name).resolve())
    for path in candidates:
        if (path / "tools" / "bridge.ts").is_file():
            return path
    raise FlyPVError(
        "no FlyPV checkout found. Clone https://github.com/jftochka/FlyPV next to this "
        "repository, or pass --flypv-repo /path/to/FlyPV (or set $FLYPV_REPO). "
        f"Looked in: {', '.join(str(c) for c in candidates[:6])}"
    )


def decode_frame(camera: dict) -> np.ndarray:
    """A base64url luminance frame from the bridge as a BGR uint8 image.

    Grey repeated across the three channels rather than converted: the retina
    turns it straight back into luminance, and anything else here would be a
    colour the camera never had.
    """
    text = camera["pixels"]
    raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    grey = np.frombuffer(raw, dtype=np.uint8)
    h, w = int(camera["height"]), int(camera["width"])
    if grey.size != w * h:
        raise FlyPVError(f"bridge sent {grey.size} pixels for a {w}x{h} frame")
    return np.repeat(grey.reshape(h, w, 1), 3, axis=2)


def room_from_plan(plan: dict, floor: float = 8.0, ceiling: float = 24.0) -> Room:
    """FlyPV's world seen from above, as the ``Room`` the dashboard draws.

    Sized to what is *near*, not to the world's own bounds and not to the
    furthest thing in it. The warehouse is a building in the middle of a
    hundred and twenty metres of map and the valley is a kilometre of trees,
    while the safety governor keeps the quad inside a geofence of a few metres:
    a top view drawn at world scale is a dot in an empty square. So the size
    comes from the bulk of the footprints (a high percentile, so one distant
    container does not set the scale), floored so an empty field does not
    collapse to a point and capped so a flight of three metres is still
    something you can see.
    """
    shapes = plan.get("footprints", [])
    reaches = sorted(max(abs(f["east"]) + f["halfEast"], abs(f["north"]) + f["halfNorth"]) for f in shapes)
    bulk = reaches[int(len(reaches) * 0.8)] if reaches else floor / 2
    size = max(floor, min(ceiling, 2 * bulk * 1.1, 2 * float(plan.get("bounds", ceiling))))
    boxes = [
        Box(
            (f["east"] - f["halfEast"], f["north"] - f["halfNorth"], 0.0),
            (f["east"] + f["halfEast"], f["north"] + f["halfNorth"], f["top"]),
            name="",
        )
        for f in shapes
        # Anything outside the drawn square would be clipped to its edge, which
        # reads as a wall that is not there.
        if abs(f["east"]) - f["halfEast"] < size / 2 and abs(f["north"]) - f["halfNorth"] < size / 2
    ]
    return Room(size_x=size, size_y=size, height=float(plan.get("bounds", 10)), boxes=boxes)


class FlyPVDrone(Drone):
    """A quadcopter flown inside FlyPV, over a JSON-lines pipe to Node."""

    name = "flypv"
    has_camera = True

    def __init__(
        self,
        repo: str | None = None,
        world: str = "warehouse",
        airframe: str = "cinewhoop3",
        rates: str = "cinematic",
        seed: int = 0xF1CE,
        camera: tuple[int, int] | None = (96, 72),
        fov_deg: float = 110.0,
        tilt_deg: float = 25.0,
        wind: tuple[float, float, float] = (0.0, 0.0, 0.0),
        turbulence: float = 0.0,
        takeoff_height_m: float = 1.0,
        limits: dict | None = None,
        command: list[str] | None = None,
        quiet: bool = False,
    ):
        # Anything that speaks the protocol on stdin and stdout will do, which
        # is what makes this testable without a Node toolchain — and leaves the
        # door open for a built bundle, or a bridge on the end of a socket
        # wrapper, without changing anything here.
        self.command = command or [shutil.which("npx") or "npx", "vite-node", "tools/bridge.ts"]
        self.repo = Path.cwd() if command else find_repo(repo)
        self.config = {
            "world": world,
            "airframe": airframe,
            "rates": rates,
            "seed": int(seed),
            "wind": {"x": wind[0], "y": wind[1], "z": wind[2]},
            "turbulence": float(turbulence),
            "takeoffHeight": float(takeoff_height_m),
            "camera": None
            if camera is None
            else {"width": int(camera[0]), "height": int(camera[1]), "fovDegrees": float(fov_deg), "tilt": float(tilt_deg)},
        }
        # The limits say what a command of 1 means. The governor in
        # ``safety.py`` caps the commands themselves, so these are the outer
        # rail: what the brain could ask for if the governor allowed it.
        if limits:
            self.config["limits"] = limits
        self.has_camera = camera is not None
        self._quiet = quiet
        self._proc: subprocess.Popen[str] | None = None
        self._tel: dict = {}
        self._frame: np.ndarray | None = None
        self._cmd = FlightCommand()
        self._t = 0.0
        self.crashes = 0
        self.hello: dict = {}
        self.plan: dict = {}
        #: The world in plan, in the shape the dashboard's top view draws.
        self.room: Room | None = None

    # ---------------------------------------------------------------- process
    def connect(self) -> None:
        if self._proc is not None:
            return
        command = self.command
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - the command is ours, the path is the user's
                command,
                cwd=self.repo,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL if self._quiet else sys.stderr,
                text=True,
                bufsize=1,
            )
        except OSError as exc:  # no node, no npx
            raise FlyPVError(f"could not start the FlyPV bridge ({' '.join(command)}): {exc}") from exc

        self.hello = self._ask({"op": "hello"})
        if self.hello.get("protocol") != PROTOCOL:
            raise FlyPVError(
                f"FlyPV bridge speaks protocol {self.hello.get('protocol')}, this backend speaks {PROTOCOL}. "
                "Update one of the two checkouts."
            )
        for key, name in (("worlds", self.config["world"]), ("airframes", self.config["airframe"]), ("rates", self.config["rates"])):
            known = self.hello.get(key, [])
            if known and name not in known:
                raise FlyPVError(f"FlyPV has no {key[:-1]} called {name!r}. It has: {', '.join(known)}")
        reset = self._ask({"op": "reset", "config": self.config})
        self._tel = reset["telemetry"]
        self.plan = reset.get("world") or self.hello.get("world") or {}
        self.room = room_from_plan(self.plan) if self.plan else None

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()

    def _ask(self, request: dict) -> dict:
        """One request, one response, and no clock.

        Blocking on purpose: the whole point of a lockstep bridge is that the
        step takes as long as it takes. There is nothing to time out *to* — a
        bridge that has died closes the pipe, and an empty line is how that
        arrives here.
        """
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise FlyPVError("the FlyPV bridge is not running; call connect() first")
        try:
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        except BrokenPipeError as exc:
            raise FlyPVError("the FlyPV bridge closed the pipe") from exc
        if not line:
            raise FlyPVError(f"the FlyPV bridge stopped (exit {proc.poll()}). Its own errors are on stderr above.")
        response = json.loads(line)
        if not response.get("ok"):
            raise FlyPVError(f"FlyPV refused {request['op']}: {response.get('error')}")
        return response

    # ---------------------------------------------------------------- api
    def takeoff(self) -> None:
        self._tel = self._ask({"op": "takeoff"})["telemetry"]

    def land(self) -> None:
        self._tel = self._ask({"op": "land"})["telemetry"]

    def emergency_stop(self) -> None:
        """Cut the motors. The quad falls, which is what an emergency stop is."""
        self._tel = self._ask({"op": "stop"})["telemetry"]

    def send(self, cmd: FlightCommand) -> None:
        self._cmd = cmd

    def step(self, dt: float) -> None:
        """Advance the simulation by ``dt`` with the command last sent.

        The same shape as ``SimDrone.step``, so ``run_sim`` drives either. What
        happens on the other side is not the same shape at all: FlyPV runs a
        1 kHz fixed step and closes its own velocity loop at 50 Hz inside this
        one call, so a client ticking at twenty hertz still gets a flight
        controller running at a thousand.
        """
        response = self._ask(
            {
                "op": "step",
                "dt": float(dt),
                "control": {
                    "climb": float(self._cmd.throttle),
                    "yaw": float(self._cmd.yaw),
                    "forward": float(self._cmd.forward),
                    "lateral": float(self._cmd.lateral),
                    "escape": bool(self._cmd.escape),
                },
                # Frames come from frame(), once a tick, rather than from every
                # step: the loop looks, thinks, then acts, and the acting is
                # several steps to one look.
                "camera": False,
            }
        )
        self._tel = response["telemetry"]
        self._t = self._tel["time"]
        self.crashes = int(self._tel.get("crashes", 0))
        self._frame = None

    def telemetry(self) -> Telemetry:
        t = self._tel
        return Telemetry(
            t=float(t.get("time", self._t)),
            alt_m=t.get("altitude"),
            vz_mps=t.get("verticalSpeed"),
            # A compass heading, and a turn rate that is positive to the right:
            # FlyPV's own euler yaw is neither, which is a trap it documents
            # and its bridge converts once so nothing downstream has to.
            yaw_deg=t.get("heading"),
            yaw_rate_dps=float(t.get("yawRate", 0.0)),
            x_m=t.get("east"),
            y_m=t.get("north"),
            battery_pct=t.get("batteryPercent"),
            flying=bool(t.get("flying", False)),
        )

    def frame(self) -> np.ndarray | None:
        """The view from the quad now, rendered on demand and cached per step."""
        if not self.has_camera:
            return None
        if self._frame is None:
            response = self._ask({"op": "look"})
            self._tel = response["telemetry"]
            camera = response.get("camera")
            self._frame = None if camera is None else decode_frame(camera)
        return self._frame

    # ---------------------------------------------------------------- extras
    @property
    def collisions(self) -> int:
        """Crashes so far, under the name ``SimDrone`` uses for its own count."""
        return self.crashes

    @property
    def phase(self) -> str:
        """Where the bridge is in the flight: grounded, arming, takeoff, flying, landing."""
        return str(self._tel.get("phase", "grounded"))

    @property
    def arming_refusal(self) -> str | None:
        """Why it would not arm, or None — a real flight controller refuses for real reasons."""
        return self._tel.get("armingRefusal")

    def recording(self) -> dict:
        """The flight so far as a FlyPV blackbox recording, ready to save and replay."""
        return self._ask({"op": "recording"})["recording"]

    def save_recording(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(json.dumps(self.recording()), encoding="utf-8")
        return out
