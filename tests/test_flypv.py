"""The FlyPV backend, against a stand-in bridge.

The client and the protocol are what is tested here: what each FlightCommand
axis turns into, how telemetry comes back, when a frame is rendered, and what
happens when the far end says no. The real bridge is TypeScript and needs a
Node toolchain, so the end-to-end flight is a separate test below that skips
itself when there is no FlyPV checkout to fly in.
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
import pytest

from flydrones.drones import flypv as flypv_module
from flydrones.drones import make_drone
from flydrones.drones.flypv import PROTOCOL, FlyPVDrone, FlyPVError, decode_frame, find_repo
from flydrones.motor.command import FlightCommand

FAKE = [sys.executable, str(Path(__file__).with_name("fake_flypv_bridge.py"))]


def drone(**kw) -> FlyPVDrone:
    d = FlyPVDrone(command=FAKE, camera=(4, 3), **kw)
    d.connect()
    return d


def test_decode_frame_is_a_bgr_image():
    pixels = bytes([0, 64, 128, 255, 7, 9])
    camera = {"width": 3, "height": 2, "pixels": base64.urlsafe_b64encode(pixels).decode().rstrip("=")}
    image = decode_frame(camera)
    assert image.shape == (2, 3, 3)
    assert image.dtype == np.uint8
    # Grey, so the three channels agree, and the rows are in image order.
    assert list(image[0, :, 0]) == [0, 64, 128]
    assert (image[..., 0] == image[..., 2]).all()


def test_decode_frame_rejects_the_wrong_number_of_pixels():
    camera = {"width": 8, "height": 8, "pixels": base64.urlsafe_b64encode(b"short").decode()}
    with pytest.raises(FlyPVError):
        decode_frame(camera)


def test_connect_checks_the_protocol():
    d = FlyPVDrone(command=[*FAKE, str(PROTOCOL + 1)])
    with pytest.raises(FlyPVError, match="protocol"):
        d.connect()
    d.close()


def test_connect_says_when_a_world_is_not_there():
    d = FlyPVDrone(command=FAKE, world="atlantis")
    with pytest.raises(FlyPVError, match="atlantis"):
        d.connect()
    d.close()


def test_command_axes_reach_the_axes_they_should():
    d = drone()
    d.takeoff()
    d.send(FlightCommand(throttle=0.4, yaw=-0.3, forward=0.6, lateral=-0.2, escape=True))
    d.step(0.05)
    # The whole backend in one assertion: FlyDrones' throttle is a vertical
    # speed, and FlyPV's climb is the same thing under a name that does not
    # collide with a stick position.
    control = d._tel["control"]
    assert control == {"climb": 0.4, "yaw": -0.3, "forward": 0.6, "lateral": -0.2, "escape": True}
    d.close()


def test_telemetry_is_mapped_into_the_units_the_governor_reads():
    d = drone()
    d.step(0.1)
    t = d.telemetry()
    assert t.alt_m == 1.25
    assert t.vz_mps == 0.5
    assert t.x_m == 2.0 and t.y_m == -3.0
    assert t.yaw_deg == 90.0
    assert t.yaw_rate_dps == 12.0
    assert t.battery_pct == 88.0
    assert t.t == pytest.approx(0.1)
    d.close()


def test_a_frame_is_rendered_once_per_step_and_cached():
    d = drone()
    first = d.frame()
    again = d.frame()
    assert first is not None and first.shape == (3, 4, 3)
    assert again is first  # the same step, so the same picture
    d.step(0.05)
    assert d.frame() is not first  # a new step, so a new one
    d.close()


def test_no_camera_means_no_frames():
    d = FlyPVDrone(command=FAKE, camera=None)
    d.connect()
    assert d.has_camera is False
    assert d.frame() is None
    d.close()


def test_the_flight_comes_back_as_a_recording(tmp_path):
    d = drone()
    d.takeoff()
    d.step(0.05)
    out = d.save_recording(tmp_path / "flight.json")
    saved = json.loads(out.read_text())
    assert saved["version"] == 18
    assert saved["setup"]["world"] == "warehouse"
    d.close()


def test_a_refusal_is_an_error_and_not_a_silent_nothing():
    d = drone()
    with pytest.raises(FlyPVError, match="positive dt"):
        d.step(0)
    d.close()


def test_talking_to_a_bridge_that_is_not_running():
    d = FlyPVDrone(command=FAKE)
    with pytest.raises(FlyPVError, match="not running"):
        d.step(0.05)


def test_phase_and_crashes_are_visible():
    d = drone()
    assert d.phase == "grounded"
    d.takeoff()
    d.step(0.05)
    assert d.phase == "flying"
    assert d.collisions == 2
    assert d.arming_refusal is None
    d.close()


def test_make_drone_knows_the_name():
    d = make_drone("flypv", command=FAKE)
    assert isinstance(d, FlyPVDrone)


def test_find_repo_says_where_it_looked(monkeypatch, tmp_path):
    # With the sibling search turned off, so the test says the same thing on a
    # machine that happens to have a FlyPV checkout next door as on one that
    # does not.
    monkeypatch.delenv("FLYPV_REPO", raising=False)
    monkeypatch.setattr(flypv_module, "SEARCH", ())
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FlyPVError, match="Looked in"):
        find_repo("/nowhere/at/all")


# ---------------------------------------------------------------- end to end
def _flypv_repo() -> Path | None:
    try:
        return find_repo(None)
    except FlyPVError:
        return None


@pytest.mark.skipif(_flypv_repo() is None or shutil.which("npx") is None,
                    reason="needs a FlyPV checkout and Node (set $FLYPV_REPO)")
def test_really_flies_in_flypv():
    """Arm, take off and hover in the real thing, then replay what it flew.

    The one test that proves the two programs still fit together: everything
    above would pass just as happily against a bridge that had drifted out of
    date.
    """
    d = FlyPVDrone(camera=(32, 24), takeoff_height_m=1.0)
    d.connect()
    try:
        assert d.hello["protocol"] == PROTOCOL
        d.takeoff()
        for _ in range(100):
            d.step(0.05)
            if d.phase == "flying":
                break
        assert d.phase == "flying"
        telemetry = d.telemetry()
        assert telemetry.flying is True
        assert 0.7 < telemetry.alt_m < 1.5
        assert d.arming_refusal is None

        frame = d.frame()
        assert frame is not None and frame.shape == (24, 32, 3)
        assert frame.std() > 0  # a picture of something, not a flat grey

        # Forward is north at a heading of zero, which is where a flight starts.
        d.send(FlightCommand(forward=0.4))
        for _ in range(60):
            d.step(0.05)
        assert d.telemetry().y_m > 0.5

        recording = d.recording()
        assert recording["setup"]["airframe"] == "cinewhoop3"
        assert recording["spans"]
    finally:
        d.close()


@pytest.mark.skipif(_flypv_repo() is None or shutil.which("npx") is None,
                    reason="needs a FlyPV checkout and Node (set $FLYPV_REPO)")
def test_the_camera_gives_the_retina_real_optic_flow():
    """The point of flying in FlyPV, asserted rather than assumed.

    A camera that renders a picture is not the same thing as a camera the fly
    can see motion in: a flat-shaded wall is a flow field of exactly zero, and
    a pattern painted on the lens instead of on the world is worse — it looks
    right and never moves. So this flies and checks the signs: still when it is
    still, front-to-back when it goes forward, and rotating the other way when
    it turns right.
    """
    from flydrones.config import load_config
    from flydrones.senses import Retina

    retina = Retina.from_config(load_config(None, {}))
    d = FlyPVDrone(camera=(96, 72), quiet=True)
    d.connect()
    try:
        d.takeoff()
        for _ in range(60):
            d.step(0.05)
        assert d.phase == "flying"

        def flow(cmd: FlightCommand):
            d.send(cmd)
            vision = None
            for _ in range(30):
                vision = retina.encode(d.frame())
                d.step(0.05)
            return vision

        still = flow(FlightCommand())
        assert abs(still.rotation) < 0.05
        assert still.eyes["L"].get("ftb").mean() < 0.02

        forward = flow(FlightCommand(forward=0.6))
        # Flying at the scene: it flows front to back, and it expands.
        assert forward.eyes["L"].get("ftb").mean() > 0.05
        assert forward.eyes["L"].get("ftb").mean() > forward.eyes["L"].get("btf").mean()
        assert forward.expansion > 0

        turning = flow(FlightCommand(yaw=0.6))
        # Rotation is positive when the scene moves right, which is the drone
        # yawing left — so a turn to the right is negative here.
        assert turning.rotation < -0.2
    finally:
        d.close()


@pytest.mark.skipif(_flypv_repo() is None or shutil.which("npx") is None,
                    reason="needs a FlyPV checkout and Node (set $FLYPV_REPO)")
def test_the_demo_flies_the_whole_loop_in_flypv():
    """The brain, the eyes, the governor and FlyPV, from the command line."""
    env = {**os.environ, "FLYPV_REPO": str(_flypv_repo())}
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "flydrones", "demo", "--drone", "flypv", "--seconds", "4"],
        capture_output=True, text=True, timeout=600, env=env, check=False,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "collisions:" in out.stdout
