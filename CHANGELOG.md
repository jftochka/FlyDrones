# Changelog

## Unreleased
FlyPV backend: fly the connectome inside a real FPV simulator (`--drone flypv`) over a JSON-lines pipe —
1 kHz rate-mode flight controller, prop inflow, battery sag, noisy gyro, five airframes, and a ray-cast
camera the optic flow is computed from. `--flypv-record` saves the flight as a FlyPV blackbox recording,
which replays exactly in its browser UI. `run_sim` now drives anything with a `step(dt)`.

## 0.1.2 - 2026-09-16
Live demo 2.0: detailed blue quadcopter with a fly mascot, furnished bedroom with walls, day/night themes, shadows,
swat-the-drone game driven by the looming pathway, click-to-stimulate neurons, clickable 3D objects, camera modes,
particles and sound.

## 0.1.1 - 2026-09-15
Browser demo on GitHub Pages (three.js, JS port of the engine, webcam hands), hero GIF, social preview image,
CI check that the browser engine matches Python.

## 0.1.0 - 2026-09-15
First public release: simulator, MaleCNS loader, MiniFly, retina, gestures, decoder, safety governor,
Tello / Crazyflie / MAVLink / ESP32 backends, dashboard, swarm, calibration, documentation.
