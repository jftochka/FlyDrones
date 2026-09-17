# Changelog

## Unreleased
The fly brain drone compositor: `flydrones compose` turns spikes and flight into music and sends it to
Pure Data, Max/MSP, TidalCycles, SuperDirt and Strudel (OSC, FUDI, `/ctrl`, `/dirt/play` and Server-Sent
Events), places notes at the spike times inside each control tick, writes the flight out as Tidal or Strudel
mini-notation, and ships the receiving patches and a browser player. `demo` and `fly` gained `--music`.
See [docs/MUSIC.md](docs/MUSIC.md).

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
