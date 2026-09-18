# Changelog

## Unreleased
Audio: `flydrones compose --render track.wav` writes a finished track offline — an arrangement, a
synthesiser and a master, in numpy and scipy with no audio library. `music/arrange.py` gives a flight a
form: seven sections in D lydian, eight chords, thirteen neuron groups mapped onto seven musical roles,
pitches snapped onto the chord that is sounding with voice leading and register clamping to keep each line
its own, and a cast that changes from section to section — inventing nothing, since every note is still a
spike (`--arrange none` renders without it). `music/master.py` measures loudness to ITU-R BS.1770-4,
calibrated against the standard's own coefficients and test tone, and delivers -14 LUFS at -1 dBTP through
a four-times-oversampled true-peak limiter, as 24-bit PCM. `--to none` gives a render-only run a sink that
goes nowhere. [`assets/flight-track.wav`](assets/flight-track.wav) is two and a half minutes of it. See
[docs/MUSIC.md](docs/MUSIC.md#rendering-it-to-a-file).

LTE: `flydrones link` reads an Orange Airbox / Flybox / Home 4G+ (Huawei HiLink or ZTE) over its own LAN API,
times a round trip to the drone's endpoint, and scores the link; the safety governor holds the drone when
that score stays low and lands it when it goes, with `--link mock` to exercise the lot without a router.
See [docs/LTE.md](docs/LTE.md).

3D flight replay: `--track` on `learn`, `radio` and `demo` writes a flight down, `docs/live/replay.html`
draws it in the 3D bedroom (path coloured by what the fly had learned, event markers, compass needle,
chase and drone cameras), and `tools/record_flight_3d.py` records that page to a GIF or an MP4 without a
GPU. See [docs/REPLAY.md](docs/REPLAY.md).

Radio Cognitive Fruit Fly (`flydrones radio`): an always-on station with a six-show programme, a front page
that shows the heading needle, the memory bar and a live station log, pack swaps and recovery so the fly
never lands for good, and the same `--to` targets as the compositor. See [docs/RADIO.md](docs/RADIO.md).

Cognition: MiniFly gains a mushroom body (visual projection neurons -> 360 Kenyon cells -> APL feedback
inhibition -> MBON-g1pedc, with PPL1 dopamine depressing the Kenyon-cell synapses that were active when
something hurt) and a central complex (an EPG ring with Delta7 inhibition, PEN shift cells driven by the
halteres, FC2 goals and PFL3 steering). `flydrones learn --compare` flies ten approaches to the chair with
and without the plasticity; `tools/bench_cognition.py` measures the lot. See [docs/COGNITION.md](docs/COGNITION.md).

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
