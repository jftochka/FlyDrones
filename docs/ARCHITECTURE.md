# Architecture

![architecture](../assets/architecture.svg)

## One control tick (default 20 Hz)

```python
frame   = drone.frame()                          # BGR image or None
vision  = retina.encode(frame)                   # per-eye grids: brightness, ftb, btf, up, down, loom, loom_speed
vision  = illusion.apply(vision, gesture, t)     # optional hand -> optic-flow illusion
tel     = drone.telemetry()                      # altitude, yaw rate, battery, position
inputs  = encoder.encode(vision, tel.yaw_rate)   # {group: Hz per neuron}
rates   = brain.tick(inputs, ms=dt*1000)         # run the connectome for one tick, mean Hz per group
cmd     = decoder.update(rates, dt)              # linear read-out + escape reflex
cmd     = safety.filter(cmd, tel, dt)            # limits always win
drone.send(cmd)
```

Code: [`runtime.py`](../src/flydrones/runtime.py).

## Modules

| module | responsibility | key classes |
|---|---|---|
| `brain/connectome.py` | signed synapse matrix, cell-type labels, group selection by regex + side, save/load `.npz`, sensorimotor subgraph, MaleCNS feather builder | `Connectome`, `GroupSpec`, `build_malecns` |
| `brain/lif.py` | event-driven LIF simulation, delays, refractoriness, Poisson input, tonic bias, cheap copies | `LIFNetwork`, `LIFParams` |
| `brain/brain.py` | ties connectome + LIF + config groups; `tick()` returns group rates; `copy()` for swarms | `Brain` |
| `brain/synthetic.py` | MiniFly | `build_minifly` |
| `senses/retina.py` | numpy optic flow per grid cell, affine looming estimate | `Retina`, `VisualFrame` |
| `senses/gestures.py` | MediaPipe / OpenCV / scripted hands, illusions | `GestureIllusion`, `MediaPipeHands`, `OpenCVHands` |
| `senses/encoder.py` | features → Poisson rates per input neuron | `InputEncoder` |
| `motor/decoder.py` | descending-neuron rates → `FlightCommand`, baselines, escape, cruise braking | `MotorDecoder` |
| `safety.py` | clamps, slew rate, ceiling fade, floor, geofence, watchdog, battery, flight time | `SafetyGovernor`, `Telemetry` |
| `drones/*` | backends | `SimDrone`, `FlyPVDrone`, `TelloDrone`, `CrazyflieDrone`, `MavlinkDrone`, `UDPBridgeDrone` |
| `calibrate.py` | stimulus battery + ridge regression read-out | `calibrate` |
| `viz/dashboard.py` | matplotlib dashboard, OpenCV window, GIF writer | `Dashboard` |

## The simulator core

Connectivity is a CSC matrix `W[post, pre]` of signed synapse counts. When neurons spike, their columns are
gathered with one vectorised `np.repeat` / `np.bincount` pass, so cost scales with **spikes × out-degree**,
not with the total number of synapses. Arriving input sits in a ring buffer for `delay / dt` steps.
Membrane updates are in-place numpy operations over all neurons; refractory neurons are tracked as a short
index list. For very dense bursts it falls back to a sparse matrix-vector product.

## The other simulator

`drones/flypv.py` runs [FlyPV](https://github.com/jftochka/FlyPV) — an FPV
simulator with a real 1 kHz rate-mode flight controller, prop inflow, battery
sag and a noisy gyro — as a child process, and speaks one JSON line per step to
it. It is a `Drone` like any other, with a `step(dt)` like `SimDrone`, so
`run_sim` drives either.

Three things make it more than a swap of one simulator for another:

- **The camera is really looking at the world.** FlyPV casts a ray per pixel
  against the geometry its physics collides with, so the flow field reaching
  T4/T5 is real parallax rather than a drawn room.
- **It is lockstep.** Nothing sleeps on either side: the step is asked for and
  the answer comes back. A brain that runs at a tenth of real time flies
  exactly the flight a fast one does, which is not true of hardware and was not
  true of `SimDrone` either.
- **The flight comes back as a FlyPV recording**, which replays exactly in its
  browser UI — so what a connectome flew can be watched from the quad's camera
  and compared against a human's on the same airframe.

The mapping is in one place, and the two halves of it are worth stating: this
repository's `throttle` is a vertical speed and FlyPV's `climb` is the same
thing under a name that does not collide with a stick position; `yaw` is
positive to the right on both sides, and the heading FlyPV reports on the wire
is a compass heading rather than its own euler yaw, which runs the other way.

## Configuration

Everything lives in YAML: [`src/flydrones/defaults.yaml`](../src/flydrones/defaults.yaml) is the full
reference, files in [`configs/`](../configs) override parts of it (`--config`). Dictionaries merge deeply,
except `terms` blocks, which replace.

## Adding things

- **New sensory channel:** compute a grid or scalar feature, add a group in `inputs:` with `feature: <name>`
  (scalar features can be passed through `InputEncoder.encode(..., extra={...})`).
- **New motor read-out:** add a group in `outputs:` and a term in `decoder.axes`.
- **New drone:** subclass `Drone` ([HARDWARE.md](HARDWARE.md#your-own-drone)).
- **New connectome** (FlyWire, BANC...): produce a `Connectome` with `weights`, `types`, `sides` and save it.

## Browser port

`docs/index.html` + `docs/live/` is the GitHub Pages demo. `engine.js` is a line-by-line JavaScript port of the
LIF simulator, retina, encoder, gesture illusions, decoder, safety governor and simulated drone.
`tools/export_web_brain.py` writes MiniFly and the default config to `docs/live/minifly.json`, and CI runs
`node tools/check_web_engine.mjs` to make sure the browser version still climbs, holds, escapes and descends like
the Python one. Rendering uses a vendored copy of three.js (MIT, `docs/vendor/`): `models.js` builds the drone, the fly mascot and the swatter from primitives, `room.js` the bedroom, `app.js` the HUD, the swat game, picking and sound. Browser-only additions to the engine: neuron poking (`Brain.poke`), non-solid moving obstacles that the camera can see, and a giant-fiber jump used by the game.
