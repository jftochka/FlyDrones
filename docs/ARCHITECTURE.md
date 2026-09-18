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
| `drones/*` | backends | `SimDrone`, `TelloDrone`, `CrazyflieDrone`, `MavlinkDrone`, `UDPBridgeDrone` |
| `calibrate.py` | stimulus battery + ridge regression read-out | `calibrate` |
| `viz/dashboard.py` | matplotlib dashboard, OpenCV window, GIF writer | `Dashboard` |

## The simulator core

Connectivity is a CSC matrix `W[post, pre]` of signed synapse counts. When neurons spike, their columns are
gathered with one vectorised `np.repeat` / `np.bincount` pass, so cost scales with **spikes × out-degree**,
not with the total number of synapses. Arriving input sits in a ring buffer for `delay / dt` steps.
Membrane updates are in-place numpy operations over all neurons; refractory neurons are tracked as a short
index list. For very dense bursts it falls back to a sparse matrix-vector product.

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

## Cognition

`brain/cognition.py` runs beside the reflex path: it builds the sparse scene code the mushroom body sees,
applies the one plastic rule (dopamine-gated depression of KC→MBON, through `LIFNetwork.synapses_between`
and `set_synapses`), reads the heading bump out of the EPG columns and turns a goal into a bump of drive on
FC2. `Pilot` feeds it the collision and escape events that count as punishment. Neither circuit is a
controller: both act through DNp03 and DNg02, and the safety governor still has the last word.
See [COGNITION.md](COGNITION.md).

## Music

`music/` is a second read-out of the same tick, running beside the flight one: a `Compositor` turns group
firing rates and the flight command into notes and controls, and sinks render them for Pure Data, Max/MSP,
TidalCycles, SuperDirt, Strudel or a file. It reads the spike raster, so notes carry the millisecond their
neurons fired rather than the control grid. Nothing in it can change a command — the music is downstream of
the safety governor. See [MUSIC.md](MUSIC.md).

## The station

`radio.py` is a programme (a list of `Show`s) on top of one `Pilot` and one `Compositor`: it applies a show's
key, tempo, cruise and learning, drives the scenario (laps at the chair, a new goal every thirty seconds),
watches for anything worth saying, and keeps the drone in the air across pack swaps and landings. What it
knows goes to listeners through `Sink.set_context`, which the browser target merges into every frame.
See [RADIO.md](RADIO.md).

## The link

`link/` reads an LTE router (Huawei HiLink or ZTE, as Orange ships them) and times a round trip to the far
end. `LinkProbe` polls on its own thread — a router takes tens of milliseconds to answer and the control
loop is 20 Hz — and the flight loop reads its last verdict, which `SafetyGovernor.check_link` turns into
hold, land or nothing. See [LTE.md](LTE.md).

## Replay

`track.py` writes a flight down — position, heading, commands, and what the mushroom body and the compass
knew at the time — and `docs/live/replay.html` renders it in the same 3D bedroom, with the path coloured by
how much the fly had learned by that point. It is the only way to see cognition in 3D, because the browser
engine does not have any: see [REPLAY.md](REPLAY.md). `tools/record_flight_3d.py` drives that page frame by
frame in headless Chromium to make the GIFs.

## Browser port

`docs/index.html` + `docs/live/` is the GitHub Pages demo. `engine.js` is a line-by-line JavaScript port of the
LIF simulator, retina, encoder, gesture illusions, decoder, safety governor and simulated drone.
`tools/export_web_brain.py` writes MiniFly and the default config to `docs/live/minifly.json`, and CI runs
`node tools/check_web_engine.mjs` to make sure the browser version still climbs, holds, escapes and descends like
the Python one. Rendering uses a vendored copy of three.js (MIT, `docs/vendor/`): `models.js` builds the drone, the fly mascot and the swatter from primitives, `room.js` the bedroom, `app.js` the HUD, the swat game, picking and sound. Browser-only additions to the engine: neuron poking (`Brain.poke`), non-solid moving obstacles that the camera can see, and a giant-fiber jump used by the game.
