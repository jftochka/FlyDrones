# Music: the fly brain drone compositor

A fly brain flying a drone is already a performance: a few hundred neurons firing at
each other a thousand times a second, a hand in front of a camera, and a machine that
answers. `flydrones compose` turns that into notes and sends them to the programs
people improvise with — **Pure Data**, **Max/MSP**, **TidalCycles** and **Strudel**.

```bash
pip install -e .
flydrones compose --to strudel          # opens an address; press play in the browser
flydrones compose --to pd               # OSC into a vanilla Pd patch
flydrones compose --to tidal            # /ctrl values for your own Tidal patterns
flydrones compose --to superdirt        # straight to SuperDirt, no Haskell at all
flydrones compose --to max              # OSC into [udpreceive 7400]
flydrones compose --to pd,tidal,strudel --seconds 300     # all at once, for five minutes
```

Nothing else is needed to try it: the flight is simulated, the brain is MiniFly, and
the whole thing runs in real time on a laptop.

- [What plays what](#what-plays-what)
- [The five minutes version, per program](#the-five-minutes-version-per-program)
- [The wire formats](#the-wire-formats)
- [Timing](#timing)
- [Writing the flight down](#writing-the-flight-down)
- [Configuration](#configuration)
- [The honest part](#the-honest-part)
- [When it is silent](#when-it-is-silent)

## What plays what

Nine voices, each listening to one weighted sum of neuron groups. A voice opens when
its groups fire above `on_hz`, stays open until they fall below `off_hz`, and cannot
retrigger faster than `min_gap_s` — the same shape as a neuron, and the reason the
result is rhythm rather than a buzz.

| voice | neurons | in the fly | what you hear |
|---|---|---|---|
| `lift` | DNg02 L+R | wing-stroke amplitude | the bass pulse; faster and higher as the fly works harder |
| `wing_left` / `wing_right` | DNg02 L, DNg02 R | one side each | two melody lines, panned apart. A turn is the two drifting apart |
| `saccade_left` / `saccade_right` | DNp03 L, R | looming-evoked flight saccades | one percussive hit per burst |
| `giant_fiber` | DNp01 L+R | the escape reflex | one enormous note, then nothing for half a second |
| `loom` | LPLC2, LC4 | approach detectors | a riser that exists only while something is coming at the drone |
| `rising` / `falling` | T4c, T4d | up / down optic flow | high texture; which one plays says which way the world is moving |
| `kenyon` | KCg-d | the sparse code of the scene | a few high sparkles: what the fly is looking at |
| `memory` | MBON-g1pedc | holds the avoidance turn off | it plays when the MBON *lets go* — the sound of having learned |
| `dopamine` | PPL1 | punishment | one note per lesson |
| `avoid` | LAL-turn | the learned turn | it did not exist on the first lap, and it takes over the piece by the tenth |

Two voices read their group differently, because their signal is not excitement: `kenyon` reads the raw
rate (Kenyon cells idle at nothing, so the distance from rest says little), and `memory` reads the distance
*below* rest, which is what learning looks like from outside. `absolute: true` and `rectify: neg` in the
voice, both documented in `defaults.yaml`.

Pitch is the firing rate above rest, quantised into a scale
(`--scale minor_pentatonic --root C3` by default; every name in
`flydrones.music.events.SCALES` works). Velocity is the same number, unquantised.

Alongside the notes, every tick carries continuous controls:

| control | range | what it is |
|---|---|---|
| `drive` | 0..1 | DNg02 above rest: how hard the fly is beating its wings |
| `balance` | -1..1 | right minus left DNg02: which way it is turning |
| `loom` | 0..1 | LPLC2/LC4: something is approaching |
| `bright` | 0..1 | R1-R6 photoreceptors: how light the room is |
| `alt`, `alt_m` | 0..1, metres | altitude |
| `climb`, `turn`, `fwd` | -1..1 | the flight command the decoder produced |
| `escape` | 0 or 1 | the giant-fiber reflex is active |
| `battery` | 0..1 | pack remaining |
| `spikes` | kHz | spikes per second across the recorded neurons |
| `memory` | 0..1 | how much of the KC→MBON weight the fly has given up |
| `heading`, `heading_hold` | 0..1 | where the compass bump is, and how sharp it is |
| `goal` | 0..1, -1 for none | the heading it is steering to |
| `mbon`, `kc`, `dopamine` | 0..1 | the mushroom body's three signals |
| `cycle`, `cps` | | where the compositor thinks the cycle is |

and a `section` — `ground`, `hover`, `climb`, `descend`, `turn`, `escape` — which is
the closest thing the flight has to an arrangement.

## The five minutes version, per program

`flydrones compose --write-patches ~/flybrain` writes every receiving file below,
filled in with the voices your configuration actually has.

### Pure Data

```bash
flydrones compose --write-patches ~/flybrain
pd ~/flybrain/flybrain.pd          # or open it from Pd
flydrones compose --to pd
```

The patch is vanilla — `[netreceive -u -b 9000] → [oscparse] → [list trim] → [route fly]`
— so no externals, and the same messages are in your Pd window if you click the `1`
above the `spigot`. It contains a two-oscillator monosynth so that something is
audible immediately; the point is the messages, which you can route into your own
patch. `--to pd-fudi` sends Pd's own protocol instead (port 3001, `flybrain-fudi.pd`),
which is worth doing when you want to read every message as text.

### Max/MSP

```bash
flydrones compose --to max
```

Open `flybrain.maxpat`: `[udpreceive 7400]` → `[route /fly/note /fly/ctrl /fly/section /fly/frame]`
→ `[unpack s s 0 0. 0. 0. 0 0 0. 0.]` and a placeholder synth.

### TidalCycles

Tidal already listens for `/ctrl` on port 6010, so the fly becomes live values inside
patterns you write, while Tidal keeps the clock:

```bash
flydrones compose --to tidal
```

```haskell
let flyF name = cF 0 name
    flyN v    = cN 60 (v ++ "_n") - 60   -- MIDI on the wire, middle C = 0 in Tidal
    flyG v    = cF 0 (v ++ "_g")

d1 $ stack [
  note (flyN "wing_left") # s "superpwm" # struct "t*4"
    # gain (0.5 + 0.4 * flyF "drive") # lpf (200 + 4000 * flyF "bright"),
  note (flyN "giant_fiber") # s "superhoover" # struct "t*2" # gain (flyG "giant_fiber")
  ]
```

`flybrain-live.tidal` is that file, listing the controls of your configuration.

If you would rather have no Haskell in the room at all, `--to superdirt` sends
`/dirt/play` to SuperDirt on 57120 directly: the compositor takes the part Tidal would
play, which is the point when the pattern is a fly.

### Strudel

```bash
flydrones compose --to strudel
```

It prints an address (`http://127.0.0.1:8765/` by default) and waits up to twenty
seconds for you to open it. The page loads Strudel from a CDN, connects to a
Server-Sent Events stream of the flight, and gives you an editable pattern where every
value comes from a neuron:

```js
note(signal(() => fly.n('wing_left', 60))).s("triangle").pan(.2).struct("t*4")
  .gain(signal(() => .35 + .5 * fly.c('drive')))
```

`fly.c(name)` is a control, `fly.n(voice)` that voice's last note, `fly.gate(voice, s)`
is 1 for `s` seconds after it fired, `fly.notes(voice, 4)` the last few. To use
[strudel.cc](https://strudel.cc) instead of the local page, paste
`flybrain-strudel.mjs`, which carries its own small connector.

## The wire formats

Everything is one frame per control tick (20 Hz by default), OSC unless stated.

```
/fly/note    s:voice s:sound i:note f:velocity f:duration f:pan i:channel i:orbit f:rate_hz f:delay_ms
/fly/ctrl    s:name f:value
/fly/section s:name
/fly/frame   f:t f:cycle f:cps i:notes
```

* **Pure Data** (`--to pd`, port 9000) and **Max** (`--to max`, port 7400) get exactly that,
  bundled per tick. `note` is a MIDI number.
* **FUDI** (`--to pd-fudi`, port 3001) is the same, as Pd messages: `note wing_left flylead 67 0.8 ...;`.
  One datagram per message, because Pd's UDP `netreceive` evaluates the first message
  in a packet and silently drops the rest.
* **TidalCycles** (`--to tidal`, port 6010) gets `/ctrl name value` pairs: every control
  above, `section` as a string, and per voice `<voice>_n` (MIDI note), `<voice>_v`
  (velocity) and `<voice>_g` (1 on the tick it fired).
* **SuperDirt** (`--to superdirt`, port 57120) gets `/dirt/play` with the usual flat
  pairs — `s note gain pan orbit sustain delta cps cycle` — in a bundle timestamped for
  when the note should sound. Generic sound names are mapped to synths SuperDirt ships
  (`flylead` → `superpwm` and so on); change `music.sounds.superdirt` for your own kit.
* **Strudel** (`--to strudel`, port 8765) gets JSON over SSE, plus the player page.
* **A file** (`--to jsonl:flight.jsonl`) gets the same JSON, one frame per line.
  `flydrones compose --replay flight.jsonl --to pd` plays it again, in real time,
  without the brain.

## Timing

The brain runs at 2 kHz internally and the control loop at 20 Hz, so a tick contains
fifty milliseconds of spikes. The compositor reads the spike raster and places each
note at the millisecond its group actually fired, rather than on the control grid.

To keep that resolution on the way out, everything is sent with `music.latency_s`
(0.2 s, Tidal's own habit) of headroom: SuperDirt gets an OSC timetag it can schedule
on, and Pd and Max get the same figure as `delay_ms`, which `[delay]` and `[del]` take
directly. `--latency 0` gives you the lowest latency and the loosest timing.

## Writing the flight down

```bash
flydrones compose --seconds 60 --score flight.tidal     # or flight.mjs for Strudel
```

The score is the flight quantised onto `--steps` slots per cycle, as mini-notation, one
`<...>` slot per cycle in order:

```haskell
d1 $ stack [
  note "<[~ -12 ~ ~ 0 ~ ~ ~] [-9 ~ -9 ~ ~ ~ -7 ~]>" # s "supersaw" # pan 0.50 # orbit 1,  -- lift
  ]
```

This is the one place in the chain that throws timing away on purpose; a grid is what
makes the result editable. `--steps 16` is the default, and the run reports how many
notes two spikes put on the same step.

## Configuration

Every voice lives in the `music:` block of
[`defaults.yaml`](../src/flydrones/defaults.yaml) and can be changed with `--config`:

```yaml
music:
  scale: dorian
  root: A2
  cps: 0.6
  tempo_from: drive        # let the wing-stroke neurons move the tempo
  voices:
    lift:
      groups: {DNg02_L: 0.5, DNg02_R: 0.5}
      sound: flybass
      mode: repeat         # repeat = arpeggio while excited, trigger = one note per burst
      on_hz: 4             # Hz above the resting rate
      off_hz: 2
      span_hz: 22          # excursion that reaches the top of the range
      low: 36
      high: 55
      every_s: 0.5         # note spacing at the threshold
      fastest_s: 0.15      # note spacing when fully excited
```

Thresholds are **deltas on the resting rate the decoder measures during warm-up**, so
the same score works on MiniFly (DNg02 idles near 31 Hz) and on MaleCNS, where it idles
somewhere else entirely. Point a voice at any group in `inputs:` or `outputs:` — if you
add a cell type to the connectome config, it can play.

Any flight can drive music, not only `compose`:

```bash
flydrones demo --music jsonl:demo.jsonl        # the scripted demo, as fast as it runs
flydrones fly --drone tello --send --music pd  # a real drone, in real time
```

## The honest part

- **The mapping is engineered, like the rest of the bridge.** Which neuron plays which
  voice, and what counts as "excited", was chosen by us and written into
  `defaults.yaml`. The brain is not making music; it is flying, and we are listening.
- **The Pure Data path is tested end to end.** Both patches are loaded by `pd -nogui`
  in development and fed real packets. The Max patch is written against the documented
  `.maxpat` format but **has not been opened in Max by us** — like the hardware
  adapters, it is untested until someone with the hardware says otherwise.
- **The Tidal and SuperDirt messages are built against their source**: `/ctrl` takes
  exactly a name and a value, `cN` gives a `Pattern Note`, and `/dirt/play` takes flat
  key-value pairs with the bundle time as its latency. They have not been run against a
  live SuperCollider here.
- **`compose` paces itself to the wall clock.** If the brain cannot keep up (a large
  MaleCNS core on a slow machine) it says so once and the music drags rather than
  skipping; `flydrones bench` tells you the real-time factor beforehand.

## When it is silent

| symptom | why |
|---|---|
| no notes at all | the brain may still be settling — the first `settle_s` seconds measure the resting rates and nothing plays; also check `--to` reached the right port |
| notes in the terminal, nothing in Pd | the patch is listening on another port, or another program already holds it: Pd prints `netreceive: bind: Address already in use` |
| Strudel page loads, no sound | browsers need a click before audio: press play; the Strudel bundle comes from a CDN, so an offline machine gets the meters and no sound |
| SuperDirt logs "scheduling delay is too long" | your `--latency` is above SuperDirt's `maxLatency`, or the clocks are out of sync |
| only `lift` and the wings ever play | that is a calm flight: `saccade_*`, `loom` and `giant_fiber` only fire when something rushes at the drone. `--gestures scripted` includes those; `--seed` picks a different arrangement |
| the same seed gives different music | it should not: the conductor, the simulator and the brain are all seeded. Please open an issue with the two runs |
