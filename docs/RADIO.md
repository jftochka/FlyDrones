# Radio Cognitive Fruit Fly

A station that does not stop, played by a fruit fly's connectome flying a drone.

```bash
flydrones radio                     # open the address it prints, press play
flydrones radio --to strudel,pd     # and send the same notes to Pure Data
flydrones radio --to superdirt --hours 8
```

Nothing is recorded and nothing repeats. A simulated drone is in the air the
whole time with a spiking fly brain flying it; the compositor
([MUSIC.md](MUSIC.md)) turns the descending neurons into notes; the programme
decides what the fly is doing and in which key. The fly learns while it is on
air, and keeps what it learned into the next show.

- [The programme](#the-programme)
- [What you see](#what-you-see)
- [Keeping it on air](#keeping-it-on-air)
- [Listening elsewhere](#listening-elsewhere)
- [Running your own](#running-your-own)
- [The honest part](#the-honest-part)

## The programme

Six shows, forty minutes, then round again. Each sets the key, the tempo, what
the hand in front of the camera is doing, and whether the mushroom body is
learning.

| show | length | what happens | what to listen for |
|---|---|---|---|
| **Dawn Chorus** | 6 min | hovering, hardly any gestures | C minor pentatonic, slow. DNg02 breathing, the room going by |
| **The Chair** | 9 min | the fly flies at a chair, again and again, learning | the dopamine note, then the `memory` voice arriving and `avoid` taking over |
| **Afterwards** | 5 min | the same room, nothing to be afraid of | Lydian, tempo following altitude; the memory thinning out as it fades |
| **Compass Rose** | 7 min | a new heading to hold every thirty seconds | whole-tone; the needle swinging and settling on the page |
| **Swat Hour** | 6 min | something rushes the drone every few seconds | blues, fast, tempo driven by lift: giant fibre, saccades, the looming riser |
| **Night Flight** | 7 min | slow, low, mostly lift | Phrygian in F2, tempo following altitude |

The order is not arbitrary. **The Chair** teaches the fly that one view of the
room means trouble, and **Afterwards** is what that sounds like once nothing is
hurting it any more: the same synapses creeping back over five minutes, on air.

## What you see

The page is the station: what is playing, how long is left, what is next, and
the brain underneath it.

- **the compass** — a needle for the heading bump and a dashed marker for the
  goal, when there is one. This is the fly's own estimate; nothing tells it
  which way it is pointing.
- **the meters** — memory, lift, looming, Kenyon-cell activity, MBON, altitude,
  brightness, battery.
- **the voices** — one lamp per voice, lit when it fires. `memory`, `avoid` and
  `dopamine` are the cognitive ones; watch them arrive during The Chair.
- **the station log** — escapes, bumps, lessons, goals reached, pack swaps and
  show changes, with the time they happened.
- **the player** — Strudel, in the tab, with the station's pattern in an editable
  box. Change it and press play again; you are patterning a live fly.

The same information is on the terminal, as a log:

```
    0:00  station    Radio Cognitive Fruit Fly on air: 6 shows, 1,584 neurons
    0:00  show       now playing: Dawn Chorus — hovering, hardly anything happening…
    0:12  show       now playing: The Chair — the fly flies at a chair until it wants nothing to do with it…
    0:20  escape     the giant fibre fired
    0:20  learning   it has learned something: memory 0.08, MBON down to 18 Hz
    0:36  goal       new heading to hold: 232° (it is at 292°)
```

## Keeping it on air

A station cannot land, and the things that stop a flight are all in the
simulator:

* **the pack runs down** in about ten minutes, and the safety governor lands
  below 20%. The station swaps the pack at 35% and says so.
* **the governor's flight timer** would land it after three minutes. The station
  raises that limit for itself; every other limit — height, fence, slew rate —
  still applies, and the brain never gets to override any of them.
* **a landing for any other reason** (a crash, a fence) is answered by putting
  the drone back in the air and planting the compass bump again, which is a real
  restart of the heading: it is relative to where the flight resumed.

If the machine cannot run the brain in real time the station says so once and
the music drags rather than skipping — `flydrones bench` will tell you
beforehand. MiniFly runs about 15× real time on a laptop core.

## Listening elsewhere

`--to` takes everything the compositor takes, so the station can drive your own
instruments instead of the browser:

```bash
flydrones radio --to pd                 # OSC into the vanilla Pd patch
flydrones radio --to superdirt          # straight into SuperDirt
flydrones radio --to tidal              # /ctrl values for your own patterns
flydrones radio --to strudel,jsonl:tonight.jsonl   # and keep the score
```

A saved score replays without the brain: `flydrones compose --replay tonight.jsonl --to pd`.

## Running your own

```bash
flydrones radio --minutes-per-show 1        # the whole programme in six minutes
flydrones radio --show chair                # one show, on repeat
flydrones radio --seed 7                    # a different conductor, same shows
flydrones radio --hours 8 --to superdirt    # a night of it
```

The programme is a list of `Show` objects in
[`radio.py`](../src/flydrones/radio.py) — name, length, key, tempo, what the
conductor does, whether the mushroom body is learning, whether it flies laps at
the chair, and how often it is given a new heading. Adding one is a line.

## The honest part

- **It is a simulation flying a simulated drone.** No hardware is in the loop;
  `flydrones fly --music` is the version that plays a real one.
- **The programme is written by us**, as is the mapping from neurons to notes.
  The fly is not composing; it is flying, and we are listening in a particular
  way. [MUSIC.md](MUSIC.md) has the whole mapping and
  [COGNITION.md](COGNITION.md) has what the fly actually knows.
- **The seeds make it reproducible, not identical.** Same seed and same
  programme give the same flight; the conductor, the drone and the brain are all
  seeded. Change the seed and it is a different night.
- **Strudel comes from a CDN** at listening time. Offline, the page still shows
  the station and the brain, and the OSC targets still play.
