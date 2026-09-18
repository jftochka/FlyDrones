# Cognition: a memory and a compass

The reflex fly is impressive and thin. Looming grows, the giant fiber fires, the
drone jumps; the world drifts up, DNg02 pushes, the drone climbs. Nothing is
remembered, nothing is anticipated, and the fly that has flown into the chair
ten times arrives at the eleventh exactly as fast.

Two circuits change that, and both are in the fly:

* the **mushroom body**, where a sparse code of what the fly is looking at meets
  a dopaminergic neuron that reports that something hurt, and the synapses that
  were active at that moment are weakened;
* the **central complex**, where a ring of neurons holds a heading, the halteres
  push it around as the body turns, and a second population steers until the
  heading matches a goal.

```bash
flydrones learn --compare        # ten approaches to the chair, with and without learning
python tools/bench_cognition.py  # every number on this page, measured again
```

- [The memory](#the-memory)
- [What the fly learns, measured](#what-the-fly-learns-measured)
- [The compass](#the-compass)
- [Steering to a goal](#steering-to-a-goal)
- [Configuration](#configuration)
- [The honest part](#the-honest-part)

## The memory

```
eye ─► VPN-MB ─► KCg-d (360 cells, ~4 inputs each) ─► MBON-g1pedc ─┐
                    ▲         │                                     │ GABA
                    └── APL ◄─┘  (feedback inhibition: keeps        ▼
                                  ~10% of the cells firing)      LAL-turn ─► DNp03
                         PPL1-g1 (dopamine) ─────────┘                    (a turn away)
```

The chain is the fly's, cell type by cell type. Visual projection neurons carry
a contrast code of the scene into the calyx. Each Kenyon cell samples about four
of them at random, so it only fires when several of its inputs are on at once,
and the APL giant interneuron — one cell per hemisphere, inhibitory, driven by
all of them — turns that into a sparse code: **12% of the Kenyon cells** in the
measurement, a different 12% for a different view.

MBON-g1pedc reads all of them and is **GABAergic**: while it fires, the
avoidance turn behind it is held off. That is the naive fly, cruising at the
chair because nothing tells it not to.

When something hurts, PPL1-g1 fires, and every Kenyon-cell synapse onto the MBON
that was active at that moment is depressed:

```
w ← w · (1 − learning_rate · dopamine · activity_of_that_Kenyon_cell)
```

That is the whole rule ([`brain/cognition.py`](../src/flydrones/brain/cognition.py)),
and it is the one from the fly: coincidence of Kenyon-cell activity and dopamine
depresses that synapse ([Hige et al. 2015](https://www.cell.com/neuron/fulltext/S0896-6273(15)00600-2),
[Aso & Rubin 2016](https://elifesciences.org/articles/16135)). Depressed
synapses creep back to where they started over five minutes, so a memory is not
forever.

Nothing else in the brain changes. The wiring is fixed; 343 numbers move.

## What the fly learns, measured

`python tools/bench_cognition.py memory` — one scene punished three times,
another one left alone:

```
                       KCs firing   MBON   turn  DNp03  DNp01  memory
naive, scene A               0.12   38.1    0.3    0.0    0.0    0.00
naive, scene B               0.10   49.8    0.0    0.0    0.0    0.00
scene A + dopamine 1         0.12   27.5    1.4    0.8    0.0    0.09
scene A + dopamine 2         0.11   10.4    7.9   10.8    0.0    0.12
scene A + dopamine 3         0.12    4.8   12.9   22.5    0.0    0.14
learned, scene A             0.11    4.0   13.3   22.9    0.0    0.14
learned, scene B             0.10   37.9    0.0    0.0    0.0    0.14
reflex (looming)             0.10   42.8    0.0   66.9   62.5    0.14
```

The punished scene stops driving the MBON, the turn is released and reaches
DNp03, the descending neuron that actually steers. The other scene is untouched
— that is the point of a sparse code. The giant fiber never learned anything and
does not need to.

In flight (`flydrones learn --compare`, ten approaches to the chair at 0.9 m/s):

| | naive fly | after ten approaches |
|---|---|---|
| closest approach | 0.24 m | **1.78 m** |
| MBON | 41 Hz | **7.8 Hz** |
| avoidance turn | 0.4 Hz | **6.4 Hz** |
| escapes | one nearly every lap | none after the second |

With `--no-learning` — the same brain, the same room, the same seed, the
plasticity switched off — the closest approach stays at 0.24 m and the fly is
still dodging at the last moment on lap ten, sometimes too late.

## The compass

```
                 ┌───────────── Delta7 (inhibits the whole ring) ◄──┐
                 ▼                                                  │
   EPG columns ──┴── neighbours ──► a single bump of activity ──────┘
        │  ▲
        ▼  │
   PEN_L / PEN_R  ◄── halteres (body rotation)
   shift the bump one column per turn
```

Sixteen columns, 22.5° each. Local excitation keeps a bump alive, Delta7
inhibition makes sure there is only one, and the PEN cells — driven by the
halteres, which are the fly's gyroscopes — push it round as the body turns
([Seelig & Jayaraman 2015](https://www.nature.com/articles/nature14446),
[Green et al. 2017](https://www.nature.com/articles/nature22343),
[Turner-Evans et al. 2017](https://elifesciences.org/articles/23496)).

Measured against the drone it is riding on (`tools/bench_cognition.py compass`):

```
yaw command body dps   gain drift deg/s  bump
       0.15     17.8   1.23       -0.00  0.95
       0.40     47.5   1.81       -0.00  0.86
      -0.40     47.5   2.58       -0.00  0.86
       0.60     71.2   1.54       -0.00  0.83
```

It turns the right way at every rate, it holds still when the drone does
(**0.0°/s** of drift, over a window where a real fly in the dark would have
lost several degrees), and the bump is sharp — but the gain is between one and
two and a half, not one. **This is a heading, not a protractor.** It is enough
to hold a course, because holding a course is a loop that nulls its own error.

Two things about the ring are worth knowing before changing it, both learned the
expensive way and both written into `brain/synthetic.py`:

* its connections carry **no weight jitter and no random dropout**, unlike every
  other pathway in MiniFly. A ring attractor is a competition between columns,
  so one lucky column with heavier synapses wins every time and the bump sticks
  to it instead of following the halteres.
* the tonic drive belongs on EPG (9 mV against a 7 mV threshold: every column
  would fire, and Delta7 is what leaves one). The shift cells sit *under*
  threshold so that they only speak when the halteres do.

## Steering to a goal

FC2 holds a goal column. PFL3 on each side sees the goal and the bump a quarter
turn apart, so one side or the other lights up depending on which way round the
gap is, and drives DNg02 asymmetrically until the two line up
([Hulse et al. 2021](https://elifesciences.org/articles/66039),
[Westeinde et al. 2024](https://www.nature.com/articles/s41586-024-07021-y)).

```python
pilot.cognition.set_goal(90.0)       # a heading in the compass's own frame
pilot.cognition.hold_current_heading()
pilot.cognition.set_goal(None)       # and the fly stops caring
```

`tools/bench_cognition.py goal` flies it:

```
  goal |error| start |error| end body turned
    90          90.0         4.6       106.0
   -90          90.0         0.0       -71.1
   150         150.0        15.0       196.8
```

The quarter-turn offset decides which way the fly turns to close the gap, and
the wrong sign is a loop that runs *away* from the goal and settles 180° from
it — which is what the first version of this did, and why the offset has a
comment and a bench instead of an argument.

## Configuration

Everything is in the `cognition:` block of
[`defaults.yaml`](../src/flydrones/defaults.yaml):

```yaml
cognition:
  enabled: true
  columns: 16
  scene_keep: 0.45          # fraction of eye cells kept in the code the KCs see
  compass_seed_s: 0.6       # the bump is planted for this long at the start of a flight
  collision_punishment: 1.0
  punish_on_escape: 0.8     # a near miss teaches as well, at a lower dose
  mushroom_body:
    learning_rate: 0.55
    forget_s: 300
    floor: 0.08             # how far one synapse may be depressed
```

Synapse counts for both circuits are named constants (`MB` and `CX`) at the top
of `brain/synthetic.py`; `tools/bench_cognition.py` is how they were chosen.

On the real connectome the same groups are regexes over MaleCNS cell types —
`KCg-d`, `MBON-g1pedc`, `PPL1`, `EPG`, `PEN`, `Delta7`, `FC2`, `PFL3` are all
real names — so a MaleCNS brain built with `flydrones build-brain` gets the same
read-outs without any code change. The thresholds are deltas on measured resting
rates for exactly that reason.

## The honest part

- **The learning rule is the fly's; the teacher is ours.** Coincidence-based
  depression of KC→MBON under dopamine is from the literature. What counts as
  punishment — a collision, and a near miss at a lower dose — was decided by us
  and is in `defaults.yaml`. A real fly's PPL1 neurons respond to heat, shock
  and bitter taste, not to a quadcopter hitting a chair.
- **The sparse code is visual, which is unusual.** Most mushroom-body work is
  olfactory. Visual input to the γd Kenyon cells is real
  ([Vogt et al. 2016](https://elifesciences.org/articles/14009)) but the code
  here — a contrast map of a camera frame — is engineering.
- **The compass is not calibrated** (see the table above), and it is set at
  take-off rather than anchored to a landmark. The fly's ER ring neurons do the
  anchoring with visual input, and they are not modelled here, so headings are
  relative to where the flight started.
- **Both circuits sit outside the control path.** They act through DNp03 and
  DNg02, the same descending neurons the reflexes use, and the safety governor
  still has the last word over all of them.
- **The browser demo does not have any of this.** `docs/live/engine.js` gets the
  neurons but not the inputs that drive them: no scene code, no dopamine, no
  goal. The browser fly is the reflex fly; cognition runs in Python.
