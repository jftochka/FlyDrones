"""flydrones command line."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__

BANNER = r"""
   ___ _       ___
  | __| |_  _ |   \ _ _ ___ _ _  ___ ___
  | _|| | || || |) | '_/ _ \ ' \/ -_|_-<
  |_| |_|\_, ||___/|_| \___/_||_\___/__/
         |__/   fruit fly connectome -> drone
"""


def _cfg(args):
    from .config import load_config

    over = {}
    if getattr(args, "brain", None):
        over.setdefault("brain", {})["source"] = args.brain
    return load_config(getattr(args, "config", None), over)


def _brain(cfg, quiet: bool = False):
    from .brain import Brain, load_connectome

    c = load_connectome(cfg["brain"]["source"])
    if not quiet:
        print(c.summary())
    return Brain(c, cfg)


def _record_or_show(dash, frames, infos, args, k):
    if args.record:
        if k % max(1, args.every) == 0:
            frames.append(dash.render(infos))
        else:
            dash.push(infos)
    elif getattr(args, "live", False):
        if k % 2 == 0:
            if not _record_or_show.window.show(dash.render(infos)):
                raise KeyboardInterrupt
        else:
            dash.push(infos)


_record_or_show.window = None


# ---------------------------------------------------------------- commands
def cmd_demo(args) -> int:
    from .drones import SimDrone
    from .runtime import Pilot, run_sim
    from .senses import ScriptedGestures, demo_timeline
    from .viz import Dashboard, LiveWindow, save_gif

    print(BANNER)
    cfg = _cfg(args)
    brain = _brain(cfg)
    pilot = Pilot(brain, SimDrone(start=(-1.5, 0.0, 0.0)), cfg, gestures=ScriptedGestures(demo_timeline()))
    dash = Dashboard(brain, [pilot], title="FlyDrones · hand -> fly eyes -> fly brain -> drone")
    music = _MusicTap(_music_cfg(cfg, args), brain, pilot, args.music) if args.music else None
    track = narrator = None
    if args.track:
        from .track import FlightNarrator, TrackRecorder

        track = TrackRecorder(title="Hand, fly eyes, fly brain, drone",
                              subtitle="the scripted demonstration, in the room it was flown in")
        narrator = FlightNarrator(track)
    frames: list = []
    if args.live:
        _record_or_show.window = LiveWindow()
    last_label = [""]

    def on_tick(k, infos):
        i = infos[0]
        label = i.illusion + (" | ESCAPE" if i.cmd.escape else "")
        if label != last_label[0]:
            print(f"t={i.t:5.1f}s alt={i.tel.alt_m:4.2f} m  {label}")
            last_label[0] = label
        if music:
            music(i)
        if track is not None:
            if i.illusion != last_label[0] and i.gesture is not None:
                track.chapter(i.t, i.illusion, "")
            track.add(i, hit=narrator.watch(i, pilot.drone))
        _record_or_show(dash, frames, infos, args, k)

    try:
        run_sim([pilot], args.seconds, hz=cfg["control"]["hz"], on_tick=on_tick)
    except KeyboardInterrupt:
        pass
    finally:
        if music:
            music.close()
    if track is not None:
        track.save(args.track)
        print(f"track -> {args.track} ({track.summary()})")
    print(f"collisions: {pilot.drone.collisions}")
    if args.record and frames:
        save_gif(frames, args.record, fps=int(cfg["control"]["hz"] / max(1, args.every)))
        print(f"saved {len(frames)} frames -> {args.record}")
    if args.log:
        _write_log(args.log, pilot.history)
    return 0


def cmd_swarm(args) -> int:
    from .drones import SimDrone
    from .runtime import Pilot, run_sim
    from .senses import GestureState, ScriptedGestures
    from .viz import Dashboard, LiveWindow, save_gif

    print(BANNER)
    cfg = _cfg(args)
    brain = _brain(cfg)
    roles = [
        ("hover", (-1.6, -1.4, 0.0), 0.0, 0.0, None),
        ("cruise", (-0.9, 0.05, 0.0), 0.0, 0.35, None),
        ("follow-hand", (-1.6, 1.0, 0.0), 0.0, 0.0,
         ScriptedGestures([(0.0, GestureState(True, 0.1, 0, 0, 0.15, "fist")), (6.0, GestureState(False, 0, 0, 1, 0, "dropped"))])),
    ]
    pilots = []
    for i, (_role, start, yaw, cruise, gest) in enumerate(roles[: args.n] + [roles[0]] * max(0, args.n - len(roles))):
        c = copy.deepcopy(cfg)
        c["decoder"]["cruise"] = cruise
        b = brain if i == 0 else brain.copy(seed=1000 + i)
        pilots.append(Pilot(b, SimDrone(start=start, yaw_deg=yaw, seed=i), c, gestures=gest, name=f"fly-{i + 1}"))
    print(f"1 connectome -> {len(pilots)} independent brains (same wiring, own spikes and noise)")
    dash = Dashboard(pilots[0].brain, pilots, title=f"FlyDrones · 1 fly brain, {len(pilots)} pilots")
    frames: list = []
    if args.live:
        _record_or_show.window = LiveWindow()
    try:
        run_sim(pilots, args.seconds, hz=cfg["control"]["hz"], on_tick=lambda k, infos: _record_or_show(dash, frames, infos, args, k))
    except KeyboardInterrupt:
        pass
    for p in pilots:
        h = p.history
        esc = sum(1 for a, b in zip(h, h[1:]) if b["escape"] and not a["escape"])
        print(f"{p.name}: final alt {h[-1]['alt']:.2f} m, pos ({h[-1]['x']:.2f}, {h[-1]['y']:.2f}), escapes {esc}, collisions {p.drone.collisions}")
    if args.record and frames:
        save_gif(frames, args.record, fps=int(cfg["control"]["hz"] / max(1, args.every)))
        print(f"saved {len(frames)} frames -> {args.record}")
    return 0


def cmd_fly(args) -> int:
    from .drones import DryRunDrone, make_drone
    from .runtime import Pilot, run_realtime

    print(BANNER)
    cfg = _cfg(args)
    brain = _brain(cfg)
    kw = {}
    if args.drone == "mavlink":
        kw = {"connection": args.mavlink, "autopilot": args.autopilot}
    elif args.drone == "esp32":
        kw = {"host": args.esp32_host}
    elif args.drone == "crazyflie":
        kw = {"uri": args.uri}
    drone = make_drone(args.drone, **kw) if args.send or args.drone == "sim" else None
    if drone is None:
        drone = DryRunDrone(_Stub(args.drone))
        print("DRY RUN: nothing will fly. Re-run with --send when the drone is in a safe, open space.")

    webcam = gestures = None
    if args.input in ("gesture", "both"):
        from .senses import make_gesture_source
        from .senses.webcam import Webcam

        webcam = Webcam(args.webcam)
        gestures = make_gesture_source(args.gestures)
    probe = None
    if getattr(args, "link", None):
        from .link import LinkError

        try:
            probe = _make_probe(args)
            probe.start()
        except LinkError as e:
            print(f"you asked to fly on LTE and there is no LTE: {e}", file=sys.stderr)
            return 2
        print(f"link: {probe.summary()}")
        if probe.last is not None and probe.last.score < cfg["safety"]["link"]["hold_below"]:
            print("the link is already below the hold threshold; the governor will not let it go anywhere")
    pilot = Pilot(brain, drone, cfg, gestures=gestures, webcam=webcam, link=probe)
    if args.input == "gesture" and drone.has_camera:
        drone.has_camera = False  # hand only

    window = None
    dash = None
    if args.live:
        from .viz import Dashboard, LiveWindow

        window = LiveWindow()
        dash = Dashboard(brain, [pilot], title=f"FlyDrones · {args.drone}")

    music = _MusicTap(_music_cfg(cfg, args), brain, pilot, args.music) if args.music else None
    tick = [0]

    def on_tick(info):
        tick[0] += 1
        if music:
            music(info)
        if window is None:
            return True
        if tick[0] % 2 == 0:
            return window.show(dash.render([info]))
        dash.push([info])
        return True

    try:
        if args.drone == "sim":
            from .runtime import run_sim

            run_sim([pilot], args.seconds or 30, hz=cfg["control"]["hz"], on_tick=lambda k, infos: on_tick(infos[0]))
        else:
            run_realtime(pilot, args.seconds, hz=cfg["control"]["hz"], on_tick=on_tick)
    finally:
        if music:
            music.close()
        if probe is not None:
            probe.stop()
            print(f"link at the end: {probe.summary()}")
    if args.log:
        _write_log(args.log, pilot.history)
    return 0


class _Stub:
    def __init__(self, name):
        self.name = name
        self.has_camera = False


def cmd_download(args) -> int:
    from .data import download_malecns

    download_malecns(args.dir)
    print("next: flydrones build-brain")
    return 0


def cmd_build(args) -> int:
    from .brain import Brain, build_malecns
    from .config import load_config

    cfg = load_config(args.config)
    c = build_malecns(args.data_dir, min_synapses=args.min_synapses)
    print(c.summary())
    b = Brain(c, cfg)  # resolves groups and warns about empty ones
    for name in list(b.input_specs) + list(b.output_specs):
        print(f"  {name:10s} {c.group(name).size:6d} neurons")
    if args.core_hops:
        c = c.sensorimotor_core(args.core_hops, args.max_neurons)
        print("sensorimotor core:", c.summary())
    out = c.save(args.out)
    print(f"saved -> {out}\nuse it: flydrones demo --brain {out}")
    return 0


def cmd_inspect(args) -> int:
    cfg = _cfg(args)
    brain = _brain(cfg)
    c = brain.connectome
    print("\ninput groups:")
    for k in brain.input_specs:
        print(f"  {k:10s} {c.group(k).size:6d}  types={brain.input_specs[k].types} side={brain.input_specs[k].side}")
    print("output groups:")
    for k in brain.output_specs:
        print(f"  {k:10s} {c.group(k).size:6d}  {brain.output_specs[k].note}")
    print("\npoke test (500 ms each):")
    brain.tick({}, 500)
    for g in [k for k in brain.input_specs if c.group(k).size]:
        r = brain.stimulate(g, args.hz, 500)
        print(f"  {g:10s} -> " + "  ".join(f"{k}={r[k]:5.1f}" for k in brain.output_specs))
    return 0


def cmd_learn(args) -> int:
    """Fly at the chair until the fly has had enough of it."""
    from .learn import approach_laps, summarise

    print(BANNER)
    cfg = _cfg(args)
    _brain(cfg)  # print the connectome summary once, before the laps
    runs = [("learning", True), ("no learning (control)", False)] if args.compare else [("learning", not args.no_learning)]
    summaries = {}
    track = narrator = None
    if args.track:
        from .track import FlightNarrator, TrackRecorder

        track = TrackRecorder(title="Learning to avoid the chair",
                              subtitle=f"{args.laps} approaches · the mushroom body is learning")
        narrator = FlightNarrator(track)
    seen_lap = [0]

    def on_tick(pilot, info, lap):
        if lap != seen_lap[0]:
            seen_lap[0] = lap
            track.chapter(info.t, f"approach {lap}",
                          f"memory {pilot.cognition.state.memory:.2f}" if pilot.cognition else "")
        hit = narrator.watch(info, pilot.drone)
        track.add(info, hit=hit)

    for label, learning in runs:
        print(f"\n--- {label} ---")
        laps = approach_laps(cfg, laps=args.laps, learning=learning, seed=args.seed,
                             on_lap=lambda lap, _p: print("  " + lap.line()),
                             on_tick=on_tick if (track is not None and learning) else None)
        s = summaries[label] = summarise(laps)
        print(f"  closest approach {s['closest_first_m']:.2f} m -> {s['closest_last_m']:.2f} m"
              f" | collisions {s['collisions_first']} -> {s['collisions_last']}"
              f" | MBON {s['mbon_first_hz']:.1f} Hz -> {s['mbon_last_hz']:.1f} Hz"
              f" | avoidance turn {s['turn_first_hz']:.1f} Hz -> {s['turn_last_hz']:.1f} Hz"
              f" | memory {s['memory']:.2f}")
        if args.csv:
            _write_log(args.csv if len(runs) == 1 else f"{label.split()[0]}-{args.csv}", [lap.as_dict() for lap in laps])
    if track is not None:
        track.save(args.track)
        print(f"track -> {args.track} ({track.summary()}); watch it with tools/record_flight_3d.mjs")
    if args.compare:
        a, b = summaries["learning"], summaries["no learning (control)"]
        print(f"\nthe same brain, the same room, the same seed: {a['closest_last_m']:.2f} m of clearance with the"
              f" mushroom body learning, {b['closest_last_m']:.2f} m without it.")
    return 0


def cmd_radio(args) -> int:
    """Radio Cognitive Fruit Fly: a station that does not stop."""
    from dataclasses import replace

    from .radio import PROGRAMME, STATION, Station

    print(BANNER)
    cfg = _music_cfg(_cfg(args), args)
    programme = list(PROGRAMME)
    if args.minutes_per_show:
        programme = [replace(show, minutes=args.minutes_per_show) for show in programme]
    if args.show:
        wanted = args.show.lower()
        programme = [s for s in programme if wanted in s.name.lower()] or programme
    station = Station(cfg, targets=args.to, seed=args.seed, programme=programme)
    total = sum(s.minutes for s in programme)
    print(f"{STATION}: {len(programme)} shows, {total:.0f} minutes a round, {station.brain.n_neurons:,} neurons")
    for show in programme:
        print(f"  {show.minutes:4.1f} min  {show.name:<14s} {show.scale}/{show.root}  {show.blurb[:74]}")
    print(station.sinks.describe())
    _wait_for_browser(station.sinks, args.wait)
    seen = [0]
    track = None
    if args.track:
        from .track import TrackRecorder

        track = TrackRecorder(title=STATION, subtitle="a station played by a fly brain")

    def on_tick(st, _frame):
        while seen[0] < len(st.entries):  # the station log is the terminal output
            e = st.entries[seen[0]]
            seen[0] += 1
            print(f"  {int(e.t // 60):3d}:{int(e.t % 60):02d}  {e.kind:<10s} {e.text}")
            if track is not None:
                if e.kind == "show":
                    track.chapter(e.t, st.show.name, st.show.blurb)
                elif e.kind not in ("station", "slow"):
                    track.event(e.t, e.kind, e.text)
        if track is not None and st.last_info is not None:
            track.add(st.last_info)

    station.start()
    on_tick(station, None)
    station.run(seconds=(args.hours * 3600 if args.hours else None), on_tick=on_tick)
    if track is not None:
        track.save(args.track)
        print(f"track -> {args.track} ({track.summary()})")
    c = station.counts
    plural = lambda n, one, many: f"{n} {one if n == 1 else many}"  # noqa: E731
    print(f"\noff air after {station.t / 60:.1f} minutes: {plural(c['shows'], 'show', 'shows')}, "
          f"{plural(c['escapes'], 'escape', 'escapes')}, {plural(c['lessons'], 'thing learned', 'things learned')}, "
          f"{plural(c['collisions'], 'bump', 'bumps')}, {plural(c['packs'], 'pack swap', 'pack swaps')}")
    return 0


def _make_probe(args):
    """Build an LTE probe from the --link flags, or None."""
    from .link import LinkProbe, make_router

    if not getattr(args, "link", None):
        return None
    kw = {}
    if args.link == "mock":
        kw["seconds_to_edge"] = float(getattr(args, "link_mock_seconds", 60) or 60)
    router = make_router(args.link, getattr(args, "link_host", None), **kw)
    target = None
    if getattr(args, "link_target", None):
        host, _, port = args.link_target.partition(":")
        target = (host, int(port or 80))
    return LinkProbe(router, target=target, period_s=float(getattr(args, "link_period", 1.0)))


def cmd_link(args) -> int:
    """Watch an Orange LTE router: what the modem sees, and what the packets see."""
    from .link import ORANGE_ROUTERS, LinkError

    if args.list:
        print("routers this knows about:")
        for name, (kind, host, label) in ORANGE_ROUTERS.items():
            print(f"  {name:<18s} {kind:<7s} {host:<14s} {label}")
        print(f"  {'huawei':<18s} {'huawei':<7s} {'192.168.8.1':<14s} any other HiLink box")
        print(f"  {'zte':<18s} {'zte':<7s} {'192.168.0.1':<14s} any other ZTE box")
        print(f"  {'mock':<18s} {'mock':<7s} {'-':<14s} a scripted drive out of coverage, for trying this")
        return 0
    print(BANNER)
    try:
        probe = _make_probe(args)
    except LinkError as e:
        print(f"no router: {e}", file=sys.stderr)
        return 2
    if probe is None:
        print("nothing to probe: pass --link orange-airbox (or --link mock, or --list)", file=sys.stderr)
        return 2
    print(f"probing {probe.router.label}"
          + (f", round trip to {probe.pinger.host}:{probe.pinger.port}" if probe.pinger else ", no round-trip target")
          + f", every {probe.period_s:g}s")
    if not args.json:
        print(f"{'time':>6s}  {'score':>5s} {'grade':<9s} link")
    rows, t0 = [], time.monotonic()
    try:
        while args.seconds <= 0 or time.monotonic() - t0 < args.seconds:
            q = probe.sample()
            rows.append(q)
            if args.json:
                print(json.dumps(q.as_dict(), default=str), flush=True)
            else:
                print(f"{time.monotonic() - t0:6.1f}  {q.line()}", flush=True)
            sleep = probe.period_s - (time.monotonic() - t0) % probe.period_s
            time.sleep(max(0.0, min(probe.period_s, sleep)))
    except KeyboardInterrupt:
        print()
    finally:
        probe.stop()
    if rows and not args.json:
        scores = sorted(q.score for q in rows)
        worst = min(rows, key=lambda q: q.score)
        print(f"\n{len(rows)} reading{'' if len(rows) == 1 else 's'}: "
              f"median score {scores[len(scores) // 2]:.2f}, worst {worst.score:.2f}"
              + (f" ({', '.join(worst.reasons)})" if worst.reasons else ""))
        hold = sum(1 for q in rows if q.score < 0.35)
        if hold:
            print(f"{hold} of them are below the hold threshold: a drone on this link would have stopped and hovered")
    if args.csv:
        _write_log(args.csv, [{"t": round(q.t, 2), "score": q.score, "grade": q.grade, "rtt_ms": q.rtt_ms,
                               "loss": q.loss, **{k: v for k, v in q.signal.as_dict().items() if k != "t"}}
                              for q in rows])
    return 0


def cmd_calibrate(args) -> int:
    from .calibrate import calibrate

    cfg = _cfg(args)
    brain = _brain(cfg)
    calibrate(brain, cfg, args.out)
    return 0


def cmd_bench(args) -> int:
    import time

    cfg = _cfg(args)
    brain = _brain(cfg)
    rng = np.random.default_rng(0)
    inputs = {k: rng.random(brain.connectome.group(k).size) * 60 for k in brain.input_specs}
    brain.tick(inputs, 100)
    t0 = time.perf_counter()
    brain.tick(inputs, args.ms)
    wall = time.perf_counter() - t0
    st = brain.net.stats
    print(f"{args.ms:.0f} ms of brain time in {wall * 1000:.0f} ms wall -> {args.ms / 1000 / wall:.2f}x real time")
    print(f"dt={brain.net.p.dt} ms, spikes so far {st.spikes:,}, synaptic events {st.synaptic_events:,}")
    return 0


# ------------------------------------------------------------------- music
def _music_cfg(cfg, args) -> dict:
    """CLI flags win over defaults.yaml, for the settings worth typing."""
    m = cfg.setdefault("music", {})
    for key in ("scale", "root", "cps", "tempo_from"):
        v = getattr(args, key, None)
        if v is not None:
            m[key] = v
    if getattr(args, "latency", None) is not None:
        m["latency_s"] = args.latency
    if getattr(args, "bpm", None):
        m["cps"] = args.bpm / 60.0 / 4.0  # four beats to a cycle, as Tidal counts them
    return cfg


def _music_start(cfg, brain, targets: str):
    """Build the compositor and the sinks, and say where the music is going."""
    from .music import Compositor, make_sinks
    from .music.events import note_name

    comp = Compositor(cfg, brain=brain)
    sinks = make_sinks(targets, cfg)
    scale = comp.scale
    print(f"compositor: {len(comp.voices)} voices on {scale.name} from {note_name(scale.root)}, "
          f"{comp.cps:.3f} cps ({comp.cps * 60 * 4:.0f} bpm)")
    print(sinks.describe())
    return comp, sinks


class _MusicTap:
    """Hangs a compositor off any flight: ``demo --music pd``, ``fly --music strudel``."""

    def __init__(self, cfg, brain, pilot, targets: str):
        self.comp, self.sinks = _music_start(cfg, brain, targets)
        self.pilot = pilot

    def __call__(self, info) -> None:
        if not self.comp.baselines and self.pilot.decoder.baseline:
            self.comp.set_baselines(self.pilot.decoder.baseline)
        self.sinks.frame(self.comp.tick(info))

    def close(self) -> None:
        self.sinks.close()
        print(self.comp.summary())


def _wait_for_browser(sinks, seconds: float) -> None:
    """A browser cannot join a flight that is already over."""
    from .music.sinks import StrudelSink

    strudel = [s for s in sinks.sinks if isinstance(s, StrudelSink)]
    if not strudel or seconds <= 0 or not sys.stdout.isatty():
        return
    hub = strudel[0].server
    print(f"open {hub.url} and press play — waiting up to {seconds:.0f}s (Ctrl+C to start anyway)")
    t0 = time.monotonic()
    try:
        while hub.clients == 0 and time.monotonic() - t0 < seconds:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
    print(f"{hub.clients} browser(s) listening" if hub.clients else "nobody listening yet; flying anyway")


def _write_scores(args, comp, frames) -> None:
    from .music import to_strudel, to_tidal, transcribe
    from .music.patterns import strudel_live_snippet, tidal_live_file

    if getattr(args, "score", None):
        score = transcribe(frames, steps=args.steps, max_cycles=args.score_cycles)
        path = Path(args.score)
        title = f"{args.seconds:.0f} s of MiniFly" if args.brain is None else f"{args.seconds:.0f} s of {args.brain}"
        sounds = (comp.cfg.get("sounds") or {})
        text = (to_strudel(score, sounds.get("strudel"), title) if path.suffix in (".mjs", ".js")
                else to_tidal(score, sounds.get("superdirt"), title))
        path.write_text(text, encoding="utf-8")
        print(f"score -> {path} ({score.notes} notes over {score.cycles} cycles"
              + (f", {score.dropped} lost to the grid" if score.dropped else "") + ")")
    if getattr(args, "write_patches", None):
        from .music.server import write_assets

        out = Path(args.write_patches)
        written = write_assets(out) + _copy_patches(out)
        voices = [v.spec.name for v in comp.voices]
        controls = sorted({c.name for f in frames[:1] for c in f.controls}) or ["alt", "drive", "loom"]
        (out / "flybrain-live.tidal").write_text(tidal_live_file(voices, controls), encoding="utf-8")
        (out / "flybrain-strudel.mjs").write_text(strudel_live_snippet(voices, controls), encoding="utf-8")
        written += [out / "flybrain-live.tidal", out / "flybrain-strudel.mjs"]
        print("patches -> " + ", ".join(str(w) for w in written))


def _copy_patches(out: Path) -> list[Path]:
    from importlib import resources

    written = []
    for name in ("flybrain.pd", "flybrain-fudi.pd", "flybrain.maxpat"):
        text = resources.files("flydrones.music").joinpath("patches", name).read_text(encoding="utf-8")
        (out / name).write_text(text, encoding="utf-8")
        written.append(out / name)
    return written


def _render_audio(args, frames) -> None:
    """Render a flight to a WAV (or an MP3, if ffmpeg is about) and say what came out."""
    import shutil
    import subprocess

    from .music.synth import SynthConfig, describe, render, write_wav

    if not getattr(args, "render", None) or not frames:
        return
    cfg = _music_cfg(_cfg(args), args)
    out = Path(args.render)
    wav = out.with_suffix(".wav")
    audio = render(frames, cfg)
    write_wav(wav, audio, SynthConfig.from_config(cfg).sample_rate)
    d = describe(audio)
    print(f"audio -> {wav} ({d['seconds']:.0f} s, peak {d['peak']:.2f}, {d['rms_dbfs']:.1f} dBFS RMS, "
          f"centroid {d['centroid_hz']:.0f} Hz, width {d['width']:.2f})")
    if out.suffix.lower() == ".mp3":
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            print(f"  (no ffmpeg on PATH, so it stayed a WAV: {wav})")
            return
        subprocess.run([ffmpeg, "-y", "-i", str(wav), "-codec:a", "libmp3lame", "-q:a", "3", str(out)],
                       check=True, capture_output=True)
        print(f"audio -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


def cmd_compose(args) -> int:
    """Fly and play at the same time, in real time."""
    from .drones import SimDrone
    from .music.conductor import describe, improvisation
    from .runtime import Pilot
    from .senses import ScriptedGestures

    print(BANNER)
    cfg = _music_cfg(_cfg(args), args)
    if args.render and not args.fast:
        args.fast = True  # rendering is offline; there is nothing to keep in step with
        print("--render is offline, so the flight runs as fast as it can")
    if args.replay:
        return _compose_replay(args, cfg)
    brain = _brain(cfg)
    gestures = None
    if args.gestures != "none":
        timeline = improvisation(args.seconds, seed=args.seed, shape=args.shape)
        gestures = ScriptedGestures(timeline)
        print(f"conductor (seed {args.seed}): {describe(timeline)}")
    pilot = Pilot(brain, SimDrone(start=(-1.5, 0.0, 0.0), seed=args.seed), cfg, gestures=gestures)
    comp, sinks = _music_start(cfg, brain, args.to)
    _wait_for_browser(sinks, args.wait)

    hz = float(cfg["control"]["hz"])
    dt = 1.0 / hz
    frames = []
    pilot.drone.connect()
    pilot.warmup(pilot.decoder.settle_s + 0.1, dt)
    comp.set_baselines(pilot.decoder.baseline)
    if cfg["control"].get("takeoff", True):
        pilot.drone.takeoff()
    t0 = time.monotonic()
    behind = False
    try:
        for k in range(int(args.seconds * hz)):
            t = k * dt
            info = pilot.tick(t, dt)
            for _ in range(4):
                pilot.drone.step(dt / 4)
            frame = comp.tick(info)
            frames.append(frame)
            sinks.frame(frame)
            if args.fast:
                continue
            slack = (t0 + (k + 1) * dt) - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            elif slack < -0.5 and not behind:
                behind = True
                print(f"warning: {-slack:.1f}s behind real time (brain at {info.rtf:.2f}x) — the music will drag")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        pilot.drone.land()
        sinks.close()
    print(comp.summary() + f"; {pilot.drone.collisions} collisions")
    _write_scores(args, comp, frames)
    _render_audio(args, frames)
    return 0


def _compose_replay(args, cfg) -> int:
    """Play a saved score again, without the brain. The flight is already in it."""
    from .music import make_sinks, read_jsonl

    frames = read_jsonl(args.replay)
    if args.render:  # rendering a saved score needs no sinks and no clock
        _render_audio(args, frames)
        if args.to in ("", "none"):
            return 0
    sinks = make_sinks(args.to, cfg)
    print(f"replaying {args.replay}: {len(frames)} frames, {sum(len(f.notes) for f in frames)} notes")
    print(sinks.describe())
    _wait_for_browser(sinks, args.wait)
    t0 = time.monotonic() - (frames[0].t if frames else 0.0)
    try:
        for f in frames:
            if not args.fast:
                slack = (t0 + f.t) - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
            sinks.frame(f)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sinks.close()
    return 0


def _write_log(path, rows) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"log -> {path}")


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flydrones", description="Plug a fruit fly connectome into a drone.")
    p.add_argument("--version", action="version", version=f"flydrones {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", help="YAML file overriding defaults")
        sp.add_argument("--brain", help="minifly or path to a built .npz brain")

    sp = sub.add_parser("demo", help="simulated drone + scripted hand gestures")
    common(sp)
    sp.add_argument("--seconds", type=float, default=22)
    sp.add_argument("--record", help="save a GIF of the dashboard")
    sp.add_argument("--every", type=int, default=2, help="record every Nth control tick")
    sp.add_argument("--live", action="store_true", help="show the dashboard in a window (needs OpenCV)")
    sp.add_argument("--log", help="write a CSV flight log")
    sp.add_argument("--music", help="also play it: comma separated targets, see `compose --help`")
    sp.add_argument("--track", help="write a 3D flight track (JSON) for docs/live/replay.html")
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("swarm", help="one connectome, several drone pilots (simulated)")
    common(sp)
    sp.add_argument("--n", type=int, default=3)
    sp.add_argument("--seconds", type=float, default=20)
    sp.add_argument("--record")
    sp.add_argument("--every", type=int, default=2)
    sp.add_argument("--live", action="store_true")
    sp.set_defaults(func=cmd_swarm)

    sp = sub.add_parser("fly", help="fly a real drone (dry run unless --send)")
    common(sp)
    sp.add_argument("--drone", choices=["sim", "tello", "crazyflie", "mavlink", "esp32"], default="tello")
    sp.add_argument("--input", choices=["camera", "gesture", "both"], default="both",
                    help="camera = drone camera optic flow, gesture = webcam hand, both = both")
    sp.add_argument("--send", action="store_true", help="really send commands to the drone")
    sp.add_argument("--seconds", type=float)
    sp.add_argument("--webcam", default="0")
    sp.add_argument("--gestures", default="auto", choices=["auto", "mediapipe", "opencv", "scripted"])
    sp.add_argument("--mavlink", default="udpin:0.0.0.0:14550")
    sp.add_argument("--autopilot", default="ardupilot", choices=["ardupilot", "px4"])
    sp.add_argument("--esp32-host", default="192.168.4.1")
    sp.add_argument("--uri", default="radio://0/80/2M/E7E7E7E7E7")
    sp.add_argument("--live", action="store_true")
    sp.add_argument("--log")
    sp.add_argument("--music", help="also play it: comma separated targets, see `compose --help`")
    sp.add_argument("--track", help="write a 3D flight track (JSON) for docs/live/replay.html")
    sp.add_argument("--link", help="fly on an LTE router and let the governor watch it: orange-airbox, "
                                   "orange-flybox, huawei, zte, mock (see `flydrones link --list`)")
    sp.add_argument("--link-host", dest="link_host")
    sp.add_argument("--link-target", dest="link_target", metavar="HOST:PORT")
    sp.add_argument("--link-period", dest="link_period", type=float, default=1.0)
    sp.add_argument("--link-mock-seconds", dest="link_mock_seconds", type=float, default=60)
    sp.set_defaults(func=cmd_fly)

    sp = sub.add_parser("compose", help="fly and play music at the same time: Pd, Max/MSP, TidalCycles, Strudel",
                        description="Fly a simulated drone with a fly brain and send the result to a live-coding "
                                    "environment. Runs in real time so you can play along.")
    common(sp)
    sp.add_argument("--to", default="print,strudel",
                    help="comma separated targets: pd, pd-fudi, max, tidal, superdirt, strudel, jsonl:FILE, print, none. "
                         "Each takes an optional :port or :host:port (default print,strudel)")
    sp.add_argument("--seconds", type=float, default=120)
    sp.add_argument("--seed", type=int, default=0, help="the conductor's seed: same seed, same gestures")
    sp.add_argument("--gestures", choices=["scripted", "none"], default="scripted",
                    help="scripted = a seeded hand in front of the camera; none = the room alone")
    sp.add_argument("--shape", choices=["flat", "arc"], default="flat",
                    help="arc gives the session a beginning, a middle and an end")
    sp.add_argument("--scale", help="minor_pentatonic, dorian, blues, chromatic, ... (see docs/MUSIC.md)")
    sp.add_argument("--root", help="root note: C3, F#2, Bb4 or a MIDI number")
    sp.add_argument("--cps", type=float, help="cycles per second for Tidal and Strudel")
    sp.add_argument("--bpm", type=float, help="the same thing in beats per minute (4 beats to a cycle)")
    sp.add_argument("--tempo-from", choices=["fixed", "drive", "alt"], dest="tempo_from",
                    help="let the flight move the tempo")
    sp.add_argument("--latency", type=float, help="scheduling headroom in seconds (default 0.2)")
    sp.add_argument("--fast", action="store_true", help="do not pace to wall clock; for writing a score in a hurry")
    sp.add_argument("--wait", type=float, default=20, help="seconds to wait for a browser before taking off")
    sp.add_argument("--score", help="write the flight as mini-notation: .tidal or .mjs")
    sp.add_argument("--steps", type=int, default=16, help="steps per cycle when quantising the score")
    sp.add_argument("--score-cycles", type=int, default=64, dest="score_cycles", help="cycles to keep in the score")
    sp.add_argument("--write-patches", metavar="DIR", dest="write_patches",
                    help="write the Pd and Max patches, the Strudel page and the Tidal file for this configuration")
    sp.add_argument("--replay", metavar="SCORE.jsonl", help="play a score saved with jsonl:FILE instead of flying")
    sp.add_argument("--render", metavar="TRACK.wav",
                    help="render the flight to audio with the built-in synthesiser (.wav, or .mp3 with ffmpeg)")
    sp.set_defaults(func=cmd_compose)

    sp = sub.add_parser("link", help="watch an Orange LTE router: signal, round trip, and what a drone would do")
    sp.add_argument("--link", default="auto", help="auto, orange-airbox, orange-flybox, orange-home-4g, "
                                                   "orange-flybox-zte, huawei, zte, mock")
    sp.add_argument("--link-host", dest="link_host", help="the router's address (default: the box's own)")
    sp.add_argument("--link-target", dest="link_target", metavar="HOST:PORT",
                    help="measure the round trip to this as well, e.g. the drone's control port")
    sp.add_argument("--link-period", dest="link_period", type=float, default=1.0, help="seconds between readings")
    sp.add_argument("--link-mock-seconds", dest="link_mock_seconds", type=float, default=60,
                    help="with --link mock: how long the scripted drive out of coverage takes")
    sp.add_argument("--seconds", type=float, default=0, help="stop after this long (default: until Ctrl+C)")
    sp.add_argument("--json", action="store_true", help="one JSON object per reading")
    sp.add_argument("--csv", help="write the readings to a CSV")
    sp.add_argument("--list", action="store_true", help="list the routers this knows about")
    sp.set_defaults(func=cmd_link)

    sp = sub.add_parser("download", help="download connectome data")
    sp.add_argument("dataset", choices=["malecns"])
    sp.add_argument("--dir", default="data/malecns_v1")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("build-brain", help="build a signed connectome .npz from MaleCNS v1.0")
    sp.add_argument("--data-dir", default="data/malecns_v1")
    sp.add_argument("--config")
    sp.add_argument("--min-synapses", type=int, default=3)
    sp.add_argument("--core-hops", type=int, default=0, help="keep only neurons within N synapses of inputs and outputs")
    sp.add_argument("--max-neurons", type=int)
    sp.add_argument("--out", default="data/malecns_brain.npz")
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("inspect", help="list input/output groups and poke each input")
    common(sp)
    sp.add_argument("--hz", type=float, default=100)
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("radio", help="Radio Cognitive Fruit Fly: an always-on station played by the brain",
                        description="A programme of shows flown and composed live: the fly hovers, learns, "
                                    "holds a heading and escapes, and the music follows what it is doing.")
    common(sp)
    sp.add_argument("--to", default="strudel", help="where the music goes (see `compose --help`)")
    sp.add_argument("--hours", type=float, help="stop after this long (default: until Ctrl+C)")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--minutes-per-show", type=float, dest="minutes_per_show",
                    help="override every show's length, for a quick listen")
    sp.add_argument("--show", help="play only the show whose name contains this")
    sp.add_argument("--scale")
    sp.add_argument("--root")
    sp.add_argument("--latency", type=float)
    sp.add_argument("--wait", type=float, default=30, help="seconds to wait for a browser before going on air")
    sp.add_argument("--track", help="write a 3D flight track (JSON) for docs/live/replay.html")
    sp.set_defaults(func=cmd_radio)

    sp = sub.add_parser("learn", help="watch the mushroom body learn to avoid the chair")
    common(sp)
    sp.add_argument("--laps", type=int, default=10, help="approaches to fly")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--no-learning", action="store_true", dest="no_learning",
                    help="same flight with the plasticity switched off")
    sp.add_argument("--compare", action="store_true", help="fly it both ways and print the two side by side")
    sp.add_argument("--csv", help="write the laps to a CSV")
    sp.add_argument("--track", help="write a 3D flight track (JSON) for docs/live/replay.html")
    sp.set_defaults(func=cmd_learn)

    sp = sub.add_parser("calibrate", help="fit the descending-neuron read-out for this brain")
    common(sp)
    sp.add_argument("--out", default="readout.json")
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("bench", help="measure simulation speed")
    common(sp)
    sp.add_argument("--ms", type=float, default=1000)
    sp.set_defaults(func=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
