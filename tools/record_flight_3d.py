"""Record the 3D replay of a flight to a GIF (and an MP4).

    flydrones learn --laps 8 --track docs/live/tracks/learn.json
    python tools/record_flight_3d.py --out assets/flight3d.gif

It serves ``docs/`` over HTTP, opens ``live/replay.html?record`` in headless
Chromium and steps the replay itself — seek, render, screenshot — so the result
is the same however slowly the machine draws it. Frames are encoded with the
ffmpeg that ships with Playwright's browsers if there is no other one.

Needs ``pip install playwright && playwright install chromium``; nothing else in
FlyDrones depends on it, which is why it is a tool and not a package extra.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def serve(directory: Path) -> tuple[int, socketserver.TCPServer]:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *_a, **_k: None  # type: ignore[assignment]
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1], httpd


def find_ffmpeg() -> str | None:
    """A real ffmpeg, or nothing.

    Not the one Playwright ships with its browsers: that build is configured
    with --disable-everything and can neither read a PNG sequence nor write a
    GIF, so pointing at it only produces a confusing error later.
    """
    return shutil.which("ffmpeg")


def chromium_path() -> str | None:
    for pattern in ("/opt/pw-browsers/chromium-*/chrome-linux/chrome",
                    "~/.cache/ms-playwright/chromium-*/chrome-linux/chrome"):
        for path in sorted(glob(str(Path(pattern).expanduser()))):
            return path
    return None


def gif_with_pillow(frames: list[Path], out: Path, fps: int, width: int) -> None:
    from PIL import Image

    images = []
    for path in frames:
        im = Image.open(path).convert("RGB")
        if width and im.width != width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        images.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0, optimize=True)


def record(args) -> int:
    from playwright.sync_api import sync_playwright

    track_path = Path(args.track)
    if not track_path.exists():
        print(f"no track at {track_path}. Write one first, e.g.\n"
              f"  flydrones learn --laps 8 --track {track_path}", file=sys.stderr)
        return 2
    track = json.loads(track_path.read_text(encoding="utf-8"))
    rel = track_path.resolve().relative_to(DOCS.resolve()) if DOCS.resolve() in track_path.resolve().parents else None
    port, httpd = serve(DOCS)
    query = (f"?record&theme={args.theme}&view={args.view}&rad={args.radius}&h={args.height_m}"
             f"&spin={args.spin}&ang={args.angle}")
    if rel is not None:
        query += f"&track=/{rel.as_posix()}"
    url = f"http://127.0.0.1:{port}/live/replay.html{query}"

    frames_dir = Path(tempfile.mkdtemp(prefix="flydrones-3d-"))
    errors: list[str] = []
    try:
        with sync_playwright() as pw:
            launch = {"args": ["--no-sandbox", "--use-gl=swiftshader", "--enable-unsafe-swiftshader",
                               "--hide-scrollbars", "--mute-audio"]}
            exe = chromium_path()
            if exe:
                launch["executable_path"] = exe
            browser = pw.chromium.launch(**launch)
            page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                    device_scale_factor=args.scale)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(url, wait_until="load")
            page.wait_for_function("window.FDR && window.FDR.ready === true", timeout=60_000)
            duration = float(page.evaluate("window.FDR.duration"))
            start = max(0.0, args.start)
            end = min(duration, args.end if args.end > 0 else duration)
            dt = args.speed / args.fps
            count = max(1, int((end - start) / dt))
            print(f"{track.get('title', track_path.name)}: {duration:.1f} s of flight, "
                  f"recording {start:.0f}-{end:.0f} s at {args.speed}x -> {count} frames")
            page.evaluate("window.FDR.seek(0)")
            for _ in range(8):  # let the scene settle before the first kept frame
                page.evaluate("window.FDR.frame(0.02)")
            for i in range(count):
                t = start + i * dt
                page.evaluate("(t) => window.FDR.seek(t)", t)
                page.evaluate("(dt) => window.FDR.frame(dt)", dt)
                page.screenshot(path=str(frames_dir / f"f{i:05d}.png"), animations="disabled")
                if i % 25 == 0:
                    print(f"  {i}/{count}  t={t:6.1f}s", flush=True)
            browser.close()
    finally:
        httpd.shutdown()
    if errors:
        print("page errors: " + "; ".join(errors[:3]), file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        # Pillow is already a FlyDrones dependency, so this always works; ffmpeg
        # only buys a better palette and the mp4.
        print("no ffmpeg on PATH — writing the GIF with Pillow instead")
        gif_with_pillow(sorted(frames_dir.glob("f*.png")), out, args.fps, args.gif_width)
        print(f"{out} ({out.stat().st_size / 1e6:.1f} MB, {count} frames at {args.fps} fps)")
        if not args.keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)
        return 0
    pattern = str(frames_dir / "f%05d.png")
    palette = frames_dir / "palette.png"
    scale = f"scale={args.gif_width}:-1:flags=lanczos"
    run = lambda cmd: subprocess.run(cmd, check=True, capture_output=True)  # noqa: E731
    run([ffmpeg, "-y", "-i", pattern, "-vf", f"{scale},palettegen=stats_mode=diff:max_colors={args.colors}", str(palette)])
    run([ffmpeg, "-y", "-framerate", str(args.fps), "-i", pattern, "-i", str(palette),
         "-lavfi", f"{scale}[x];[x][1:v]paletteuse=dither={args.dither}:diff_mode=rectangle",
         "-loop", "0", str(out)])
    megabytes = out.stat().st_size / 1e6
    print(f"{out} ({megabytes:.1f} MB, {count} frames at {args.fps} fps)")
    if megabytes > 10:
        print("  that is a big GIF. A 3D room changes every pixel every frame, so the usual cures are\n"
              "  --speed (fewer frames for the same flight), --gif-width, --colors — or --mp4, which is\n"
              "  about a tenth of the size for the same footage.")
    if args.mp4:
        mp4 = out.with_suffix(".mp4")
        run([ffmpeg, "-y", "-framerate", str(args.fps), "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-crf", "20", str(mp4)])
        print(f"{mp4} ({mp4.stat().st_size / 1e6:.1f} MB)")
    if not args.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    else:
        print(f"frames kept in {frames_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", default=str(DOCS / "live" / "tracks" / "learn.json"))
    p.add_argument("--out", default=str(ROOT / "assets" / "flight3d.gif"))
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--speed", type=float, default=4.0, help="flight seconds per recorded second")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=0.0, help="0 = to the end of the flight")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--scale", type=float, default=1.0, help="device pixel ratio while rendering")
    p.add_argument("--gif-width", type=int, default=560, dest="gif_width")
    p.add_argument("--colors", type=int, default=64, help="GIF palette size; fewer is a much smaller file")
    p.add_argument("--dither", default="none",
                   help="ffmpeg paletteuse dither: none is smallest, bayer:bayer_scale=3 is smoothest")
    p.add_argument("--theme", default="day", choices=["day", "night"])
    p.add_argument("--view", default="orbit", choices=["orbit", "chase", "drone"])
    p.add_argument("--radius", type=float, default=4.8, help="camera distance in the orbit view")
    p.add_argument("--height-m", type=float, default=2.7, dest="height_m", help="camera height in metres")
    p.add_argument("--angle", type=float, default=1.9, help="starting orbit angle, radians")
    p.add_argument("--spin", type=float, default=0.006, help="orbit speed, radians per flight second")
    p.add_argument("--mp4", action="store_true", help="also write an .mp4 next to the gif")
    p.add_argument("--keep-frames", action="store_true", dest="keep_frames")
    return record(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
