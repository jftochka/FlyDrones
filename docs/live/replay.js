// Watch a flight that was flown in Python.
//
// The demo next door flies its own copy of the brain in JavaScript. This page
// does not fly anything: it loads a track written by `flydrones learn --track`
// (or `radio --track`, or `demo --track`) and renders it in the same bedroom,
// with the whole path drawn at once and the brain's state along the bottom.
// That is the point of it — the mushroom body and the compass only exist on the
// Python side, so the only way to see them in 3D is to replay them.
//
//   replay.html?track=./tracks/learn.json&theme=night&view=orbit
//   replay.html?record            deterministic stepping for tools/record_flight_3d.mjs
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { createDrone, createFly } from "./models.js";
import { createRoom } from "./room.js";

const qs = new URLSearchParams(location.search);
const RECORD = qs.has("record");
const $ = (id) => document.getElementById(id);
const V = (x, y, z) => new THREE.Vector3(x, z, -y); // sim (z up) -> three (y up)
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const KIND_COLOR = { escape: 0xffb020, bump: 0xff4d8d, learning: 0x39ff88, forgetting: 0x6f7f97, goal: 0xb388ff, show: 0xffb020, lap: 0x4cc9f0 };

// ===================================================================== track
const track = await (async () => {
  const url = new URL(qs.get("track") || "./tracks/learn.json", import.meta.url);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`no track at ${url} (${r.status}) — write one with: flydrones learn --track docs/live/tracks/learn.json`);
  return r.json();
})().catch((e) => { $("err").style.display = "grid"; $("err").textContent = String(e.message || e); throw e; });

const F = Object.fromEntries(track.fields.map((name, i) => [name, i]));
const rows = track.frames;
const T0 = rows[0][F.t];
const DURATION = rows[rows.length - 1][F.t] - T0;
const at = (i, name) => rows[clamp(i, 0, rows.length - 1)][F[name]];

$("t-title").textContent = track.title || "FlyDrones";
$("t-sub").textContent = track.subtitle || "";

/** Linear interpolation between the two frames around `t` (angles wrap). */
function sample(t) {
  const x = clamp(t, 0, DURATION) + T0;
  let lo = 0, hi = rows.length - 1;
  while (lo < hi - 1) { const mid = (lo + hi) >> 1; if (rows[mid][F.t] <= x) lo = mid; else hi = mid; }
  const a = rows[lo], b = rows[hi], span = b[F.t] - a[F.t];
  const k = span > 1e-6 ? clamp((x - a[F.t]) / span, 0, 1) : 0;
  const mix = (name) => a[F[name]] + (b[F[name]] - a[F[name]]) * k;
  const wrap = (name) => { const d = ((b[F[name]] - a[F[name]] + 540) % 360) - 180; return a[F[name]] + d * k; };
  return {
    i: lo, t: x - T0,
    x: mix("x"), y: mix("y"), z: mix("z"), yaw: wrap("yaw"),
    throttle: mix("throttle"), yaw_cmd: mix("yaw_cmd"), forward: mix("forward"),
    escape: a[F.escape] > 0, hit: a[F.hit] > 0,
    memory: mix("memory"), heading: a[F.heading] < 0 ? null : wrap("heading"),
    goal: a[F.goal] < 0 ? null : a[F.goal], mbon: mix("mbon"), kc: mix("kc"),
    loom: mix("loom"), drive: mix("drive"),
  };
}

// ================================================================== three.js
const canvas = $("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: RECORD });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const scene = new THREE.Scene();
{
  const pm = new THREE.PMREMGenerator(renderer);
  scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.45;
}
const camera = new THREE.PerspectiveCamera(45, 1, 0.02, 60);
camera.position.set(-4.4, 2.6, 3.4);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 0.8; controls.maxDistance = 14; controls.target.set(-0.4, 1, 0);
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(512, 512), 0.3, 0.3, 0.9);
composer.addPass(bloom); composer.addPass(new OutputPass());

const room = createRoom(scene);
const droneRoot = new THREE.Group(); scene.add(droneRoot);
const drone = createDrone(); drone.group.scale.setScalar(1.9); droneRoot.add(drone.group);
const fly = createFly(); drone.group.add(fly.group);
droneRoot.traverse((o) => { if (o.isMesh) o.castShadow = true; });

let theme = qs.get("theme") === "day" ? "day" : "night";
function applyTheme() {
  room.setTheme(theme, { scene, bloom, renderer });
  scene.environmentIntensity = theme === "night" ? 0.12 : 0.45;
  if (theme === "night") { bloom.threshold = 0.42; bloom.strength = 0.75; } else { bloom.threshold = 0.9; bloom.strength = 0.3; }
  drone.setGlow(theme === "night" ? 1 : 0.35);
  document.body.classList.toggle("day", theme === "day");
  $("theme").textContent = theme === "night" ? "day" : "night";
}

// ================================================= the path, drawn all at once
// Blue where the fly had learned nothing, green where it had. Two copies: the
// whole flight, faint, so the shape of every approach is visible from the first
// frame, and the part already flown, bright, drawn over it.
// A flight is not always continuous: the learning experiment puts the drone
// back at the start between approaches, and a straight line across the room
// from the end of one to the start of the next is a lie about where it flew.
// So the path is drawn as segments, and a jump longer than half a metre in one
// frame is simply not drawn.
const JUMP = 0.6;
const cold = new THREE.Color(0x4cc9f0), warm = new THREE.Color(0x39ff88), tmpC = new THREE.Color();
const point = (i) => V(at(i, "x"), at(i, "y"), at(i, "z") + 0.2);
// Colour the path against this flight's own memory, not an absolute scale: a
// fly that learned a little still went from knowing nothing to knowing that.
const maxMemory = Math.max(0.02, ...rows.map((r) => r[F.memory]));
const learned = (i) => clamp(at(i, "memory") / maxMemory, 0, 1);

// Continuous stretches of flying. A jump longer than half a metre in one frame
// is the experiment putting the drone back at the start, not a flight path.
const runs = [];
{
  let from = 0;
  for (let i = 1; i < rows.length; i++) {
    if (point(i - 1).distanceTo(point(i)) > JUMP) {
      if (i - 1 - from > 3) runs.push([from, i - 1]);
      from = i;
    }
  }
  if (rows.length - 1 - from > 3) runs.push([from, rows.length - 1]);
}

// The path as a tube rather than a line: a line is one pixel wide whatever the
// screen is, and one pixel of pale green over a wooden floor is not an
// illustration of anything.
const RADIAL = 6;
function tube(from, to) {
  const pts = [];
  for (let i = from; i <= to; i += 2) pts.push(point(i));
  if (pts.length < 2) return null;
  const geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), Math.max(8, pts.length * 2), 0.022, RADIAL, false);
  const count = geo.attributes.position.count;
  const colors = new Float32Array(count * 3);
  const perRing = RADIAL + 1;
  for (let v = 0; v < count; v++) {
    const k = Math.floor(v / perRing) / Math.max(1, count / perRing - 1);  // 0..1 along the tube
    const frame = Math.round(from + k * (to - from));
    tmpC.copy(cold).lerp(warm, learned(frame));
    colors.set([tmpC.r, tmpC.g, tmpC.b], v * 3);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return geo;
}
const paths = [];
for (const [from, to] of runs) {
  const geo = tube(from, to);
  if (!geo) continue;
  const ghost = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.22, depthWrite: false }));
  const flownGeo = geo.clone();
  const flown = new THREE.Mesh(flownGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95, depthWrite: false }));
  scene.add(ghost); scene.add(flown);
  paths.push({ from, to, flownGeo, total: flownGeo.index.count });
}
// and its shadow on the floor, which is what makes the height readable
const floorPts = [];
for (const [from, to] of runs) {
  for (let i = from + 1; i <= to; i++) {
    const a = point(i - 1), b = point(i);
    floorPts.push(a.x, 0.012, a.z, b.x, 0.012, b.z);
  }
}
{
  const g = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(floorPts, 3));
  scene.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x4cc9f0, transparent: true, opacity: 0.22, depthWrite: false })));
}

// a drop line under the drone: without one, height in a still frame is a guess
const dropGeo = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(6), 3));
const drop = new THREE.Line(dropGeo, new THREE.LineBasicMaterial({ color: 0x4cc9f0, transparent: true, opacity: 0.35 }));
scene.add(drop);
const shadowRing = new THREE.Mesh(new THREE.RingGeometry(0.07, 0.11, 32),
  new THREE.MeshBasicMaterial({ color: 0x4cc9f0, transparent: true, opacity: 0.4, side: THREE.DoubleSide, depthWrite: false }));
shadowRing.rotation.x = -Math.PI / 2; scene.add(shadowRing);

// ------------------------------------------------------------------- markers
function labelSprite(text, color, size = 0.5) {
  const pad = 16, font = "600 40px ui-monospace, Menlo, monospace";
  const probe = document.createElement("canvas").getContext("2d");
  probe.font = font;
  const w = Math.ceil(probe.measureText(text).width) + pad * 2, h = 72;
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const g = cv.getContext("2d");
  g.fillStyle = "rgba(6,10,16,.78)"; g.strokeStyle = color; g.lineWidth = 3;
  g.beginPath(); g.roundRect(2, 2, w - 4, h - 4, 14); g.fill(); g.stroke();
  g.font = font; g.fillStyle = color; g.textBaseline = "middle"; g.fillText(text, pad, h / 2 + 2);
  const tx = new THREE.CanvasTexture(cv); tx.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tx, transparent: true, depthWrite: false, depthTest: false }));
  s.scale.set((size * w) / h, size, 1);
  return s;
}
const markers = (track.events || []).map((e) => {
  const s = sample(e.t - T0);
  const color = KIND_COLOR[e.kind] ?? 0xffffff;
  const g = new THREE.Group();
  g.position.copy(V(s.x, s.y, s.z + 0.2));
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.16, 0.012, 10, 40),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85, depthWrite: false }));
  ring.rotation.x = -Math.PI / 2; g.add(ring);
  const label = labelSprite(e.text, `#${color.toString(16).padStart(6, "0")}`, 0.17);
  label.position.y = 0.3; label.material.opacity = 0; g.add(label);
  g.visible = false; scene.add(g);
  return { ...e, at: e.t - T0, group: g, ring, label };
});
const chairLabel = labelSprite("the chair", "#4cc9f0", 0.2);
chairLabel.position.set(1.45, 1.35, 0); scene.add(chairLabel);

// ================================================================== playback
let time = 0, playing = !RECORD, speed = 1, view = qs.get("view") || "orbit";
const chapters = (track.chapters || []).map((ch) => ({ ...ch, at: ch.t - T0 }));

function chapterAt(t) {
  let out = null;
  for (const ch of chapters) if (t >= ch.at - 1e-6) out = ch;
  return out;
}

function updateScene(dt) {
  const s = sample(time);
  droneRoot.position.copy(V(s.x, s.y, s.z + 0.2));
  droneRoot.rotation.set(0, THREE.MathUtils.degToRad(s.yaw), 0);
  drone.group.rotation.z = -s.forward * 0.25;
  drone.group.rotation.x = s.yaw_cmd * 0.12;
  drone.update(dt, { flying: s.z > 0.05, throttle: s.throttle, escape: s.escape, hit: s.hit ? 1 : 0, time });
  fly.update(dt, { flying: s.z > 0.05, escape: s.escape, lookYaw: 0 });
  room.highlight("chair", s.loom);
  room.update(camera, dt);

  // the flown part of the path, and where the drone is over the floor
  for (const p of paths) {
    const k = s.i <= p.from ? 0 : s.i >= p.to ? 1 : (s.i - p.from) / (p.to - p.from);
    p.flownGeo.setDrawRange(0, Math.round(p.total * k / 6) * 6);
  }
  const dp = drop.geometry.attributes.position;
  dp.setXYZ(0, droneRoot.position.x, 0.012, droneRoot.position.z);
  dp.setXYZ(1, droneRoot.position.x, droneRoot.position.y, droneRoot.position.z);
  dp.needsUpdate = true;
  shadowRing.position.set(droneRoot.position.x, 0.014, droneRoot.position.z);

  // markers wake up as the playhead reaches them and fade a few seconds later
  let caption = null;
  for (const m of markers) {
    const age = time - m.at;
    m.group.visible = age > -0.01;
    if (age >= 0) {
      const fade = clamp(1 - (age - 2.6) / 1.4, 0.18, 1);
      m.ring.material.opacity = 0.85 * fade;
      m.ring.scale.setScalar(1 + Math.max(0, 1.6 - age) * 1.4);
      m.label.material.opacity = 0;
      if (age < 2.6 && (!caption || m.at > caption.at)) caption = m;
    }
  }
  // Only the newest event keeps its label in the room: two things that happened
  // a second apart happen in almost the same place, and two labels on top of
  // each other are worse than none.
  if (caption) caption.label.material.opacity = clamp(2.4 - (time - caption.at), 0, 1);
  const cap = $("caption");
  cap.textContent = caption ? caption.text : "";
  cap.style.opacity = caption ? 1 : 0;
  cap.style.color = caption ? `#${(KIND_COLOR[caption.kind] ?? 0xffffff).toString(16).padStart(6, "0")}` : "#fff";

  const ch = chapterAt(time);
  $("chapter").textContent = ch ? (ch.text ? `${ch.name} — ${ch.text}` : ch.name) : "";

  // the brain panel
  const set = (name, value, text, scale = 1) => {
    $(`v-${name}`).textContent = text;
    const bar = $(`b-${name}`);
    if (bar) bar.style.width = `${clamp(value * scale, 0, 1) * 100}%`;
  };
  set("memory", s.memory, s.memory.toFixed(2));
  set("mbon", s.mbon / 120, Math.round(s.mbon));
  set("drive", s.drive, s.drive.toFixed(2));
  set("loom", s.loom, s.loom.toFixed(2));
  set("alt", s.z / 2.2, `${s.z.toFixed(2)} m`);
  $("v-heading").textContent = s.heading == null ? "—" : `${Math.round(s.heading)}°`;
  if (s.heading != null) $("needle").setAttribute("transform", `rotate(${s.heading})`);
  $("goalNeedle").setAttribute("opacity", s.goal == null ? 0 : 1);
  if (s.goal != null) $("goalNeedle").setAttribute("transform", `rotate(${s.goal})`);

  // camera
  const target = droneRoot.position.clone();
  if (view === "chase") {
    const yaw = THREE.MathUtils.degToRad(s.yaw);
    const back = new THREE.Vector3(-Math.cos(yaw) * 1.6, 0.6, Math.sin(yaw) * 1.6);
    camera.position.lerp(target.clone().add(back), RECORD ? 0.25 : 0.08);
    camera.lookAt(target.clone().add(new THREE.Vector3(Math.cos(yaw) * 0.6, 0, -Math.sin(yaw) * 0.6)));
  } else if (view === "drone") {
    const yaw = THREE.MathUtils.degToRad(s.yaw);
    const eye = drone.camera.getWorldPosition(new THREE.Vector3());
    camera.position.copy(eye).add(new THREE.Vector3(Math.cos(yaw) * 0.12, 0, -Math.sin(yaw) * 0.12));
    camera.lookAt(camera.position.clone().add(new THREE.Vector3(Math.cos(yaw), -0.05, -Math.sin(yaw))));
  } else if (RECORD) {
    // a slow orbit that keeps both the drone and the middle of the room in shot
    const ang = parseFloat(qs.get("ang") || "1.9") + time * parseFloat(qs.get("spin") || "0.006");
    const rad = parseFloat(qs.get("rad") || "4.8"), h = parseFloat(qs.get("h") || "2.7");
    // Mostly the room, a little the drone: a camera that chases hard turns the
    // path into a tangle around the middle of the frame.
    const tg = new THREE.Vector3(-0.2, 1.05, 0).lerp(target, 0.3);
    camera.position.set(tg.x + Math.cos(ang) * rad, h, tg.z + Math.sin(ang) * rad);
    camera.lookAt(tg);
  } else {
    controls.target.lerp(target.clone().lerp(new THREE.Vector3(0, 1, 0), 0.35), 0.05);
    controls.update();
  }
  fly.group.visible = view !== "drone";

  const mm = (x) => `${Math.floor(x / 60)}:${String(Math.floor(x % 60)).padStart(2, "0")}`;
  $("clock").textContent = `${mm(time)} / ${mm(DURATION)}`;
  $("seek").value = String(Math.round((time / DURATION) * 1000));
}

// ------------------------------------------------------------------ controls
$("play").addEventListener("click", () => { playing = !playing; $("play").textContent = playing ? "❚❚" : "▶"; });
$("seek").addEventListener("input", (e) => { time = (Number(e.target.value) / 1000) * DURATION; playing = false; $("play").textContent = "▶"; });
const SPEEDS = [1, 2, 4, 0.5];
$("speed").addEventListener("click", () => {
  speed = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length];
  $("speed").textContent = `${speed}×`;
});
$("view").addEventListener("click", () => {
  view = view === "orbit" ? "chase" : view === "chase" ? "drone" : "orbit";
  $("view").textContent = view;
});
$("theme").addEventListener("click", () => { theme = theme === "night" ? "day" : "night"; applyTheme(); });
window.addEventListener("keydown", (e) => {
  if (e.key === " ") { $("play").click(); e.preventDefault(); }
  if (e.key === "ArrowRight") time = Math.min(DURATION, time + 2);
  if (e.key === "ArrowLeft") time = Math.max(0, time - 2);
});
function resize() {
  const w = canvas.clientWidth || window.innerWidth, h = canvas.clientHeight || window.innerHeight;
  renderer.setSize(w, h, false); composer.setSize(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

applyTheme();
resize();
updateScene(0.001);
composer.render();

if (RECORD) {
  document.body.classList.add("rec");
  window.FDR = {
    ready: true,
    duration: DURATION,
    seek(t) { time = clamp(t, 0, DURATION); },
    frame(dt = 1 / 30) { updateScene(dt); composer.render(); },
    setView(v) { view = v; },
    setTheme(t) { theme = t; applyTheme(); },
    events: track.events || [],
    chapters: track.chapters || [],
  };
} else {
  $("play").textContent = playing ? "❚❚" : "▶";
  let prev = performance.now();
  const loop = (now) => {
    const dt = Math.max(0, Math.min(0.1, (now - prev) / 1000)); prev = now;
    if (playing) {
      time += dt * speed;
      if (time > DURATION) time = 0;  // a flight is a loop here, like a GIF
    }
    updateScene(dt);
    composer.render();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
