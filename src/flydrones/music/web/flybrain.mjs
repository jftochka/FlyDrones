// FlyDrones -> Strudel: a live fly brain as pattern values.
//
// `flydrones compose --to strudel` serves this file next to an /events stream
// that carries one JSON frame per control tick: the notes the compositor made
// and the continuous controls it measured. Strudel keeps the clock; the fly
// supplies the values, which is the division of labour a live coder wants.
//
//   const fly = connectFly();            // same origin, or pass a full URL
//   note(fly.sig('wing_left')).s('triangle').gain(fly.ctl('drive'))
//
// Works as an ES module and as a plain <script type="module"> import; it also
// parks itself on globalThis.fly so a REPL can reach it without a binding.

const clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);

export class Fly {
  constructor() {
    this.controls = {};
    this.last = {};          // voice -> last note event
    this.recent = {};        // voice -> rolling array of the last 16 notes
    this.section = 'ground';
    this.cps = 0.5;
    this.cycle = 0;
    this.t = 0;
    this.frames = 0;
    this.connected = false;
    this._handlers = [];
  }

  // ---- reading the fly ---------------------------------------------------
  c(name, dflt = 0) { const v = this.controls[name]; return Number.isFinite(v) ? v : dflt; }
  n(voice, dflt = 60) { const e = this.last[voice]; return e ? e.note : dflt; }
  vel(voice, dflt = 0.8) { const e = this.last[voice]; return e ? e.velocity : dflt; }
  age(voice) { const e = this.last[voice]; return e ? (performance.now() - e.at) / 1000 : Infinity; }
  gate(voice, seconds = 0.15) { return this.age(voice) < seconds ? 1 : 0; }
  notes(voice, n = 4, dflt = 60) {
    const a = this.recent[voice] || [];
    return a.length ? a.slice(-n).map((e) => e.note) : [dflt];
  }
  onNote(fn) { this._handlers.push(fn); return this; }

  // ---- the same three things, as Strudel patterns -----------------------
  // `signal` comes from Strudel. It is a global once initStrudel() has run,
  // and in scope inside evaluated pattern code; useStrudel covers the case
  // where it is neither (a bundler, or an import that did not touch globals).
  useStrudel(scope = {}) { this._signal = scope.signal || globalThis.signal; return this; }
  _sig(fn) {
    const signal = this._signal || globalThis.signal;
    if (!signal) throw new Error('Strudel is not loaded yet: call initStrudel(), or fly.useStrudel({ signal })');
    return signal(fn);
  }
  sig(voice, dflt = 60) { return this._sig(() => this.n(voice, dflt)); }
  ctl(name, dflt = 0) { return this._sig(() => this.c(name, dflt)); }
  gated(voice, seconds = 0.15) { return this._sig(() => this.gate(voice, seconds)); }

  // ---- the feed ----------------------------------------------------------
  apply(frame) {
    this.frames += 1;
    this.t = frame.t ?? this.t;
    this.cps = frame.cps ?? this.cps;
    this.cycle = frame.cycle ?? this.cycle;
    this.section = frame.section ?? this.section;
    Object.assign(this.controls, frame.controls || {});
    for (const e of frame.notes || []) {
      const ev = { ...e, at: performance.now() };
      this.last[e.voice] = ev;
      const a = (this.recent[e.voice] = this.recent[e.voice] || []);
      a.push(ev);
      if (a.length > 16) a.shift();
      for (const fn of this._handlers) { try { fn(ev); } catch (err) { console.error(err); } }
    }
  }
}

export function connectFly(url = '/events', { fly = new Fly(), onStatus = () => {} } = {}) {
  const open = () => {
    const es = new EventSource(url);
    es.onopen = () => { fly.connected = true; onStatus('connected'); };
    es.onmessage = (m) => { try { fly.apply(JSON.parse(m.data)); } catch (e) { console.error(e); } };
    es.onerror = () => {
      fly.connected = false;
      onStatus('reconnecting');  // EventSource retries on its own; say so and let it
    };
    fly.source = es;
  };
  open();
  globalThis.fly = fly;
  return fly;
}

// Handy for meters and colours: 0..1 from a control that may be -1..1.
export const unipolar = (x) => clamp((x + 1) / 2, 0, 1);

export default connectFly;
