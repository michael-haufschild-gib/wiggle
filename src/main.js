// App shell: loads the embedded photo, builds the rig and the GPU simulation, runs the
// fixed-timestep loop and wires the control panel.
// URL options: ?grid=160 (mesh columns), ?preset=shimmy, ?view=1..6, ?rig, ?paused, ?selftest.
(async function () {
  const W = window.Wiggle;
  const $ = (id) => document.getElementById(id);
  const query = new URLSearchParams(location.search);

  // Visible rectangle of image space: margins left/right/top so sway and bounce stay in frame,
  // and the bottom 56 px of the photo cropped so the cut edge of the thighs never shows.
  const VIEW = { x: -70, y: -50, w: 1164, h: 1530 };

  const DEFAULTS = Object.freeze({
    h: 1 / 960,                   // simulation step (s)
    cs: 950,                      // shear-wave speed of the tissue's attachment (px/s), ~0.7 m/s
    cp: 1000,                     // in-plane wave speed between neighbouring columns (px/s)
    cb: 1500,                     // bulk wave speed for area preservation (px/s), 0 = off
    softness: 1,                  // divides the three wave speeds: 2 = half as stiff
    damping: 1,                   // scales the per-region damping ratios
    viscosity: 25,                // ripple smoothing rate (1/s)
    stretchLimit: 0.45,           // tissue cannot be displaced further than this fraction of its thickness
    sMax: 1,                      // Jacobi relaxation threshold
    kinematicBelow: 8,            // vertices with thinner tissue than this (px) follow the skeleton exactly
    grabHz: 9,                    // stiffness of the pointer grab
    debugMode: 0,
  });
  const params = { ...DEFAULTS };

  function fatal(message) {
    $('fatal').textContent = message;
    $('fatal').hidden = false;
    $('wrap').hidden = true;
  }

  if (!navigator.gpu) {
    fatal('WebGPU is not available in this browser. Open this page in a current Chrome, Edge or Safari.');
    return;
  }

  let sim, rig, bitmap;
  try {
    // The photo is embedded as base64 because a file:// page may not read pixels of images on disk.
    const b64 = window.WIGGLE_IMAGE.slice(window.WIGGLE_IMAGE.indexOf(',') + 1);
    const bytes = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));
    bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }), { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
    const nx = Math.max(48, Math.min(384, parseInt(query.get('grid'), 10) || 160));
    rig = W.Rig.buildRig(bitmap, nx, Math.round(nx * 1.5));
    sim = await W.Sim.create($('gpu'), bitmap, rig, params);
  } catch (err) {
    console.error(err);
    fatal('Could not start WebGPU: ' + err.message);
    return;
  }
  sim.onError = (message) => fatal('WebGPU error: ' + message);
  sim.device.lost.then((info) => fatal('The GPU device was lost: ' + info.message));
  sim.setView(VIEW.x, VIEW.y, VIEW.w, VIEW.h);

  const motion = new W.Motion();
  motion.intensity = 0.8;
  const state = { paused: query.has('paused'), speed: 1, stepOnce: false, substeps: 0, fps: 0 };

  // ---- Controls -------------------------------------------------------------------------
  function slider(host, label, min, max, step, get, set, format) {
    const row = document.createElement('div');
    row.className = 'row';
    const id = 'r-' + label.toLowerCase().replace(/\W+/g, '-');
    row.innerHTML = `<label for="${id}">${label}</label><output></output><input id="${id}" type="range" min="${min}" max="${max}" step="${step}">`;
    const input = row.querySelector('input'), out = row.querySelector('output');
    const show = () => { out.textContent = format(get()); };
    input.value = get();
    input.addEventListener('input', () => { set(parseFloat(input.value)); show(); });
    show();
    $(host).appendChild(row);
    return { sync: () => { input.value = get(); show(); } };
  }
  function segmented(host, options, isActive, choose) {
    const buttons = options.map(([value, label]) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = label;
      b.addEventListener('click', () => { choose(value); sync(); });
      $(host).appendChild(b);
      return [value, b];
    });
    const sync = () => buttons.forEach(([value, b]) => b.setAttribute('aria-pressed', String(isActive(value))));
    sync();
    return { sync };
  }

  const fixed = (digits) => (v) => v.toFixed(digits);
  const PRESETS = [['mix', 'Mix'], ['shimmy', 'Shimmy'], ['hips', 'Hips'], ['bounce', 'Bounce'], ['twist', 'Twist'], ['idle', 'Idle']];
  const presets = segmented('presets', PRESETS, (v) => motion.preset === v, (v) => { motion.preset = v; });
  const sliders = [
    slider('motion-sliders', 'Intensity', 0, 1.5, 0.05, () => motion.intensity, (v) => { motion.intensity = v; }, fixed(2)),
    slider('motion-sliders', 'Tempo', 0.5, 1.5, 0.05, () => motion.tempo, (v) => { motion.tempo = v; }, fixed(2)),
    slider('tissue-sliders', 'Softness', 0.5, 1.6, 0.05, () => params.softness, (v) => { params.softness = v; }, fixed(2)),
    slider('tissue-sliders', 'Damping', 0.3, 3, 0.05, () => params.damping, (v) => { params.damping = v; }, fixed(2)),
    slider('tissue-sliders', 'In-plane stiffness', 400, 2000, 50, () => params.cp, (v) => { params.cp = v; }, (v) => (v / W.Rig.PX_PER_M).toFixed(1) + ' m/s'),
    slider('tissue-sliders', 'Volume preservation', 0, 3000, 100, () => params.cb, (v) => { params.cb = v; }, (v) => (v > 0 ? (v / W.Rig.PX_PER_M).toFixed(1) + ' m/s' : 'off')),
    slider('tissue-sliders', 'Substeps per 1/60 s', 8, 32, 4, () => Math.round(1 / (60 * params.h)), (v) => { params.h = 1 / (60 * v); }, fixed(0)),
  ];
  const speeds = segmented('speeds', [[1, '1×'], [0.5, '½×'], [0.25, '¼×'], [0.1, '⅒×']], (v) => state.speed === v, (v) => { state.speed = v; });

  $('jolt').addEventListener('click', () => motion.jolt());
  $('view').addEventListener('change', (e) => { params.debugMode = parseInt(e.target.value, 10); });
  $('rig').addEventListener('change', drawOverlay);
  $('pause').addEventListener('click', () => { state.paused = !state.paused; $('pause').setAttribute('aria-pressed', String(state.paused)); });
  $('step').addEventListener('click', () => { state.stepOnce = true; });
  $('reset').addEventListener('click', () => { motion.reset(); sim.reset(); });

  if (PRESETS.some(([name]) => name === query.get('preset'))) { motion.preset = query.get('preset'); presets.sync(); }
  if (query.has('view')) { params.debugMode = parseInt(query.get('view'), 10) || 0; $('view').value = String(params.debugMode); }
  if (query.has('rig')) $('rig').checked = true;
  $('pause').setAttribute('aria-pressed', String(state.paused));

  // ---- Layout ---------------------------------------------------------------------------
  const wrap = $('wrap'), canvas = $('gpu'), overlay = $('overlay');
  function fit() {
    const stage = $('stage');
    const aspect = VIEW.w / VIEW.h;
    const h = Math.max(120, Math.min(stage.clientHeight - 32, (stage.clientWidth - 32) / aspect));
    const w = h * aspect;
    wrap.style.width = w + 'px';
    wrap.style.height = h + 'px';
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = overlay.width = Math.max(1, Math.round(w * dpr));
    canvas.height = overlay.height = Math.max(1, Math.round(h * dpr));
    drawOverlay();
  }
  /** Rig overlay: skinning capsules, bone pivots and tissue regions in the bind pose. */
  function drawOverlay() {
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!$('rig').checked) return;
    const s = overlay.width / VIEW.w;
    ctx.setTransform(s, 0, 0, s, -VIEW.x * s, -VIEW.y * s);
    ctx.lineWidth = 2 / s;
    ctx.font = `${11 / s}px system-ui`;
    ctx.strokeStyle = 'rgba(120, 255, 210, .9)';
    for (const g of W.Rig.REGIONS) {
      ctx.beginPath();
      ctx.ellipse(g.c[0], g.c[1], g.r[0], g.r[1], 0, Math.PI, Math.PI * 2);                         // above centre
      ctx.ellipse(g.c[0], g.c[1], g.r[0], g.r.length < 3 ? g.r[1] : g.r[2], 0, 0, Math.PI);       // below centre
      ctx.closePath();
      ctx.stroke();
    }
    W.Rig.BONES.forEach((b, i) => {
      const hue = (i * 222.5) % 360;
      ctx.strokeStyle = `hsla(${hue}, 90%, 65%, .9)`;
      ctx.fillStyle = `hsl(${hue}, 90%, 65%)`;
      for (const c of b.caps) {
        const a = Math.atan2(c[3] - c[1], c[2] - c[0]);
        ctx.beginPath();
        ctx.arc(c[0], c[1], c[4], a + Math.PI / 2, a - Math.PI / 2);
        ctx.arc(c[2], c[3], c[4], a - Math.PI / 2, a + Math.PI / 2);
        ctx.closePath();
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(b.pivot[0], b.pivot[1], 5 / s, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText(b.name, b.pivot[0] + 8 / s, b.pivot[1] - 6 / s);
    });
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }
  new ResizeObserver(fit).observe($('stage'));
  fit();

  // ---- Pointer grab ---------------------------------------------------------------------
  let grab = null;
  const toImage = (e) => {
    const r = wrap.getBoundingClientRect();
    return [VIEW.x + ((e.clientX - r.left) / r.width) * VIEW.w, VIEW.y + ((e.clientY - r.top) / r.height) * VIEW.h];
  };
  wrap.addEventListener('pointerdown', async (e) => {
    wrap.setPointerCapture(e.pointerId);
    wrap.classList.add('grabbing');
    const start = toImage(e);
    const mine = (grab = { start, now: start, ready: false });
    // Pick the vertex under the pointer in the deformed mesh, then grab around its rest position.
    const { state: s } = await sim.readState();
    if (grab !== mine) return;
    let best = 0, bestD = Infinity;
    for (let i = 0; i < rig.n; i++) {
      const dx = s[i * 4] - start[0], dy = s[i * 4 + 1] - start[1];
      if (dx * dx + dy * dy < bestD) { bestD = dx * dx + dy * dy; best = i; }
    }
    sim.grab.x = rig.rest[best * 4];
    sim.grab.y = rig.rest[best * 4 + 1];
    sim.grab.d0 = [0, 0];
    sim.grab.d1 = [0, 0];
    sim.grab.active = true;
    mine.ready = true;
  });
  wrap.addEventListener('pointermove', (e) => { if (grab) grab.now = toImage(e); });
  const release = () => { grab = null; sim.grab.active = false; wrap.classList.remove('grabbing'); };
  wrap.addEventListener('pointerup', release);
  wrap.addEventListener('pointercancel', release);

  // ---- Frame loop -----------------------------------------------------------------------
  // Fixed simulation step with an accumulator, so behaviour does not depend on display rate.
  const MAX_PER_FRAME = Math.min(48, W.MAX_SUBSTEPS);
  let last = performance.now(), acc = 0, hudAt = 0;
  function tick(now) {
    requestAnimationFrame(tick);
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    state.fps += ((dt > 0 ? 1 / dt : 0) - state.fps) * 0.05;
    let n = 0;
    if (!state.paused) {
      acc += dt * state.speed;
      n = Math.floor(acc / params.h);
      if (n > MAX_PER_FRAME) { n = MAX_PER_FRAME; acc = 0; } else { acc -= n * params.h; }
    } else if (state.stepOnce) {
      n = Math.round(1 / (60 * params.h));
    }
    state.stepOnce = false;
    state.substeps = n;
    if (grab && grab.ready) sim.grab.d1 = [grab.now[0] - grab.start[0], grab.now[1] - grab.start[1]];
    sim.frame(n, n > 0 ? motion.advance(n * params.h) : null);
    if (now - hudAt > 250) {
      hudAt = now;
      const gpu = sim._timing ? `${sim.gpuMs.sim.toFixed(2)} ms sim · ${sim.gpuMs.render.toFixed(2)} ms draw` : 'timing unavailable';
      $('hud').textContent = `${state.fps.toFixed(0)} fps · ${n} substeps/frame\n${rig.nx}×${rig.ny} mesh · ${rig.n} vertices\nGPU ${gpu}`;
    }
  }

  // Hooks for tests and automation.
  const app = { sim, motion, params, rig, state, DEFAULTS, VIEW, initialPreset: motion.preset, initialIntensity: motion.intensity, syncControls: () => { sliders.forEach((s) => s.sync()); presets.sync(); speeds.sync(); } };
  window.wiggle = app;

  if (query.has('selftest')) {
    const report = await W.selftest(app);
    $('selftest').textContent = report.text;
    $('selftest').hidden = false;
  }
  app.ready = true;
  requestAnimationFrame(tick);
})();
