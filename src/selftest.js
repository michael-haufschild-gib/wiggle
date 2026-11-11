// Physics self-test (open index.html?selftest). Drives the simulation with a fixed step,
// reads vertex state back from the GPU and checks what a viewer would notice: the photo is
// undistorted at rest, rigid parts stay rigid, soft tissue rings at a plausible frequency and
// dies out, nothing blows up at extreme settings, and the result does not depend on frame rate.
(function () {
  const W = (window.Wiggle = window.Wiggle || {});

  /**
   * Drops the whole body 'drop' px over 0.1 s, holds it, and records how vertex 'probe' rings
   * relative to its target. Returns the ring frequency (Hz), damping ratio and the trace (px).
   * The caller sets parameters and resets the simulation first.
   */
  W.ringdown = async function (app, probe, drop = 30, frames = 200) {
    const { sim, params } = app;
    const RAMP = 6;
    const substeps = Math.round(1 / (60 * params.h));
    const frameDt = substeps * params.h;
    const trace = [];
    for (let f = 0; f < frames; f++) {
      const t = Math.min(1, (f + 1) / RAMP);
      sim.frame(substeps, W.Motion.translatedBones(0, drop * t * t * (3 - 2 * t)), false);
      const s = await sim.readState();
      if (f >= RAMP) trace.push(s.state[probe * 4 + 1] - s.target[probe * 4 + 3]);
    }
    // Zero crossings give the period; the peak of each half-cycle gives the decay.
    const crossings = [];
    const peaks = [];
    let peak = 0;
    for (let k = 1; k < trace.length; k++) {
      peak = Math.max(peak, Math.abs(trace[k - 1]));
      if ((trace[k - 1] < 0) !== (trace[k] < 0)) {
        crossings.push((k - 1 + trace[k - 1] / (trace[k - 1] - trace[k])) * frameDt);
        peaks.push(peak);
        peak = 0;
      }
    }
    const usable = peaks.filter((p) => p > 0.2).length;   // half-cycles above the noise floor
    let hz = 0, zeta = 0;
    if (usable >= 3) {
      hz = (usable - 1) / (2 * (crossings[usable - 1] - crossings[0]));
      const decrement = Math.log(peaks[1] / peaks[usable - 1]) / (usable - 2);   // per half cycle
      zeta = decrement / Math.hypot(Math.PI, decrement);
    }
    const first = Math.max(...trace.slice(0, 30).map(Math.abs));
    const tail = Math.max(...trace.slice(-30).map(Math.abs));
    return { hz, zeta, usable, first, tail, seconds: trace.length * frameDt, trace };
  };

  W.selftest = async function (app) {
    const { sim, motion, params, rig, DEFAULTS } = app;
    const results = [];
    const record = (name, pass, detail) => { results.push({ name, pass, detail }); };
    const nx = rig.nx, ny = rig.ny, n = rig.n;

    const restart = (overrides = {}) => {
      Object.assign(params, DEFAULTS, overrides);
      motion.reset();
      motion.preset = 'idle';
      motion.intensity = 1;
      motion.tempo = 1;
      sim.reset();
    };
    const framesPer60 = () => Math.round(1 / (60 * params.h));
    /** Displacement of vertex i from its skinned target. */
    const offset = (s, i) => [s.state[i * 4] - s.target[i * 4 + 2], s.state[i * 4 + 1] - s.target[i * 4 + 3]];
    const offsetLength = (s, i) => Math.hypot(...offset(s, i));
    const isKinematic = (i) => rig.rest[i * 4 + 3] < params.kinematicBelow;
    const visible = (i) => rig.mat[i * 4 + 2] >= 0.5;

    // 1. Rest: with the skeleton still, the mesh must stay on the photo.
    restart();
    for (let f = 0; f < 120; f++) sim.frame(framesPer60(), W.Motion.identityBones(), false);
    let s = await sim.readState();
    let worst = 0;
    for (let i = 0; i < n; i++) worst = Math.max(worst, offsetLength(s, i));
    record('Rest pose is undistorted', worst < 0.05, `max displacement ${worst.toFixed(4)} px (limit 0.05)`);

    // 2. Shimmy: rigid parts follow the skeleton exactly, soft tissue visibly moves.
    restart();
    motion.preset = 'shimmy';
    let rigidError = 0, facePeak = 0, chestPeak = 0, bellyPeak = 0;
    for (let f = 0; f < 180; f++) {
      sim.frame(framesPer60(), motion.advance(framesPer60() * params.h), false);
      if (f % 6 !== 5) continue;
      s = await sim.readState();
      for (let i = 0; i < n; i++) if (isKinematic(i)) rigidError = Math.max(rigidError, offsetLength(s, i));
      facePeak = Math.max(facePeak, offsetLength(s, rig.probes.face));
      chestPeak = Math.max(chestPeak, offsetLength(s, rig.probes.chestL), offsetLength(s, rig.probes.chestR));
      bellyPeak = Math.max(bellyPeak, offsetLength(s, rig.probes.belly));
    }
    record('Rigid parts follow the skeleton', rigidError < 0.01 && facePeak < 0.5,
      `kinematic error ${rigidError.toFixed(4)} px, face ${facePeak.toFixed(3)} px`);
    record('Soft tissue jiggles, thicker tissue more', chestPeak > 5 && chestPeak > 2 * bellyPeak,
      `chest peak ${chestPeak.toFixed(1)} px, belly peak ${bellyPeak.toFixed(1)} px`);

    // 3. Ring-down: drop the whole body 30 px in 0.1 s, hold it, and watch the chest apex.
    restart();
    const apex = rig.probes.chestL;
    const { hz, zeta, usable, first, tail, seconds } = await W.ringdown(app, apex);
    const designHz = W.Rig.REGIONS[0].ringHz;
    record('Chest rings at its design frequency', usable >= 3 && Math.abs(hz - designHz) < 0.25 * designHz,
      `${hz.toFixed(2)} Hz over ${usable} half-cycles (design ${designHz} Hz ± 25%)`);
    record('Ringing is underdamped and dies out', zeta > 0.1 && zeta < 0.4 && tail < 0.05 * first,
      `damping ratio ${zeta.toFixed(2)} (0.1–0.4), ${first.toFixed(1)} px → ${tail.toFixed(2)} px after ${seconds.toFixed(1)} s`);

    // 4. Stability: softest, least damped, stiffest coupling, strongest and fastest motion.
    restart({ softness: 1.6, damping: 0.3, cp: 2000, cb: 3000 });
    motion.preset = 'mix';
    motion.intensity = 1.5;
    motion.tempo = 1.5;
    let invalid = 0, overLimit = 0, minArea = Infinity;
    const restArea = rig.cellX * rig.cellY;
    for (let f = 0; f < 1200; f++) {
      sim.frame(framesPer60(), motion.advance(framesPer60() * params.h), false);
      if (f % 40 !== 39) continue;
      s = await sim.readState();
      for (let i = 0; i < n; i++) {
        const len = offsetLength(s, i);
        if (!Number.isFinite(len)) { invalid++; continue; }
        const limit = params.stretchLimit * rig.rest[i * 4 + 3];
        if (!isKinematic(i) && rig.mat[i * 4 + 1] > 0) overLimit = Math.max(overLimit, len / limit);
      }
      // Signed on-screen area of every fully visible cell relative to its area in the photo: a
      // cell that folds over goes negative. (Cells in transparent gaps may fold: that is one
      // outline sliding over another.)
      const area = (v, a, b, c, d) => 0.5 * ((v[c * 4] - v[a * 4]) * (v[d * 4 + 1] - v[b * 4 + 1]) - (v[d * 4] - v[b * 4]) * (v[c * 4 + 1] - v[a * 4 + 1]));
      for (let y = 0; y < ny - 1; y++) {
        for (let x = 0; x < nx - 1; x++) {
          const a = y * nx + x, b = a + 1, d = a + nx, c = d + 1;
          if (!(visible(a) && visible(b) && visible(c) && visible(d))) continue;
          minArea = Math.min(minArea, area(s.state, a, b, c, d) / restArea);
        }
      }
    }
    record('Stable at extreme settings (20 s)', invalid === 0 && overLimit <= 1.02 && minArea > 0.1,
      `${invalid} invalid values, peak stretch ${(overLimit * 100).toFixed(0)}% of limit, smallest visible cell ${(minArea * 100).toFixed(0)}% of its photo area (must stay above 10%)`);

    // 5. Frame-rate independence: the same 2 s of shimmy at 60 and at 120 frames per second.
    const shimmyEnd = async (substeps, frames) => {
      restart();
      motion.preset = 'shimmy';
      for (let f = 0; f < frames; f++) sim.frame(substeps, motion.advance(substeps * params.h), false);
      const r = await sim.readState();
      return [r.state[apex * 4], r.state[apex * 4 + 1]];
    };
    const at60 = await shimmyEnd(framesPer60(), 120);
    const at120 = await shimmyEnd(framesPer60() / 2, 240);
    const gap = Math.hypot(at60[0] - at120[0], at60[1] - at120[1]);
    record('Same result at 60 and 120 fps', gap < 1, `chest apex differs by ${gap.toFixed(3)} px (limit 1)`);

    restart();
    motion.preset = app.initialPreset || 'mix';
    motion.intensity = app.initialIntensity || 1;
    const pass = results.every((r) => r.pass);
    const text = `SELF-TEST ${pass ? 'PASSED' : 'FAILED'}\n` + results.map((r) => `${r.pass ? 'PASS' : 'FAIL'}  ${r.name}\n      ${r.detail}`).join('\n');
    console.log(text);
    window.__selftest = { pass, results };
    return { pass, results, text };
  };
})();
