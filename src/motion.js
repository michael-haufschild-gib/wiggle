// Skeleton motion: procedural "shake" presets drive the pelvis, spine and chest; the head and
// arms follow through on damped springs. Produces one skinning matrix per bone.
// Coordinates are image pixels (y down); depth z points at the camera, so yaw (rotation about
// the vertical axis) shifts near surfaces sideways and gives parallax.
(function () {
  const W = (window.Wiggle = window.Wiggle || {});
  const TAU = Math.PI * 2;
  const rad = (deg) => (deg * Math.PI) / 180;
  const BONE_SLOTS = 16;
  const INTERNAL_HZ = 240;

  // 4x4 column-major matrices as plain arrays.
  function mul(a, b) {
    const o = new Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
      }
    }
    return o;
  }
  const translation = (x, y, z = 0) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
  /** Roll about the view axis (positive = clockwise on screen), then yaw about the vertical axis. */
  function rotation(roll, yaw) {
    const cr = Math.cos(roll), sr = Math.sin(roll), cy = Math.cos(yaw), sy = Math.sin(yaw);
    return mul([cr, sr, 0, 0, -sr, cr, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [cy, 0, -sy, 0, 0, 1, 0, 0, sy, 0, cy, 0, 0, 0, 0, 1]);
  }
  const about = (p, m) => mul(translation(p[0], p[1], p[2]), mul(m, translation(-p[0], -p[1], -p[2])));
  const apply = (m, p) => [m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]];
  /** Rigid 2D motion that carries segment a0-b0 onto a1-b1 (matching midpoint and direction). */
  function segmentTransform(a0, b0, a1, b1) {
    const turn = Math.atan2(b1[1] - a1[1], b1[0] - a1[0]) - Math.atan2(b0[1] - a0[1], b0[0] - a0[0]);
    return mul(translation((a1[0] + b1[0]) / 2, (a1[1] + b1[1]) / 2), mul(rotation(turn, 0), translation(-(a0[0] + b0[0]) / 2, -(a0[1] + b0[1]) / 2)));
  }

  /** Semi-implicit damped spring; s = { x, v }. 'force' is an extra acceleration. */
  function spring(s, target, omega, zeta, dt, force = 0) {
    s.v += (omega * omega * (target - s.x) - 2 * zeta * omega * s.v + force) * dt;
    s.x += s.v * dt;
  }

  const smoothstep = (a, b, x) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

  // Oscillator rates in Hz.
  const RATES = { shimmy: 3.0, hips: 1.5, bounce: 2.3, twist: 0.6, breath: 0.25, idle: 0.13 };

  // Each preset gives the weight of the four moves; 'mix' drifts between them.
  const PRESETS = {
    idle: () => ({ shimmy: 0, hips: 0, bounce: 0, twist: 0 }),
    shimmy: () => ({ shimmy: 1, hips: 0, bounce: 0.2, twist: 0 }),
    hips: () => ({ shimmy: 0, hips: 1, bounce: 0, twist: 0 }),
    bounce: () => ({ shimmy: 0, hips: 0, bounce: 1, twist: 0 }),
    twist: () => ({ shimmy: 0, hips: 0, bounce: 0, twist: 1 }),
    mix: (t) => {
      const a = 0.5 + 0.5 * Math.sin((TAU * t) / 9.0);
      const b = 0.5 + 0.5 * Math.sin((TAU * t) / 6.3 + 1.7);
      const c = 0.5 + 0.5 * Math.sin((TAU * t) / 11.0 + 4.0);
      const shimmy = smoothstep(0.45, 0.8, a);
      return { shimmy, hips: smoothstep(0.35, 0.75, b), bounce: 0.3 + 0.7 * smoothstep(0.4, 0.9, c), twist: 0.35 * (1 - shimmy) };
    },
  };

  const ARM_LENGTH = 420;   // px, shoulder to elbow-ish pendulum length
  const ARM_GAIN = 0.8;

  class Motion {
    constructor() {
      this.preset = 'mix';
      this.intensity = 1;
      this.tempo = 1;
      this.pivot = {};
      this.slot = {};
      W.Rig.BONES.forEach((b, i) => { this.pivot[b.name] = b.pivot; this.slot[b.name] = i; });
      this.reset();
    }

    reset() {
      this.t = 0;
      this.w = { shimmy: 0, hips: 0, bounce: 0, twist: 0 };
      this.phase = { shimmy: 0, hips: 0, bounce: 0, twist: 0, breath: 0, idle: 0 };
      this.kickX = { x: 0, v: 0 };
      this.kickY = { x: 0, v: 0 };
      this.kickSide = 1;
      this.headYaw = { x: 0, v: 0 };
      this.headRoll = { x: 0, v: 0 };
      this.headX = { x: 0, v: 0 };
      this.headY = { x: 0, v: 0 };
      this.arm = { L: { x: 0, v: 0, px: null, pv: null }, R: { x: 0, v: 0, px: null, pv: null } };
      this.M = {};
    }

    /** One-off jolt: the whole body drops and recoils. */
    jolt() {
      this.kickY.v += 1100;
      this.kickSide = -this.kickSide;
      this.kickX.v += 260 * this.kickSide;
    }

    /** Advances the motion by dt seconds and returns the bone matrices (16 slots x 16 floats). */
    advance(dt) {
      const steps = Math.max(1, Math.round(dt * INTERNAL_HZ));
      for (let i = 0; i < steps; i++) this._step(dt / steps);
      return this.matrices();
    }

    matrices() {
      const out = W.Motion.identityBones();
      for (const name in this.M) out.set(this.M[name], this.slot[name] * 16);
      return out;
    }

    _step(dt) {
      this.t += dt;
      const target = PRESETS[this.preset](this.t);
      const blend = 1 - Math.exp(-dt / 0.3);
      for (const k in this.w) this.w[k] += (target[k] - this.w[k]) * blend;
      for (const k in this.phase) {
        this.phase[k] += TAU * RATES[k] * (k === 'breath' || k === 'idle' ? 1 : this.tempo) * dt;
      }
      spring(this.kickX, 0, TAU * 2.2, 0.55, dt);
      spring(this.kickY, 0, TAU * 2.2, 0.55, dt);

      // Primary channels (px and radians).
      const I = this.intensity, w = this.w, ph = this.phase, P = this.pivot, M = this.M;
      const shimmy = w.shimmy * I * Math.sin(ph.shimmy);
      const shimmyQ = w.shimmy * I * Math.cos(ph.shimmy);
      const hips = w.hips * I * Math.sin(ph.hips);
      const twist = w.twist * I * Math.sin(ph.twist);
      const twistLag = w.twist * I * Math.sin(ph.twist - 0.6);
      const bounce = w.bounce * I * (Math.sin(ph.bounce) + 0.25 * Math.sin(2 * ph.bounce));
      const breath = Math.sin(ph.breath);
      const idle = Math.sin(ph.idle);

      const pelvisRoll = rad(-5.5) * hips + rad(0.5) * idle;
      const pelvisYaw = rad(5) * twistLag - rad(1.8) * shimmy;
      const spineRoll = rad(3.2) * hips;
      const spineYaw = rad(2) * twistLag;
      const chestRoll = rad(1.6) * hips + rad(1.2) * shimmyQ;
      const chestYaw = rad(7) * shimmy + rad(6) * twist;

      // Torso chain. Hips shift and roll; spine and chest counter them so the ribcage stays upright.
      M.root = translation(5 * idle + this.kickX.x, 24 * bounce + 2.5 * w.shimmy * I * Math.sin(2 * ph.shimmy) + this.kickY.x);
      M.pelvis = mul(M.root, mul(translation(30 * hips, 0), about(P.pelvis, rotation(pelvisRoll, pelvisYaw))));
      M.spine = mul(M.pelvis, mul(translation(-13 * hips, 0), about(P.spine, rotation(spineRoll, spineYaw))));
      M.chest = mul(M.spine, mul(translation(-9 * hips, -1.5 * breath), about(P.chest, rotation(chestRoll, chestYaw))));
      M.shoulderL = mul(M.chest, translation(0, -6 * shimmyQ - 1.5 * breath));
      M.shoulderR = mul(M.chest, translation(0, 6 * shimmyQ - 1.5 * breath));

      // Head: stays mostly level and facing the camera, trailing the neck base on a spring.
      const worldYaw = pelvisYaw + spineYaw + chestYaw;
      const worldRoll = pelvisRoll + spineRoll + chestRoll;
      spring(this.headYaw, 0.15 * worldYaw, TAU * 2.5, 0.7, dt);
      spring(this.headRoll, 0.25 * worldRoll, TAU * 2.5, 0.7, dt);
      const neck = apply(M.chest, P.neck);
      const dx = neck[0] - P.neck[0], dy = neck[1] - P.neck[1];
      spring(this.headX, 0.6 * dx, TAU * 3.5, 0.75, dt);
      spring(this.headY, 0.92 * dy, TAU * 3.5, 0.75, dt);
      const ex = this.headX.x - dx, ey = this.headY.x - dy;
      const localYaw = this.headYaw.x - worldYaw, localRoll = this.headRoll.x - worldRoll;
      M.neck = mul(translation(0.5 * ex, 0.5 * ey), mul(M.chest, about(P.neck, rotation(0.5 * localRoll, 0.5 * localYaw))));
      M.head = mul(translation(ex, ey), mul(M.chest, about(P.head, rotation(localRoll, localYaw))));

      // Arms: the upper arm swings as a pendulum driven by the shoulder's sideways acceleration;
      // the hand stays planted on the thigh (it follows the pelvis) and the forearm spans the two.
      for (const side of ['L', 'R']) {
        const a = this.arm[side];
        const shoulder = M['shoulder' + side];
        const joint = apply(shoulder, P['upperArm' + side]);
        let acc = 0;
        if (a.px !== null) {
          const v = (joint[0] - a.px) / dt;
          if (a.pv !== null) acc = (v - a.pv) / dt;
          a.pv = v;
        }
        a.px = joint[0];
        spring(a, 0, TAU * 1.4, 0.35, dt, (ARM_GAIN * acc) / ARM_LENGTH);
        a.x = Math.max(-0.2, Math.min(0.2, a.x));
        const upper = mul(shoulder, about(P['upperArm' + side], rotation(a.x, 0)));
        const elbow = P['forearm' + side], wrist = P['hand' + side];
        M['upperArm' + side] = upper;
        M['hand' + side] = M.pelvis;
        M['forearm' + side] = segmentTransform(elbow, wrist, apply(upper, elbow), apply(M.pelvis, wrist));
      }
    }

    static identityBones() {
      const out = new Float32Array(BONE_SLOTS * 16);
      for (let b = 0; b < BONE_SLOTS; b++) { out[b * 16] = 1; out[b * 16 + 5] = 1; out[b * 16 + 10] = 1; out[b * 16 + 15] = 1; }
      return out;
    }

    /** All bones translated by (x, y): used by tests to drive the body directly. */
    static translatedBones(x, y) {
      const out = Motion.identityBones();
      for (let b = 0; b < BONE_SLOTS; b++) { out[b * 16 + 12] = x; out[b * 16 + 13] = y; }
      return out;
    }
  }

  W.Motion = Motion;
})();
