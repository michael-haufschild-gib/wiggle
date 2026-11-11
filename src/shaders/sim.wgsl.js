// Compute shaders for the soft-tissue simulation.
//
// Model: every grid vertex is a column of soft tissue of thickness d standing on a rigid
// substrate (the skinned skeleton target). Mass is proportional to d, the attachment has
// natural frequency w = sqrt(3) * cs / d, neighbouring columns are coupled by in-plane distance
// constraints (wave speed cp) and by cell-area constraints (bulk wave speed cb).
// Solver: XPBD with small steps, one Jacobi iteration per substep. Each vertex sums the
// corrections of its constraints and divides by max(1, gain / sMax). 'gain' bounds the vertex's
// row of the linearised iteration matrix (stiffness fraction times the number of vertices the
// constraint moves: 1 attachment, 2 distance, 4 area). Predict-then-project is stable while that
// bound stays below 4/3, so with sMax = 1 soft constraints integrate as plain dynamics and stiff
// ones degrade to averaged Jacobi instead of diverging.
//
// Units: image pixels and seconds. All constraints are measured against the current skinned
// target, so the tissue's rest shape follows the skeleton.
(function () {
  const W = (window.Wiggle = window.Wiggle || {});

  W.GLOBALS_WGSL = /* wgsl */ `
struct Globals {
  grid: vec4f,       // NX, NY, cell width, cell height (px)
  image: vec4f,      // image width, height (px), kinematic thickness threshold (px), time (s)
  step: vec4f,       // h (s), 1 / (substeps * h), sMax, stretch limit (fraction of thickness)
  material: vec4f,   // cs (px/s), cp (px/s), unused, damping scale
  material2: vec4f,  // cb (px/s), viscosity blend per substep, (grab omega * h)^2, unused
  grabA: vec4f,      // grab anchor.xy (rest px), radius (px), active (0 or 1)
  grabD: vec4f,      // drag delta at frame start .xy and frame end .zw (px)
  view: vec4f,       // clip-space scale.xy and offset.xy
  debug: vec4f,      // debug view mode, unused x3
};
`;

  W.SKIN_WGSL = W.GLOBALS_WGSL + /* wgsl */ `
struct Bones {
  prev: array<mat4x4f, 16>,
  curr: array<mat4x4f, 16>,
};

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<uniform> bones: Bones;
@group(0) @binding(2) var<storage, read> rest: array<vec4f>;       // x, y, depth, thickness
@group(0) @binding(3) var<storage, read> skinIdx: array<vec4u>;
@group(0) @binding(4) var<storage, read> skinW: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> tgt: array<vec4f>;  // target at frame start .xy, frame end .zw

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = u32(G.grid.x) * u32(G.grid.y);
  let i = gid.x;
  if (i >= n) { return; }
  let p = vec4f(rest[i].xyz, 1.0);
  let idx = skinIdx[i];
  let w = skinW[i];
  var a = vec2f(0.0);
  var b = vec2f(0.0);
  for (var k = 0; k < 4; k++) {
    a += w[k] * (bones.prev[idx[k]] * p).xy;
    b += w[k] * (bones.curr[idx[k]] * p).xy;
  }
  tgt[i] = vec4f(a, b);
}
`;

  W.STEP_WGSL = W.GLOBALS_WGSL + /* wgsl */ `
struct Sub { alpha: f32, pad0: f32, pad1: f32, pad2: f32 };

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<uniform> sub: Sub;
@group(0) @binding(2) var<storage, read> rest: array<vec4f>;      // x, y, depth, thickness
@group(0) @binding(3) var<storage, read> mat: array<vec4f>;       // zeta, frequency multiplier, coverage, region
@group(0) @binding(4) var<storage, read> tgt: array<vec4f>;
@group(0) @binding(5) var<storage, read> stateIn: array<vec4f>;   // pos.xy, vel.xy
@group(0) @binding(6) var<storage, read_write> stateOut: array<vec4f>;

const SQRT3 = 1.7320508;

struct Node {
  p: vec2f,    // predicted position
  t: vec2f,    // skinned target at this substep
  v: vec2f,    // velocity
  vt: vec2f,   // target velocity
  w: f32,      // inverse mass (0 = kinematic)
  d: f32,      // effective thickness
  ok: bool,    // inside the grid
};

fn loadNode(x: i32, y: i32) -> Node {
  var n: Node;
  let nx = i32(G.grid.x);
  let ny = i32(G.grid.y);
  if (x < 0 || y < 0 || x >= nx || y >= ny) { return n; }
  let i = u32(y * nx + x);
  let tg = tgt[i];
  n.ok = true;
  n.t = mix(tg.xy, tg.zw, sub.alpha);
  n.vt = (tg.zw - tg.xy) * G.step.y;
  n.d = rest[i].w;
  if (n.d < G.image.z) {
    n.p = n.t;
    n.v = n.vt;
    n.w = 0.0;
  } else {
    let s = stateIn[i];
    n.p = s.xy + s.zw * G.step.x;
    n.v = s.zw;
    n.w = 1.0 / n.d;
  }
  return n;
}

// Area constraint of the quad a-b-c-d (signed area = half the cross product of its diagonals).
// Returns the position correction for corner 'which' in .xy and its gain bound in .z.
fn areaTerm(a: Node, b: Node, c: Node, d: Node, which: i32, complianceBase: f32) -> vec3f {
  let u = c.p - a.p;
  let v = d.p - b.p;
  let ut = c.t - a.t;
  let vt = d.t - b.t;
  let area = 0.5 * (u.x * v.y - u.y * v.x);
  let area0 = 0.5 * (ut.x * vt.y - ut.y * vt.x);
  let gu = 0.25 * dot(u, u);
  let gv = 0.25 * dot(v, v);
  let dAvg = 0.25 * (a.d + b.d + c.d + d.d);
  let denom = (a.w + c.w) * gv + (b.w + d.w) * gu + complianceBase * abs(area0) / max(dAvg, 1e-3);
  var g: vec2f;
  var wi: f32;
  switch which {
    case 0 { g = 0.5 * vec2f(-v.y, v.x); wi = a.w; }
    case 1 { g = 0.5 * vec2f(u.y, -u.x); wi = b.w; }
    case 2 { g = 0.5 * vec2f(v.y, -v.x); wi = c.w; }
    default { g = 0.5 * vec2f(-u.y, u.x); wi = d.w; }
  }
  let lambda = -(area - area0) / denom;
  return vec3f(wi * lambda * g, wi * length(g) * (length(u) + length(v)) / denom);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let nx = u32(G.grid.x);
  let ny = u32(G.grid.y);
  if (gid.x >= nx || gid.y >= ny) { return; }
  let ix = i32(gid.x);
  let iy = i32(gid.y);
  let i = gid.y * nx + gid.x;
  let h = G.step.x;

  // 3x3 neighbourhood, index = (dy + 1) * 3 + (dx + 1); the centre is nb[4].
  var nb: array<Node, 9>;
  for (var k = 0; k < 9; k++) {
    nb[k] = loadNode(ix + k % 3 - 1, iy + k / 3 - 1);
  }
  let c = nb[4];
  if (c.w == 0.0) {
    stateOut[i] = vec4f(c.t, c.vt);
    return;
  }

  let x0 = stateIn[i].xy;
  var dp = vec2f(0.0);
  var gain = 0.0;

  // 1. Attachment to the substrate, stiffening as the stretch limit is approached.
  //    Margin vertices (frequency multiplier 0) have none: they only follow their neighbours.
  let omega = SQRT3 * G.material.x * mat[i].y / c.d;
  let reach = G.step.w * c.d;
  let u = c.p - c.t;
  let r = min(length(u) / reach, 0.95);
  let k2 = omega * omega * h * h / (1.0 - r * r);
  let sPin = k2 / (1.0 + k2);
  dp -= u * sPin;
  gain += sPin;

  // 2. In-plane distance constraints to the 8 neighbours (diagonals at half stiffness, which
  //    makes the square lattice isotropic).
  let cellLen = 0.5 * (G.grid.z + G.grid.w);
  let cmp = cellLen / (G.material.y * h);
  for (var k = 0; k < 9; k++) {
    if (k == 4 || !nb[k].ok) { continue; }
    let n = nb[k];
    let gamma = select(1.0, 0.5, k % 2 == 0);
    let len0 = length(c.t - n.t);
    let dv = c.p - n.p;
    let len = max(length(dv), 1e-6);
    let compliance = cmp * cmp / (gamma * 0.5 * (c.d + n.d));
    let sc = c.w / (c.w + n.w + compliance);
    dp -= dv * (sc * (len - len0) / len);
    gain += 2.0 * sc;
  }

  // 3. Area preservation of the 4 incident cells.
  if (G.material2.x > 1.0) {
    let cb = cellLen / (G.material2.x * h);
    let base = cb * cb;
    var t: vec3f;
    if (nb[0].ok && nb[1].ok && nb[3].ok) { t = areaTerm(nb[0], nb[1], nb[4], nb[3], 2, base); dp += t.xy; gain += t.z; }
    if (nb[1].ok && nb[2].ok && nb[5].ok) { t = areaTerm(nb[1], nb[2], nb[5], nb[4], 3, base); dp += t.xy; gain += t.z; }
    if (nb[3].ok && nb[7].ok && nb[6].ok) { t = areaTerm(nb[3], nb[4], nb[7], nb[6], 1, base); dp += t.xy; gain += t.z; }
    if (nb[5].ok && nb[8].ok && nb[7].ok) { t = areaTerm(nb[4], nb[5], nb[8], nb[7], 0, base); dp += t.xy; gain += t.z; }
  }

  // 4. Pointer grab: pull a Gaussian patch of tissue by the drag delta.
  if (G.grabA.w > 0.5) {
    let q = rest[i].xy - G.grabA.xy;
    let g = exp(-dot(q, q) / (G.grabA.z * G.grabA.z));
    if (g > 0.02) {
      let goal = c.t + g * mix(G.grabD.xy, G.grabD.zw, sub.alpha);
      let kg = G.material2.z * g;
      let sg = kg / (1.0 + kg);
      dp += (goal - c.p) * sg;
      gain += sg;
    }
  }

  var pNew = c.p + dp / max(1.0, gain / G.step.z);
  // Hard stop: attached tissue cannot leave its substrate by more than the stretch limit.
  let off = pNew - c.t;
  let offLen = length(off);
  if (mat[i].y > 0.0 && offLen > reach) { pNew = c.t + off * (reach / offLen); }

  // Damping acts on the velocity relative to the substrate: a dashpot of ratio zeta, then
  // viscosity that relaxes towards the neighbours' mean (it removes ripples, not bulk motion).
  var rel = (pNew - x0) / h - c.vt;
  rel *= exp(-2.0 * mat[i].x * G.material.w * omega * h);
  var relAvg = vec2f(0.0);
  var count = 0.0;
  for (var k = 0; k < 9; k++) {
    if (k == 4 || !nb[k].ok) { continue; }
    relAvg += nb[k].v - nb[k].vt;
    count += 1.0;
  }
  rel = mix(rel, relAvg / max(count, 1.0), G.material2.y);
  stateOut[i] = vec4f(pNew, c.vt + rel);
}
`;
})();
