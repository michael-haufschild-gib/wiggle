// Render shader: draws the deformed grid mesh textured with the photo.
// The vertex stage reads simulated positions straight from the state storage buffer. The
// photo's own shading travels with the tissue, which is right for a surface that moves under a
// fixed light, so no relighting is applied.
(function () {
  const W = (window.Wiggle = window.Wiggle || {});

  W.RENDER_WGSL = W.GLOBALS_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> rest: array<vec4f>;     // x, y, depth, thickness
@group(0) @binding(2) var<storage, read> state: array<vec4f>;    // pos.xy, vel.xy
@group(0) @binding(3) var<storage, read> tgt: array<vec4f>;      // skinned target, frame end in .zw
@group(0) @binding(4) var<storage, read> skinIdx: array<vec4u>;
@group(0) @binding(5) var<storage, read> skinW: array<vec4f>;
@group(0) @binding(6) var samp: sampler;
@group(0) @binding(7) var img: texture_2d<f32>;

// Debug modes (G.debug.x): 0 photo, 1 thickness, 2 skin weights, 3 depth, 4 displacement,
// 5 mesh, 6 area strain.
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) grid: vec2f,
  @location(2) dbg: vec3f,
};

fn heat(t: f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0) * 4.0;
  return clamp(vec3f(1.5 - abs(x - 3.0), 1.5 - abs(x - 2.0), 1.5 - abs(x - 1.0)), vec3f(0.0), vec3f(1.0));
}

fn boneColor(i: u32) -> vec3f {
  return 0.5 + 0.5 * cos(6.2831853 * (f32(i) * 0.618034 + vec3f(0.0, 0.33, 0.67)));
}

fn quadArea(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
  let u = c - a;
  let v = d - b;
  return 0.5 * (u.x * v.y - u.y * v.x);
}

// Area of the cells around vertex (ix, iy) relative to the same cells of the skinned target.
fn areaRatio(ix: u32, iy: u32, nx: u32, ny: u32) -> f32 {
  var area = 0.0;
  var area0 = 0.0;
  for (var cy = 0u; cy < 2u; cy++) {
    for (var cx = 0u; cx < 2u; cx++) {
      if (ix + cx < 1u || iy + cy < 1u || ix + cx >= nx || iy + cy >= ny) { continue; }
      let a = (iy + cy - 1u) * nx + ix + cx - 1u;
      let b = a + 1u;
      let d = a + nx;
      let c = d + 1u;
      area += quadArea(state[a].xy, state[b].xy, state[c].xy, state[d].xy);
      area0 += quadArea(tgt[a].zw, tgt[b].zw, tgt[c].zw, tgt[d].zw);
    }
  }
  return area / max(area0, 1e-3);
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let nx = u32(G.grid.x);
  let ny = u32(G.grid.y);
  let ix = i % nx;
  let iy = i / nx;
  var o: VOut;
  o.pos = vec4f(state[i].xy * G.view.xy + G.view.zw, 0.0, 1.0);
  o.uv = rest[i].xy / G.image.xy;
  o.grid = vec2f(f32(ix), f32(iy));

  let mode = i32(G.debug.x + 0.5);
  var dbg = vec3f(0.0);
  if (mode == 1) {
    dbg = heat(rest[i].w / 160.0);
  } else if (mode == 2) {
    let idx = skinIdx[i];
    let w = skinW[i];
    for (var k = 0; k < 4; k++) { dbg += w[k] * boneColor(idx[k]); }
  } else if (mode == 3) {
    dbg = vec3f(rest[i].z / 300.0);
  } else if (mode == 4) {
    dbg = heat(length(state[i].xy - tgt[i].zw) / 40.0);
  } else if (mode == 6) {
    let e = clamp((areaRatio(ix, iy, nx, ny) - 1.0) * 3.0, -1.0, 1.0);
    dbg = mix(vec3f(0.5), select(vec3f(0.1, 0.3, 1.0), vec3f(1.0, 0.25, 0.1), e > 0.0), abs(e));
  }
  o.dbg = dbg;
  return o;
}

@fragment
fn fs(v: VOut) -> @location(0) vec4f {
  // The source alpha tops out at 253/255; rescale so the body is fully opaque.
  let c = min(textureSample(img, samp, v.uv) * (255.0 / 253.0), vec4f(1.0));
  let cell = abs(fract(v.grid + 0.5) - 0.5) / max(fwidth(v.grid), vec2f(1e-6));
  let mode = i32(G.debug.x + 0.5);
  if (mode == 0) { return c; }

  let a = max(c.a, 0.35);
  if (mode == 5) {
    let wire = 1.0 - smoothstep(0.0, 1.2, min(cell.x, cell.y));
    return vec4f(mix(c.rgb + vec3f(0.06) * (a - c.a), vec3f(0.2, 1.0, 0.8) * a, wire * 0.7), a);
  }
  return vec4f(mix(c.rgb, v.dbg * a, 0.8), a);
}
`;
})();
