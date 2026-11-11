// Rig for "Blonde Model in Navy Swimsuit.png" (1024 x 1536): bones, skinning capsules and
// soft-tissue regions, authored in image pixels against the alpha silhouette. buildRig() turns
// them into per-vertex maps for the grid mesh.
(function () {
  const W = (window.Wiggle = window.Wiggle || {});

  const IMAGE_W = 1024;
  const IMAGE_H = 1536;
  const PX_PER_M = 1400; // face height (~265 px) and shoulder width (~490 px) both give ~1400 px/m
  const cm = (v) => (v * PX_PER_M) / 100;

  // Bones. 'caps' are capsules [x0, y0, x1, y1, radius] covering the VISIBLE surface the bone
  // moves; 'sigma' is the width of the weight falloff around them. 'pivot' is [x, y, depth].
  const BONES = [
    { name: 'anchor', pivot: [512, 1720, 0], sigma: 110, caps: [[-200, 1720, 1224, 1720, 110]] },
    { name: 'root', pivot: [553, 1290, 0], sigma: 1, caps: [] },
    { name: 'pelvis', pivot: [553, 1290, 0], sigma: 70, caps: [[330, 1270, 770, 1270, 120], [320, 1440, 790, 1440, 120]] },
    { name: 'spine', pivot: [535, 1090, 0], sigma: 70, caps: [[420, 1050, 650, 1050, 70]] },
    { name: 'chest', pivot: [505, 860, 0], sigma: 60, caps: [[300, 640, 650, 640, 90], [385, 820, 720, 810, 130]] },
    { name: 'neck', pivot: [455, 500, 0], sigma: 35, caps: [[455, 430, 455, 495, 50]] },
    { name: 'head', pivot: [465, 410, 0], sigma: 40, caps: [[465, 130, 478, 340, 150]] },
    { name: 'shoulderL', pivot: [320, 560, 0], sigma: 35, caps: [[290, 555, 190, 585, 45]] },
    { name: 'upperArmL', pivot: [178, 600, 60], sigma: 28, caps: [[172, 640, 165, 980, 55]] },
    { name: 'forearmL', pivot: [163, 1010, 60], sigma: 28, caps: [[160, 1040, 145, 1340, 50]] },
    { name: 'handL', pivot: [140, 1400, 60], sigma: 30, caps: [[140, 1400, 135, 1530, 50]] },
    { name: 'shoulderR', pivot: [590, 560, 0], sigma: 35, caps: [[600, 555, 665, 580, 45]] },
    { name: 'upperArmR', pivot: [672, 600, 60], sigma: 28, caps: [[748, 950, 768, 1000, 38]] },
    { name: 'forearmR', pivot: [775, 1010, 60], sigma: 28, caps: [[790, 1050, 865, 1380, 46]] },
    { name: 'handR', pivot: [880, 1420, 60], sigma: 30, caps: [[878, 1420, 895, 1520, 50]] },
  ];

  // Soft-tissue regions. Each is an ellipse-like footprint with its own radius to the left,
  // right, top and bottom of the centre: r = [left, right, up, down] in px. Tissue is 'cm' thick
  // over the inner PLATEAU of every radius and eases to zero at the rim with no slope at either
  // end.
  //
  // The plateau must reach the body part's visible outline, so the part moves as one mass, outline
  // included; if only the middle moved inside a still outline it would read as liquid sloshing
  // in a bag. The ring between plateau and rim is where the mesh stretches to make up for that
  // motion, so each radius is chosen to put the ring where stretch does not show: plain fabric,
  // smooth skin, or the transparent margin. Where the neighbour is a part that must keep its
  // shape (the left arm beside the chest) the radius is short and the outline moves less.
  //
  // 'zeta' is the damping ratio of a tissue column against its own attachment; 'freq' scales
  // the attachment frequency (hair is light and floppy, not a thick layer); 'ringHz' is the
  // ring-down frequency the self-test expects from the region.
  const PLATEAU = 0.33;
  const REGIONS = [
    { name: 'chestL', c: [395, 800], r: [215, 300, 210, 283], cm: 10.5, zeta: 0.15, freq: 1, ringHz: 2.5 },
    { name: 'chestR', c: [684, 800], r: [300, 320, 210, 283], cm: 10.5, zeta: 0.15, freq: 1 },
    { name: 'belly', c: [530, 1120], r: [210, 210, 170, 170], cm: 3.0, zeta: 0.3, freq: 1 },
    { name: 'hipL', c: [300, 1310], r: [170, 170, 230, 230], cm: 4.5, zeta: 0.3, freq: 1 },
    { name: 'hipR', c: [770, 1310], r: [170, 170, 230, 230], cm: 4.5, zeta: 0.3, freq: 1 },
    { name: 'thighL', c: [370, 1500], r: [220, 220, 170, 170], cm: 5.0, zeta: 0.3, freq: 1 },
    { name: 'thighR', c: [730, 1500], r: [220, 220, 170, 170], cm: 5.0, zeta: 0.3, freq: 1 },
    { name: 'armL', c: [165, 780], r: [85, 85, 220, 220], cm: 2.5, zeta: 0.4, freq: 1 },
    { name: 'hairBun', c: [445, 75], r: [150, 150, 95, 95], cm: 3.0, zeta: 0.3, freq: 0.5 },
    { name: 'hairL', c: [318, 340], r: [60, 60, 190, 190], cm: 2.5, zeta: 0.3, freq: 0.45 },
    { name: 'hairR', c: [628, 330], r: [55, 55, 170, 170], cm: 2.5, zeta: 0.3, freq: 0.45 },
  ];

  /** Tissue thickness (px) of region g at image point (x, y); 0 outside its rim. */
  function regionThickness(g, x, y) {
    const dx = x - g.c[0];
    const dy = y - g.c[1];
    const q = Math.hypot(dx / (dx < 0 ? g.r[0] : g.r[1]), dy / (dy < 0 ? g.r[2] : g.r[3]));
    if (q >= 1) return 0;
    const e = Math.max(0, (q - PLATEAU) / (1 - PLATEAU));
    return cm(g.cm) * (1 - e * e * (3 - 2 * e));
  }

  const DEPTH_RADIUS = 130;   // px: half-depth of the torso, used to inflate the silhouette
  // Thickness may not change faster than this per px. Tissue can be displaced by at most
  // stretchLimit * thickness (0.3), so neighbours pushed to their limits differ by at most
  // 0.3 * 1.5 = 0.45 px per px and the mesh cannot fold over itself.
  const MAX_THICKNESS_SLOPE = 1.5;
  const MARGIN_PX = 100;      // how far the tissue maps are extended into transparent areas
  const MARGIN_DECAY = 0.975; // thickness (so mass and coupling) falloff per px across the margin

  function distToSegment(px, py, x0, y0, x1, y1) {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const len2 = dx * dx + dy * dy;
    const t = len2 > 0 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / len2)) : 0;
    return Math.hypot(px - (x0 + t * dx), py - (y0 + t * dy));
  }

  /** Mean alpha (0..1) of the source image in a small box around each grid vertex. */
  function sampleCoverage(bitmap, nx, ny) {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const data = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
    const cov = new Float32Array(nx * ny);
    const R = 2;
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const cx = Math.round((ix / (nx - 1)) * (bitmap.width - 1));
        const cy = Math.round((iy / (ny - 1)) * (bitmap.height - 1));
        let sum = 0;
        let count = 0;
        for (let y = Math.max(0, cy - R); y <= Math.min(bitmap.height - 1, cy + R); y++) {
          for (let x = Math.max(0, cx - R); x <= Math.min(bitmap.width - 1, cx + R); x++) {
            sum += data[(y * bitmap.width + x) * 4 + 3];
            count++;
          }
        }
        cov[iy * nx + ix] = sum / (count * 255);
      }
    }
    return cov;
  }

  /** Chamfer distance (px) from each opaque vertex to the silhouette edge; 0 outside. */
  function insideDistance(cov, nx, ny, cellX, cellY) {
    const INF = 1e9;
    const dist = new Float32Array(nx * ny);
    for (let i = 0; i < dist.length; i++) dist[i] = cov[i] >= 0.5 ? INF : 0;
    const diag = Math.hypot(cellX, cellY);
    const relax = (i, j, cost) => { if (dist[j] + cost < dist[i]) dist[i] = dist[j] + cost; };
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const i = iy * nx + ix;
        if (ix > 0) relax(i, i - 1, cellX);
        if (iy > 0) relax(i, i - nx, cellY);
        if (ix > 0 && iy > 0) relax(i, i - nx - 1, diag);
        if (ix < nx - 1 && iy > 0) relax(i, i - nx + 1, diag);
      }
    }
    for (let iy = ny - 1; iy >= 0; iy--) {
      for (let ix = nx - 1; ix >= 0; ix--) {
        const i = iy * nx + ix;
        if (ix < nx - 1) relax(i, i + 1, cellX);
        if (iy < ny - 1) relax(i, i + nx, cellY);
        if (ix < nx - 1 && iy < ny - 1) relax(i, i + nx + 1, diag);
        if (ix > 0 && iy < ny - 1) relax(i, i + nx - 1, diag);
      }
    }
    return dist;
  }

  function boxBlur(src, nx, ny, passes) {
    let a = src;
    for (let p = 0; p < passes; p++) {
      const b = new Float32Array(a.length);
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          let sum = 0;
          let count = 0;
          for (let y = Math.max(0, iy - 1); y <= Math.min(ny - 1, iy + 1); y++) {
            for (let x = Math.max(0, ix - 1); x <= Math.min(nx - 1, ix + 1); x++) {
              sum += a[y * nx + x];
              count++;
            }
          }
          b[iy * nx + ix] = sum / count;
        }
      }
      a = b;
    }
    return a;
  }

  /**
   * Builds the per-vertex maps for an nx * ny grid over the image.
   * @returns rest (x, y, depth, thickness), mat (zeta, freq multiplier, coverage, region),
   *   skinIdx / skinW (4 bones per vertex) and probe vertex indices used by tests and the HUD.
   */
  function buildRig(bitmap, nx, ny) {
    const n = nx * ny;
    const cellX = IMAGE_W / (nx - 1);
    const cellY = IMAGE_H / (ny - 1);
    const cov = sampleCoverage(bitmap, nx, ny);

    // Tissue thickness, damping and frequency scale from the regions (opaque vertices only).
    const thick = new Float32Array(n);
    const zeta = new Float32Array(n);
    const freq = new Float32Array(n).fill(1);
    const region = new Float32Array(n).fill(-1);
    for (let i = 0; i < n; i++) {
      if (cov[i] < 0.5) continue;
      const x = (i % nx) * cellX;
      const y = Math.floor(i / nx) * cellY;
      for (let k = 0; k < REGIONS.length; k++) {
        const g = REGIONS[k];
        const d = regionThickness(g, x, y);
        if (d > thick[i]) { thick[i] = d; zeta[i] = g.zeta; freq[i] = g.freq; region[i] = k; }
      }
    }

    // Limit the thickness slope across the visible body (chamfer sweeps, repeated to converge).
    const diagLen = Math.hypot(cellX, cellY);
    const cap = (i, j, len) => { if (cov[j] >= 0.5) thick[i] = Math.min(thick[i], thick[j] + MAX_THICKNESS_SLOPE * len); };
    for (let sweep = 0; sweep < 2; sweep++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const i = iy * nx + ix;
          if (cov[i] < 0.5) continue;
          if (ix > 0) cap(i, i - 1, cellX);
          if (iy > 0) cap(i, i - nx, cellY);
          if (ix > 0 && iy > 0) cap(i, i - nx - 1, diagLen);
          if (ix < nx - 1 && iy > 0) cap(i, i - nx + 1, diagLen);
        }
      }
      for (let iy = ny - 1; iy >= 0; iy--) {
        for (let ix = nx - 1; ix >= 0; ix--) {
          const i = iy * nx + ix;
          if (cov[i] < 0.5) continue;
          if (ix < nx - 1) cap(i, i + 1, cellX);
          if (iy < ny - 1) cap(i, i + nx, cellY);
          if (ix < nx - 1 && iy < ny - 1) cap(i, i + nx + 1, diagLen);
          if (ix > 0 && iy < ny - 1) cap(i, i + nx - 1, diagLen);
        }
      }
    }

    // Extend the tissue maps into transparent areas so silhouettes can move. Margin vertices get
    // a frequency multiplier of 0 (no attachment): they are light followers that stretch between
    // the moving outline and the kinematic background instead of pinning the outline.
    const passes = Math.round(MARGIN_PX / cellX);
    const decay = Math.pow(MARGIN_DECAY, cellX);
    for (let pass = 0; pass < passes; pass++) {
      const next = thick.slice();
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const i = iy * nx + ix;
          if (cov[i] >= 0.5) continue;
          for (let y = Math.max(0, iy - 1); y <= Math.min(ny - 1, iy + 1); y++) {
            for (let x = Math.max(0, ix - 1); x <= Math.min(nx - 1, ix + 1); x++) {
              const j = y * nx + x;
              const d = thick[j] * decay;
              if (d > next[i]) { next[i] = d; zeta[i] = zeta[j]; freq[i] = 0; region[i] = region[j]; }
            }
          }
        }
      }
      thick.set(next);
    }

    // Depth: inflate the silhouette to a rounded cross-section and add the tissue on top. It
    // uses the extended thickness, so a protruding part keeps its depth right up to and past its
    // outline: when the torso turns, the part shifts sideways as a whole, outline included.
    // The cross-section is a quarter sine, not a circle: a circle's depth rises vertically at
    // the outline, and a yaw then folds the outermost cells over themselves. The sine's slope
    // is at most pi / 2, so cells only foreshorten.
    const dist = insideDistance(cov, nx, ny, cellX, cellY);
    let depth = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      depth[i] = DEPTH_RADIUS * Math.sin((Math.PI / 2) * Math.min(dist[i] / DEPTH_RADIUS, 1)) + 0.9 * thick[i];
    }
    depth = boxBlur(depth, nx, ny, 2);

    // Skin weights: logistic falloff of the signed distance to each bone's capsules, top 4.
    const skinIdx = new Uint32Array(n * 4);
    const skinW = new Float32Array(n * 4);
    const weights = new Float64Array(BONES.length);
    const order = BONES.map((_, b) => b);
    for (let i = 0; i < n; i++) {
      const x = (i % nx) * cellX;
      const y = Math.floor(i / nx) * cellY;
      for (let b = 0; b < BONES.length; b++) {
        let sd = Infinity;
        for (const c of BONES[b].caps) sd = Math.min(sd, distToSegment(x, y, c[0], c[1], c[2], c[3]) - c[4]);
        weights[b] = sd === Infinity ? 0 : 1 / (1 + Math.exp(sd / BONES[b].sigma));
      }
      order.sort((p, q) => weights[q] - weights[p]);
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += weights[order[k]];
      for (let k = 0; k < 4; k++) {
        skinIdx[i * 4 + k] = order[k];
        skinW[i * 4 + k] = sum > 0 ? weights[order[k]] / sum : (k === 0 ? 1 : 0);
      }
    }

    const rest = new Float32Array(n * 4);
    const mat = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      rest.set([(i % nx) * cellX, Math.floor(i / nx) * cellY, depth[i], thick[i]], i * 4);
      mat.set([zeta[i], freq[i], cov[i], region[i]], i * 4);
    }

    // Probes: the vertex at each region's centre, one on the right breast's free outline and
    // one on the face.
    const vertexAt = (x, y) => Math.round((y / IMAGE_H) * (ny - 1)) * nx + Math.round((x / IMAGE_W) * (nx - 1));
    const probes = { face: vertexAt(480, 270), chestOutline: vertexAt(815, 800) };
    for (const g of REGIONS) probes[g.name] = vertexAt(g.c[0], g.c[1]);

    return { nx, ny, n, cellX, cellY, rest, mat, skinIdx, skinW, probes };
  }

  W.Rig = { IMAGE_W, IMAGE_H, PX_PER_M, BONES, REGIONS, buildRig };
})();
