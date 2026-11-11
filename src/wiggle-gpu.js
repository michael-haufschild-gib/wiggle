// WebGPU side of the prototype: buffers, pipelines and the per-frame command stream
// skin -> N x step (ping-pong) -> render.
(function () {
  const W = (window.Wiggle = window.Wiggle || {});

  const MAX_SUBSTEPS = 64;
  const SUB_SLOT_BYTES = 256;   // one dynamic-offset uniform slot per substep
  const BONE_FLOATS = 16 * 16;
  const GLOBALS_FLOATS = 36;

  async function compile(device, label, code) {
    const module = device.createShaderModule({ label, code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length) {
      throw new Error(`${label} shader failed to compile: ` + errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
    }
    return module;
  }

  class Sim {
    /**
     * @param canvas  target canvas
     * @param bitmap  the photo (ImageBitmap, straight alpha)
     * @param rig     output of Wiggle.Rig.buildRig
     * @param params  live tuning object, read every frame (see main.js DEFAULTS)
     */
    static async create(canvas, bitmap, rig, params) {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('No WebGPU adapter is available on this machine.');
      const timing = adapter.features.has('timestamp-query');
      const device = await adapter.requestDevice({ requiredFeatures: timing ? ['timestamp-query'] : [] });
      const sim = new Sim(device, canvas, rig, params);
      await sim._init(bitmap, timing);
      return sim;
    }

    constructor(device, canvas, rig, params) {
      this.device = device;
      this.canvas = canvas;
      this.rig = rig;
      this.params = params;
      this.nx = rig.nx;
      this.ny = rig.ny;
      this.n = rig.n;
      this.cur = 0;          // index of the state buffer holding the latest positions
      this.time = 0;
      this.grab = { active: false, x: 0, y: 0, radius: 90, d0: [0, 0], d1: [0, 0] };
      this.gpuMs = { sim: 0, render: 0 };
      this.onError = null;
      this._view = [1, 1, 0, 0];
      this._globals = new Float32Array(GLOBALS_FLOATS);
      this._sub = new Float32Array((MAX_SUBSTEPS * SUB_SLOT_BYTES) / 4);
      this._bones = new Float32Array(BONE_FLOATS * 2);
      this._lastSub = 1;
    }

    async _init(bitmap, timing) {
      const device = this.device;
      const { rig, n } = this;
      device.addEventListener('uncapturederror', (e) => {
        console.error('[wiggle] GPU error: ' + e.error.message);
        if (this.onError) this.onError(e.error.message);
      });

      this.context = this.canvas.getContext('webgpu');
      const format = navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({ device, format, alphaMode: 'premultiplied' });

      const B = GPUBufferUsage;
      const make = (size, usage) => device.createBuffer({ size, usage });
      const storage = (data) => {
        const buf = make(data.byteLength, B.STORAGE | B.COPY_DST | B.COPY_SRC);
        device.queue.writeBuffer(buf, 0, data);
        return buf;
      };
      this.globalsBuf = make(GLOBALS_FLOATS * 4, B.UNIFORM | B.COPY_DST);
      this.subBuf = make(MAX_SUBSTEPS * SUB_SLOT_BYTES, B.UNIFORM | B.COPY_DST);
      this.bonesBuf = make(BONE_FLOATS * 2 * 4, B.UNIFORM | B.COPY_DST);
      this.restBuf = storage(rig.rest);
      this.matBuf = storage(rig.mat);
      this.skinIdxBuf = storage(rig.skinIdx);
      this.skinWBuf = storage(rig.skinW);
      this.targetBuf = make(n * 16, B.STORAGE | B.COPY_DST | B.COPY_SRC);
      this.stateBuf = [0, 1].map(() => make(n * 16, B.STORAGE | B.COPY_DST | B.COPY_SRC));

      // Grid triangles, diagonals alternating so the mesh has no directional bias.
      const { nx, ny } = this;
      const indices = new Uint32Array((nx - 1) * (ny - 1) * 6);
      let o = 0;
      for (let y = 0; y < ny - 1; y++) {
        for (let x = 0; x < nx - 1; x++) {
          const a = y * nx + x, b = a + 1, d = a + nx, c = d + 1;
          if ((x + y) % 2 === 0) { indices.set([a, b, c, a, c, d], o); } else { indices.set([a, b, d, b, c, d], o); }
          o += 6;
        }
      }
      this.indexCount = indices.length;
      this.indexBuf = make(indices.byteLength, B.INDEX | B.COPY_DST);
      device.queue.writeBuffer(this.indexBuf, 0, indices);

      const texture = device.createTexture({
        size: [bitmap.width, bitmap.height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      device.queue.copyExternalImageToTexture({ source: bitmap }, { texture, premultipliedAlpha: true }, [bitmap.width, bitmap.height]);
      const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

      // Bind group layouts.
      const C = GPUShaderStage.COMPUTE, V = GPUShaderStage.VERTEX, F = GPUShaderStage.FRAGMENT;
      const uniform = (binding, visibility, dynamic = false) => ({ binding, visibility, buffer: { type: 'uniform', hasDynamicOffset: dynamic } });
      const readOnly = (binding, visibility) => ({ binding, visibility, buffer: { type: 'read-only-storage' } });
      const readWrite = (binding) => ({ binding, visibility: C, buffer: { type: 'storage' } });
      const layout = (entries) => device.createBindGroupLayout({ entries });
      const skinLayout = layout([uniform(0, C), uniform(1, C), readOnly(2, C), readOnly(3, C), readOnly(4, C), readWrite(5)]);
      const stepLayout = layout([uniform(0, C), uniform(1, C, true), readOnly(2, C), readOnly(3, C), readOnly(4, C), readOnly(5, C), readWrite(6)]);
      const renderLayout = layout([
        uniform(0, V | F), readOnly(1, V), readOnly(2, V), readOnly(3, V), readOnly(4, V), readOnly(5, V),
        { binding: 6, visibility: F, sampler: {} },
        { binding: 7, visibility: F, texture: {} },
      ]);

      const [skinModule, stepModule, renderModule] = await Promise.all([
        compile(device, 'skin', W.SKIN_WGSL),
        compile(device, 'step', W.STEP_WGSL),
        compile(device, 'render', W.RENDER_WGSL),
      ]);
      const computePipe = (module, bgl) => device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: { module, entryPoint: 'main' },
      });
      this.skinPipe = computePipe(skinModule, skinLayout);
      this.stepPipe = computePipe(stepModule, stepLayout);
      const blend = { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' };
      this.renderPipe = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [renderLayout] }),
        vertex: { module: renderModule, entryPoint: 'vs' },
        fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format, blend: { color: blend, alpha: blend } }] },
        primitive: { topology: 'triangle-list' },
      });

      // Bind groups; index [k] of the per-state groups reads state buffer k.
      const group = (bgl, buffers) => device.createBindGroup({
        layout: bgl,
        entries: buffers.map((resource, binding) => ({ binding, resource })),
      });
      const buf = (buffer, size) => (size ? { buffer, size } : { buffer });
      this.skinGroup = group(skinLayout, [buf(this.globalsBuf), buf(this.bonesBuf), buf(this.restBuf), buf(this.skinIdxBuf), buf(this.skinWBuf), buf(this.targetBuf)]);
      this.stepGroup = [0, 1].map((k) => group(stepLayout, [
        buf(this.globalsBuf), buf(this.subBuf, 16), buf(this.restBuf), buf(this.matBuf), buf(this.targetBuf),
        buf(this.stateBuf[k]), buf(this.stateBuf[1 - k]),
      ]));
      this.renderGroup = [0, 1].map((k) => group(renderLayout, [
        buf(this.globalsBuf), buf(this.restBuf), buf(this.stateBuf[k]), buf(this.targetBuf), buf(this.skinIdxBuf), buf(this.skinWBuf),
        sampler, texture.createView(),
      ]));

      if (timing) {
        this._timing = {
          querySet: device.createQuerySet({ type: 'timestamp', count: 4 }),
          resolve: make(32, B.QUERY_RESOLVE | B.COPY_SRC),
          read: make(32, B.COPY_DST | B.MAP_READ),
          busy: false,
        };
      }
      this.reset();
    }

    /** Puts every vertex back on its rest position with the skeleton in the bind pose. */
    reset() {
      const { n, rig, device } = this;
      const state = new Float32Array(n * 4);
      const target = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) {
        const x = rig.rest[i * 4], y = rig.rest[i * 4 + 1];
        state[i * 4] = x; state[i * 4 + 1] = y;
        target.set([x, y, x, y], i * 4);
      }
      device.queue.writeBuffer(this.stateBuf[0], 0, state);
      device.queue.writeBuffer(this.stateBuf[1], 0, state);
      device.queue.writeBuffer(this.targetBuf, 0, target);
      const identity = W.Motion.identityBones();
      this._bones.set(identity, 0);
      this._bones.set(identity, BONE_FLOATS);
      this.cur = 0;
      this.time = 0;
      this.grab.active = false;
    }

    /** Sets the visible rectangle of image space: x0, y0, width, height in image pixels. */
    setView(x0, y0, w, h) {
      this._view = [2 / w, -2 / h, (-2 * x0) / w - 1, (2 * y0) / h + 1];
    }

    _writeGlobals(substeps) {
      const p = this.params, g = this._globals, rig = this.rig, grab = this.grab;
      if (substeps > 0) this._lastSub = substeps;
      const grabStep = Math.PI * 2 * p.grabHz * p.h;
      g.set([this.nx, this.ny, rig.cellX, rig.cellY], 0);
      g.set([W.Rig.IMAGE_W, W.Rig.IMAGE_H, p.kinematicBelow, this.time], 4);
      g.set([p.h, 1 / (this._lastSub * p.h), p.sMax, p.stretchLimit], 8);
      // Softer tissue = slower waves: softness divides all three wave speeds.
      g.set([p.cs / p.softness, p.cp / p.softness, 0, p.damping], 12);
      g.set([p.cb / p.softness, 1 - Math.exp(-p.viscosity * p.h), grabStep * grabStep, 0], 16);
      g.set([grab.x, grab.y, grab.radius, grab.active ? 1 : 0], 20);
      g.set([grab.d0[0], grab.d0[1], grab.d1[0], grab.d1[1]], 24);
      g.set(this._view, 28);
      g.set([p.debugMode, 0, 0, 0], 32);
      this.device.queue.writeBuffer(this.globalsBuf, 0, g);
    }

    /**
     * Simulates 'substeps' fixed steps towards the skeleton pose 'bones' (the pose at the END of
     * those steps), then draws. Pass substeps = 0 to redraw only, render = false to simulate only.
     */
    frame(substeps, bones, render = true) {
      const device = this.device;
      if (substeps > MAX_SUBSTEPS) throw new Error(`at most ${MAX_SUBSTEPS} substeps per frame`);
      const simulate = substeps > 0;
      if (simulate) {
        this._bones.copyWithin(0, BONE_FLOATS, BONE_FLOATS * 2);
        this._bones.set(bones, BONE_FLOATS);
        device.queue.writeBuffer(this.bonesBuf, 0, this._bones);
        for (let k = 0; k < substeps; k++) this._sub[(k * SUB_SLOT_BYTES) / 4] = (k + 1) / substeps;
        device.queue.writeBuffer(this.subBuf, 0, this._sub, 0, (substeps * SUB_SLOT_BYTES) / 4);
      }
      this._writeGlobals(substeps);

      const T = this._timing;
      const enc = device.createCommandEncoder();
      if (simulate) {
        const gx = Math.ceil(this.nx / 8), gy = Math.ceil(this.ny / 8);
        const pass = enc.beginComputePass(T ? { timestampWrites: { querySet: T.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined);
        pass.setPipeline(this.skinPipe);
        pass.setBindGroup(0, this.skinGroup);
        pass.dispatchWorkgroups(Math.ceil(this.n / 64));
        pass.setPipeline(this.stepPipe);
        for (let k = 0; k < substeps; k++) {
          pass.setBindGroup(0, this.stepGroup[this.cur], [k * SUB_SLOT_BYTES]);
          pass.dispatchWorkgroups(gx, gy);
          this.cur ^= 1;
        }
        pass.end();
        this.time += substeps * this.params.h;
        this.grab.d0 = this.grab.d1.slice();
      }
      if (render) {
        const pass = enc.beginRenderPass({
          colorAttachments: [{ view: this.context.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }],
          ...(T ? { timestampWrites: { querySet: T.querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 } } : {}),
        });
        pass.setPipeline(this.renderPipe);
        pass.setBindGroup(0, this.renderGroup[this.cur]);
        pass.setIndexBuffer(this.indexBuf, 'uint32');
        pass.drawIndexed(this.indexCount);
        pass.end();
      }
      const readTiming = T && simulate && render && !T.busy;
      if (readTiming) {
        enc.resolveQuerySet(T.querySet, 0, 4, T.resolve, 0);
        enc.copyBufferToBuffer(T.resolve, 0, T.read, 0, 32);
      }
      device.queue.submit([enc.finish()]);
      if (readTiming) {
        T.busy = true;
        T.read.mapAsync(GPUMapMode.READ).then(() => {
          const t = new BigInt64Array(T.read.getMappedRange().slice(0));
          T.read.unmap();
          T.busy = false;
          const simMs = Number(t[1] - t[0]) / 1e6, renderMs = Number(t[3] - t[2]) / 1e6;
          if (simMs >= 0 && simMs < 1000 && renderMs >= 0 && renderMs < 1000) {
            this.gpuMs.sim += (simMs - this.gpuMs.sim) * 0.1;
            this.gpuMs.render += (renderMs - this.gpuMs.render) * 0.1;
          }
        }, () => { T.busy = false; });
      }
    }

    /**
     * Reads back every vertex: state = pos.xy, vel.xy; target = skinned position at the start
     * (.xy) and end (.zw) of the last simulated frame.
     */
    async readState() {
      const device = this.device;
      const bytes = this.n * 16;
      const out = device.createBuffer({ size: bytes * 2, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(this.stateBuf[this.cur], 0, out, 0, bytes);
      enc.copyBufferToBuffer(this.targetBuf, 0, out, bytes, bytes);
      device.queue.submit([enc.finish()]);
      await out.mapAsync(GPUMapMode.READ);
      const all = new Float32Array(out.getMappedRange().slice(0));
      out.unmap();
      out.destroy();
      const n4 = this.n * 4;
      return { state: all.subarray(0, n4), target: all.subarray(n4) };
    }
  }

  W.Sim = Sim;
  W.MAX_SUBSTEPS = MAX_SUBSTEPS;
})();
