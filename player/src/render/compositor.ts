/**
 * WebGL2 compositor (spec 9.6). Draws one composite frame: every visible layer that is active in
 * that frame, in ascending z (or the order set with setOrder), as a textured rectangle at its canvas rect.
 *
 *  - Color planes arrive as their coded Y, U, V planes and are converted to RGB here, in a pass of
 *    their own at the picture's size, with the exact BT.601/709/2020 formulas and nearest-neighbour
 *    chroma, as fflv does (spec B.10). Measured: Chrome/Edge's own conversion of a BT.709 VideoFrame
 *    gets the blue of saturated colors wrong by up to ~10 levels (yellow Y=218 U=16 V=138 gives
 *    blue 11, not 0). A frame whose pixel format cannot be copied is uploaded as a VideoFrame and
 *    converted by the browser.
 *  - Alpha planes arrive as raw coded luma and are uploaded as R8. Limited-range alpha (Y 16..235,
 *    the default) gets the range mapping clamp((v − 16/255) · 255/219, 0, 1); full-range alpha
 *    (lossless layers) is used as is — either way alpha 0 → 0.0 and 255 → 1.0 exactly. (Measured: letting Chrome/Edge convert the alpha VideoFrame to RGB gives
 *    Y=235 → 253/255, and Edge also corrupts the last column — see LVF_SPEC.md appendix A.)
 *    If a frame's pixel format cannot be copied, the VideoFrame is uploaded and .r used as is;
 *    `alphaRangeFix` then optionally applies the same mapping (debug toggle).
 *  - Colors are straight (non-premultiplied); final alpha = alpha plane × layer opacity.
 *  - Frames padded to even size (content_size < coded size) are cropped back before scaling.
 *  - Video textures are allocated once per picture size (texStorage2D) and then only overwritten
 *    (texSubImage2D), instead of being reallocated by every upload.
 *  - A lost WebGL context is rebuilt when the browser restores it: program, textures, stills.
 */
import type { ColorFormat, ColorPlanes, CompositeFrame, LumaPlane } from '../decode/frames';
import { isActive, type BlendMode, type LvfMeta } from '../format/lvf';
import { zOrder } from '../format/order';

export interface LayerState {
  visible: boolean;
  opacity: number;
}

const VS = `#version 300 es
in vec2 aPos;
uniform vec4 uRect;   // x, y, w, h in canvas pixels (y down)
uniform vec2 uCanvas;
uniform vec2 uUvScale; // content / coded size (crops even-size padding)
out vec2 vUv;
void main() {
  vec2 p = uRect.xy + aPos * uRect.zw;
  gl_Position = vec4(p.x / uCanvas.x * 2.0 - 1.0, 1.0 - p.y / uCanvas.y * 2.0, 0.0, 1.0);
  vUv = aPos * uUvScale;
}`;

const FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uColor;
uniform sampler2D uAlpha;
uniform int uAlphaSource;   // 0 opaque, 1 alpha VideoFrame (.r, browser-converted), 2 color .a (stills), 3 raw luma
uniform bool uAlphaRangeFix;
uniform bool uAlphaFull;    // raw luma is full-range alpha (lossless layers)
uniform float uOpacity;
uniform int uBlend;         // 0 normal, 1 add, 2 multiply, 3 screen
out vec4 outColor;
void main() {
  vec4 c = texture(uColor, vUv);
  float a = 1.0;
  if (uAlphaSource == 3) {
    float y = texture(uAlpha, vUv).r;
    a = uAlphaFull ? y : clamp((y - 16.0 / 255.0) * 255.0 / 219.0, 0.0, 1.0);
  } else if (uAlphaSource == 1) {
    a = texture(uAlpha, vUv).r;
    if (uAlphaRangeFix) a = clamp((a - 16.0 / 255.0) * 255.0 / 219.0, 0.0, 1.0);
  } else if (uAlphaSource == 2) {
    a = c.a;
  }
  a *= uOpacity;
  if (uBlend == 2) outColor = vec4(mix(vec3(1.0), c.rgb, a), 1.0);   // with blendFunc(DST_COLOR, ZERO)
  else if (uBlend == 3) outColor = vec4(c.rgb * a, 1.0);            // with blendFunc(ONE, ONE_MINUS_SRC_COLOR)
  else outColor = vec4(c.rgb, a);                                    // with blendFunc(SRC_ALPHA, ...)
}`;

/** The YUV → RGB pass: one fragment per picture pixel, written to the layer's RGBA texture. */
const VS_FULL = `#version 300 es
in vec2 aPos;
void main() {
  gl_Position = vec4(aPos * 2.0 - 1.0, 0.0, 1.0);
}`;

const FS_YUV = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D uY;
uniform highp sampler2D uU;   // U, or interleaved UV (NV12)
uniform highp sampler2D uV;
uniform bool uNv12;
uniform ivec2 uShift;         // chroma subsampling (log2) in x and y
uniform bool uRgb;            // planes are G, B, R
uniform vec2 uLuma;           // Y offset, Y scale
uniform vec4 uCoef;           // V→R, U→G, V→G, U→B per chroma step
out vec4 outColor;
float fetch(sampler2D t, ivec2 p, int c) { return floor(texelFetch(t, p, 0)[c] * 255.0 + 0.5); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);   // row 0 of the target is the picture's top row, as in the planes
  ivec2 c = p >> uShift;
  float y = fetch(uY, p, 0);
  float u = fetch(uU, c, 0);
  float v = uNv12 ? fetch(uU, c, 1) : fetch(uV, c, 0);
  vec3 rgb;
  if (uRgb) {
    rgb = vec3(v, y, u);
  } else {
    float l = (y - uLuma.x) * uLuma.y;
    float cu = u - 128.0;
    float cv = v - 128.0;
    rgb = vec3(l + cv * uCoef.x, l + (cu * uCoef.y + cv * uCoef.z), l + cu * uCoef.w);
  }
  outColor = vec4(clamp(floor(rgb + 0.5), 0.0, 255.0) / 255.0, 1.0);
}`;

const BLEND_IDS: Record<BlendMode, number> = { normal: 0, add: 1, multiply: 2, screen: 3 };

/** A texture with immutable storage of one size and format. */
interface SizedTexture {
  tex: WebGLTexture;
  width: number;
  height: number;
  format: number;
}

interface VideoTextures {
  /** RGB of the picture (the converted planes, or the uploaded VideoFrame). */
  color: SizedTexture | null;
  /** The coded planes (Y, U or UV, V) of the last converted frame. */
  planes: (SizedTexture | null)[];
  alpha: SizedTexture | null;
  hasAlpha: boolean;
  /** Identity of the composite frame whose planes are in the textures. */
  source: CompositeFrame | null;
}

type StillLoader = (layerIndex: number) => Promise<Blob>;

export class Compositor {
  readonly gl: WebGL2RenderingContext;
  alphaRangeFix = false;
  /** Testing aid: clear to this color instead of canvas.background. */
  backgroundOverride: [number, number, number] | null = null;
  /** Called when a lost WebGL context has been rebuilt (and again when its stills are back). */
  onRestored: () => void = () => {};
  private prog!: WebGLProgram;
  private readonly u: Record<string, WebGLUniformLocation | null> = {};
  private yuvProg!: WebGLProgram;
  private readonly yu: Record<string, WebGLUniformLocation | null> = {};
  private yuvFbo!: WebGLFramebuffer;
  private meta: LvfMeta | null = null;
  private stillLoader: StillLoader | null = null;
  private order: number[] = [];
  private video = new Map<number, VideoTextures>();
  private stills = new Map<number, WebGLTexture>();
  private bg: [number, number, number] = [0, 0, 0];
  /** Bumped by release(): a load() or still re-upload that started earlier no longer applies. */
  private loadToken = 0;
  private lost = false;

  constructor(readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: false });
    if (!gl) throw new Error('WebGL2 is not available');
    this.gl = gl;
    this.init();
    canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault(); // lets the browser restore the context
      this.lost = true;
    });
    canvas.addEventListener('webglcontextrestored', () => this.restore());
  }

  /** Program, vertex data and fixed state; everything a context loss takes away except textures. */
  private init(): void {
    const gl = this.gl;
    this.prog = link(gl, VS, FS);
    for (const n of ['uRect', 'uCanvas', 'uUvScale', 'uColor', 'uAlpha', 'uAlphaSource', 'uAlphaRangeFix', 'uAlphaFull', 'uOpacity', 'uBlend']) {
      this.u[n] = gl.getUniformLocation(this.prog, n);
    }
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); // aPos, bound to 0 in both programs
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.yuvProg = link(gl, VS_FULL, FS_YUV);
    for (const n of ['uY', 'uU', 'uV', 'uNv12', 'uShift', 'uRgb', 'uLuma', 'uCoef']) this.yu[n] = gl.getUniformLocation(this.yuvProg, n);
    gl.useProgram(this.yuvProg);
    gl.uniform1i(this.yu.uY, 0);
    gl.uniform1i(this.yu.uU, 1);
    gl.uniform1i(this.yu.uV, 2);
    this.yuvFbo = gl.createFramebuffer()!;
    gl.useProgram(this.prog);
    gl.uniform1i(this.u.uColor, 0);
    gl.uniform1i(this.u.uAlpha, 1);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  }

  /**
   * Prepare for a file: canvas size, draw order, textures; decodes still images once (all at the
   * same time). Nothing changes until everything is ready, and a load overtaken by release() or a
   * newer load() leaves no trace.
   */
  async load(meta: LvfMeta, stillBlob: StillLoader): Promise<void> {
    this.release();
    const token = this.loadToken;
    const bitmaps = await decodeStills(meta, stillBlob);
    if (token !== this.loadToken) {
      for (const b of bitmaps.values()) b.close();
      return;
    }
    this.meta = meta;
    this.stillLoader = stillBlob;
    this.canvas.width = meta.canvas.width;
    this.canvas.height = meta.canvas.height;
    const hex = meta.canvas.background;
    this.bg = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255) as [number, number, number];
    this.order = zOrder(meta.layers);
    this.createVideoSlots();
    this.uploadStills(bitmaps);
  }

  /** Texture slots for the video layers; storage is allocated by the first upload. */
  private createVideoSlots(): void {
    for (const [i, L] of this.meta!.layers.entries()) {
      if (L.kind === 'video') this.video.set(i, { color: null, planes: [], alpha: null, hasAlpha: L.has_alpha, source: null });
    }
  }

  /** Upload decoded stills (and close the bitmaps). While the context is lost, restore() redoes it. */
  private uploadStills(bitmaps: Map<number, ImageBitmap>): void {
    const gl = this.gl;
    for (const [i, bmp] of bitmaps) {
      if (!this.lost) {
        const old = this.stills.get(i); // a restore that overtook another one
        if (old) gl.deleteTexture(old);
        const tex = newTexture(gl);
        gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
        this.stills.set(i, tex);
      }
      bmp.close();
    }
  }

  /**
   * The context is back: every GL object is gone. Rebuild the program and the textures; video
   * textures refill from the next draw, stills are decoded again.
   */
  private restore(): void {
    this.lost = false;
    this.init();
    this.thumb = null;
    this.video.clear();
    this.stills.clear();
    const meta = this.meta;
    const loader = this.stillLoader;
    if (meta && loader) {
      this.createVideoSlots();
      const token = this.loadToken;
      decodeStills(meta, loader).then(
        (bitmaps) => {
          if (token !== this.loadToken || this.lost) {
            for (const b of bitmaps.values()) b.close();
            return;
          }
          this.uploadStills(bitmaps);
          this.onRestored();
        },
        (e) => console.error('[lvf] could not restore still layers:', e),
      );
    }
    this.onRestored();
  }

  /** Draw the layers in this order (layer indices, bottom first) instead of the file's z order. */
  setOrder(order: readonly number[]): void {
    this.order = [...order];
  }

  /**
   * Draw `frame` (P2: everything on screen comes from this one composite frame; stills are chosen
   * by its frame index). `frame` may be null before the first frame is decoded.
   */
  draw(frame: CompositeFrame | null, states: LayerState[]): void {
    const gl = this.gl;
    const meta = this.meta;
    if (this.lost) return;
    if (meta && frame && !frame.isClosed) {
      for (const li of this.order) {
        const st = states[li];
        if (st.visible && st.opacity > 0 && isActive(meta.layers[li], frame.frameIndex)) this.uploadLayer(frame, li);
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.disable(gl.BLEND);
    const bg = this.backgroundOverride ?? this.bg;
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!meta || !frame || frame.isClosed) return;
    const f = frame.frameIndex;
    gl.useProgram(this.prog);
    gl.uniform2f(this.u.uCanvas, this.canvas.width, this.canvas.height);
    gl.uniform1i(this.u.uAlphaRangeFix, this.alphaRangeFix ? 1 : 0);
    gl.enable(gl.BLEND);
    for (const li of this.order) {
      const L = meta.layers[li];
      const st = states[li];
      if (!st.visible || st.opacity <= 0 || !isActive(L, f)) continue;
      const b = this.bindLayer(frame, li);
      if (!b) continue;
      setBlend(gl, L.blend);
      gl.uniform4f(this.u.uRect, L.rect.x, L.rect.y, L.rect.w, L.rect.h);
      gl.uniform2f(this.u.uUvScale, b.uvx, b.uvy);
      gl.uniform1i(this.u.uAlphaSource, b.alphaSource);
      gl.uniform1f(this.u.uOpacity, st.opacity);
      gl.uniform1i(this.u.uBlend, BLEND_IDS[L.blend]);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    gl.disable(gl.BLEND);
  }

  /**
   * Bind layer `li`'s textures for `frame` (uploading the planes if this frame's are not in them
   * yet) and set its alpha uniforms. Null when the layer has no picture in this frame.
   */
  private bindLayer(frame: CompositeFrame, li: number): { alphaSource: number; uvx: number; uvy: number } | null {
    const gl = this.gl;
    const L = this.meta!.layers[li];
    let alphaSource = 0;
    let uvx = 1;
    let uvy = 1;
    if (L.kind === 'video') {
      if (L.content_size) {
        uvx = L.content_size[0] / L.coded_width;
        uvy = L.content_size[1] / L.coded_height;
      }
      gl.uniform1i(this.u.uAlphaFull, L.alpha_range === 'full' ? 1 : 0);
      const planes = frame.planes.get(li);
      const tex = this.video.get(li)!;
      if (!(planes?.color || planes?.colorPlanes) || tex.source !== frame || !tex.color) return null;
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex.color!.tex);
      if (tex.alpha) {
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, tex.alpha.tex);
        alphaSource = planes.alphaLuma ? 3 : 1;
      }
    } else {
      const tex = this.stills.get(li);
      if (!tex) return null;
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      alphaSource = 2;
    }
    return { alphaSource, uvx, uvy };
  }

  /**
   * Put video layer `li`'s planes of `frame` into its textures, unless they are there already. Runs
   * before a draw sets its own state: a conversion pass changes the framebuffer, viewport, program
   * and texture bindings.
   */
  private uploadLayer(frame: CompositeFrame, li: number): void {
    const gl = this.gl;
    const tex = this.video.get(li);
    const planes = frame.planes.get(li);
    if (!tex || !planes || tex.source === frame) return;
    if (planes.colorPlanes) this.convert(tex, planes.colorPlanes);
    else if (planes.color) tex.color = upload(gl, tex.color, planes.color, gl.BROWSER_DEFAULT_WEBGL);
    else return;
    if (tex.hasAlpha && planes.alphaLuma) tex.alpha = uploadLuma(gl, tex.alpha, planes.alphaLuma);
    else if (tex.hasAlpha && planes.alpha) tex.alpha = upload(gl, tex.alpha, planes.alpha, gl.NONE);
    tex.source = frame;
  }

  /** Upload the coded planes and convert them to RGB into `tex.color` (picture size). */
  private convert(tex: VideoTextures, p: ColorPlanes): void {
    const gl = this.gl;
    const [sx, sy] = CHROMA_SHIFT[p.format];
    const cw = (p.width + sx) >> sx;
    const ch = (p.height + sy) >> sy;
    const nv12 = p.format === 'NV12';
    const L = p.layout;
    gl.activeTexture(gl.TEXTURE0);
    tex.color = sized(gl, tex.color, p.width, p.height, gl.RGBA8);
    tex.planes[0] = uploadPlane(gl, tex.planes[0] ?? null, p.data, L[0], p.width, p.height, 1);
    gl.activeTexture(gl.TEXTURE1);
    tex.planes[1] = uploadPlane(gl, tex.planes[1] ?? null, p.data, L[1], cw, ch, nv12 ? 2 : 1);
    gl.activeTexture(gl.TEXTURE2);
    if (!nv12) tex.planes[2] = uploadPlane(gl, tex.planes[2] ?? null, p.data, L[2], cw, ch, 1);
    else gl.bindTexture(gl.TEXTURE_2D, tex.planes[1].tex); // unused, but bound to something valid
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.yuvFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex.color.tex, 0);
    gl.viewport(0, 0, p.width, p.height);
    gl.disable(gl.BLEND);
    gl.useProgram(this.yuvProg);
    gl.uniform1i(this.yu.uNv12, nv12 ? 1 : 0);
    gl.uniform2i(this.yu.uShift, sx, sy);
    gl.uniform1i(this.yu.uRgb, p.matrix === 'rgb' ? 1 : 0);
    if (p.matrix !== 'rgb') {
      // the same numbers as fflv's YUV tables (crates/fflv/src/pixel.rs), rounded to f32 there too
      const [kr, kb] = p.matrix;
      const kg = 1 - kr - kb;
      const cs = p.fullRange ? 1 : 255 / 224;
      gl.uniform2f(this.yu.uLuma, p.fullRange ? 0 : 16, p.fullRange ? 1 : 255 / 219);
      gl.uniform4f(this.yu.uCoef, 2 * (1 - kr) * cs, ((-2 * kb * (1 - kb)) / kg) * cs, ((-2 * kr * (1 - kr)) / kg) * cs, 2 * (1 - kb) * cs);
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(this.prog);
  }

  /**
   * Render each of `layers` in `frame` on its own, fitted into w×h — the layer panel's thumbnails.
   * Blend mode and opacity are ignored so a layer's own pixels are shown, and transparent areas
   * come back with alpha 0. Returns straight-alpha RGBA rows (top to bottom) per layer, or null
   * for a layer with no picture in this frame (a still is drawn whether or not it is active).
   * All layers are drawn into one offscreen target and read back at once: one GPU sync per call.
   */
  thumbnails(frame: CompositeFrame, layers: number[], w: number, h: number): (Uint8ClampedArray<ArrayBuffer> | null)[] {
    const gl = this.gl;
    const meta = this.meta;
    const result: (Uint8ClampedArray<ArrayBuffer> | null)[] = layers.map(() => null);
    if (!meta || frame.isClosed || !layers.length || this.lost) return result;
    for (const li of layers) if (meta.layers[li].kind === 'video' && isActive(meta.layers[li], frame.frameIndex)) this.uploadLayer(frame, li);
    const perPass = Math.max(1, Math.floor((gl.getParameter(gl.MAX_TEXTURE_SIZE) as number) / h));
    gl.useProgram(this.prog);
    gl.uniform2f(this.u.uCanvas, w, h);
    gl.uniform1i(this.u.uAlphaRangeFix, this.alphaRangeFix ? 1 : 0);
    gl.uniform1f(this.u.uOpacity, 1);
    gl.uniform1i(this.u.uBlend, 0);
    gl.disable(gl.BLEND);
    for (let first = 0; first < layers.length; first += perPass) {
      const slots = Math.min(perPass, layers.length - first);
      const t = this.thumbTarget(w, h * slots);
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
      gl.viewport(0, 0, w, h * slots);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      const drawn: boolean[] = [];
      for (let k = 0; k < slots; k++) {
        const li = layers[first + k];
        const L = meta.layers[li];
        const b = L.kind === 'video' && !isActive(L, frame.frameIndex) ? null : this.bindLayer(frame, li);
        drawn.push(b !== null);
        if (!b) continue;
        const s = Math.min(w / L.rect.w, h / L.rect.h);
        const dw = Math.max(1, Math.round(L.rect.w * s));
        const dh = Math.max(1, Math.round(L.rect.h * s));
        gl.viewport(0, k * h, w, h);
        gl.uniform4f(this.u.uRect, (w - dw) / 2, (h - dh) / 2, dw, dh);
        gl.uniform2f(this.u.uUvScale, b.uvx, b.uvy);
        gl.uniform1i(this.u.uAlphaSource, b.alphaSource);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
      const out = new Uint8ClampedArray(w * h * slots * 4);
      gl.readPixels(0, 0, w, h * slots, gl.RGBA, gl.UNSIGNED_BYTE, out);
      const row = w * 4;
      for (let k = 0; k < slots; k++) {
        if (!drawn[k]) continue;
        const px = new Uint8ClampedArray(new ArrayBuffer(w * h * 4));
        // slot k occupies rows k*h .. (k+1)*h of the read-back, bottom to top
        for (let r = 0; r < h; r++) px.set(out.subarray((k * h + h - 1 - r) * row, (k * h + h - r) * row), r * row);
        result[first + k] = px;
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return result;
  }

  private thumb: { fbo: WebGLFramebuffer; tex: WebGLTexture; w: number; h: number } | null = null;

  /** The offscreen w×h render target for thumbnails, (re)created when the size changes. */
  private thumbTarget(w: number, h: number): { fbo: WebGLFramebuffer; tex: WebGLTexture } {
    const gl = this.gl;
    if (this.thumb && this.thumb.w === w && this.thumb.h === h) return this.thumb;
    if (this.thumb) {
      gl.deleteFramebuffer(this.thumb.fbo);
      gl.deleteTexture(this.thumb.tex);
    }
    // Create it on unit 0 and unbind: a texture that is both the render target and bound to a
    // sampler unit is a feedback loop, and WebGL then refuses to draw.
    gl.activeTexture(gl.TEXTURE0);
    const tex = newTexture(gl);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.thumb = { fbo, tex, w, h };
    return this.thumb;
  }

  /** Read back canvas pixels (RGBA, rows top to bottom). Call right after draw(), in the same task. */
  readPixels(x: number, y: number, w: number, h: number): Uint8Array {
    const gl = this.gl;
    const out = new Uint8Array(w * h * 4);
    gl.readPixels(x, this.canvas.height - y - h, w, h, gl.RGBA, gl.UNSIGNED_BYTE, out);
    const row = w * 4;
    const flipped = new Uint8Array(out.length);
    for (let r = 0; r < h; r++) flipped.set(out.subarray((h - 1 - r) * row, (h - r) * row), r * row);
    return flipped;
  }

  release(): void {
    const gl = this.gl;
    this.loadToken++;
    for (const t of this.video.values()) {
      if (t.color) gl.deleteTexture(t.color.tex);
      for (const p of t.planes) if (p) gl.deleteTexture(p.tex);
      if (t.alpha) gl.deleteTexture(t.alpha.tex);
    }
    for (const t of this.stills.values()) gl.deleteTexture(t);
    if (this.thumb) {
      gl.deleteFramebuffer(this.thumb.fbo);
      gl.deleteTexture(this.thumb.tex);
      this.thumb = null;
    }
    this.video.clear();
    this.stills.clear();
    this.meta = null;
    this.stillLoader = null;
  }
}

/** Decode the still layers' images, all at once; on failure none is left open. */
async function decodeStills(meta: LvfMeta, stillBlob: StillLoader): Promise<Map<number, ImageBitmap>> {
  const opts: ImageBitmapOptions = { premultiplyAlpha: 'none', colorSpaceConversion: 'none' };
  const stills = meta.layers.flatMap((L, i) => (L.kind === 'still' ? [i] : []));
  const results = await Promise.allSettled(stills.map(async (i) => createImageBitmap(await stillBlob(i), opts)));
  const out = new Map<number, ImageBitmap>();
  results.forEach((r, k) => r.status === 'fulfilled' && out.set(stills[k], r.value));
  const failed = results.find((r) => r.status === 'rejected');
  if (failed) {
    for (const b of out.values()) b.close();
    throw failed.reason;
  }
  return out;
}

/** `cur` if it has this size and format, else a new texture with that storage (`cur` is deleted). */
function sized(gl: WebGL2RenderingContext, cur: SizedTexture | null, width: number, height: number, format: number): SizedTexture {
  if (cur && cur.width === width && cur.height === height && cur.format === format) {
    gl.bindTexture(gl.TEXTURE_2D, cur.tex);
    return cur;
  }
  if (cur) gl.deleteTexture(cur.tex);
  const tex = newTexture(gl);
  gl.texStorage2D(gl.TEXTURE_2D, 1, format, width, height);
  return { tex, width, height, format };
}

/** Upload a VideoFrame at its display size (the size WebGL uploads it at); returns the texture used. */
function upload(gl: WebGL2RenderingContext, cur: SizedTexture | null, frame: VideoFrame, colorspace: number): SizedTexture {
  const t = sized(gl, cur, frame.displayWidth, frame.displayHeight, gl.RGBA8);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, colorspace);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, frame);
  return t;
}

function uploadLuma(gl: WebGL2RenderingContext, cur: SizedTexture | null, luma: LumaPlane): SizedTexture {
  return uploadPlane(gl, cur, luma.data, luma, luma.width, luma.height, 1);
}

/** Chroma subsampling (log2, x and y) of the copied color formats. */
const CHROMA_SHIFT: Record<ColorFormat, [number, number]> = { I420: [1, 1], I422: [1, 0], I444: [0, 0], NV12: [1, 1] };

/** Upload one 8-bit plane (1 or 2 interleaved components) of a copied frame, rows at `at.stride` bytes. */
function uploadPlane(
  gl: WebGL2RenderingContext,
  cur: SizedTexture | null,
  data: Uint8Array,
  at: { offset: number; stride: number },
  width: number,
  height: number,
  components: 1 | 2,
): SizedTexture {
  const t = sized(gl, cur, width, height, components === 1 ? gl.R8 : gl.RG8);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.pixelStorei(gl.UNPACK_ROW_LENGTH, at.stride / components);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, components === 1 ? gl.RED : gl.RG, gl.UNSIGNED_BYTE, data, at.offset);
  gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0); // it would apply to VideoFrame uploads too
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  return t;
}

function setBlend(gl: WebGL2RenderingContext, mode: BlendMode): void {
  switch (mode) {
    case 'add':
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE, gl.ZERO, gl.ONE);
      break;
    case 'multiply':
      gl.blendFuncSeparate(gl.DST_COLOR, gl.ZERO, gl.ZERO, gl.ONE);
      break;
    case 'screen':
      gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_COLOR, gl.ZERO, gl.ONE);
      break;
    default:
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }
}

function newTexture(gl: WebGL2RenderingContext): WebGLTexture {
  const t = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

function link(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const sh = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`shader: ${gl.getShaderInfoLog(s)}`);
    return s;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, 0, 'aPos');
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`program: ${gl.getProgramInfoLog(p)}`);
  return p;
}
