/**
 * WebGL2 compositor (spec 9.6). Draws one composite frame: every visible layer that is active in
 * that frame, in ascending z (or the order set with setOrder), as a textured rectangle at its canvas rect.
 *
 *  - Color planes are uploaded straight from the VideoFrame (the browser converts YUV → RGB).
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
import type { CompositeFrame, LumaPlane } from '../decode/frames';
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

const BLEND_IDS: Record<BlendMode, number> = { normal: 0, add: 1, multiply: 2, screen: 3 };

/** A texture with immutable storage of one size and format. */
interface SizedTexture {
  tex: WebGLTexture;
  width: number;
  height: number;
  format: number;
}

interface VideoTextures {
  color: SizedTexture | null;
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
    const loc = gl.getAttribLocation(this.prog, 'aPos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
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
      if (L.kind === 'video') this.video.set(i, { color: null, alpha: null, hasAlpha: L.has_alpha, source: null });
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
      if (!planes?.color) return null;
      if (tex.source !== frame) {
        tex.color = upload(gl, tex.color, planes.color, gl.BROWSER_DEFAULT_WEBGL);
        if (tex.hasAlpha && planes.alphaLuma) tex.alpha = uploadLuma(gl, tex.alpha, planes.alphaLuma);
        else if (tex.hasAlpha && planes.alpha) tex.alpha = upload(gl, tex.alpha, planes.alpha, gl.NONE);
        tex.source = frame;
      }
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
  const t = sized(gl, cur, luma.width, luma.height, gl.R8);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.pixelStorei(gl.UNPACK_ROW_LENGTH, luma.stride);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, luma.width, luma.height, gl.RED, gl.UNSIGNED_BYTE, luma.data, luma.offset);
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
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`program: ${gl.getProgramInfoLog(p)}`);
  return p;
}
