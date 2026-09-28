/**
 * DOM bindings: file open / drag & drop, transport, keyboard, layer panel, debug panel (spec 9.8).
 */
import { formatTime, ptsUs } from '../format/timing';
import { isActive, type LayerMeta } from '../format/lvf';
import type { Player } from '../player';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/** Frames moved by Shift + arrow keys. */
const BIG_STEP = 10;

/**
 * After a mouse click, controls give focus back to the page so Space and the arrow keys keep
 * driving the player. Keyboard activation (`detail` is 0) keeps its focus, as a keyboard user expects.
 */
const blurAfterMouse = (el: HTMLElement) => el.addEventListener('click', (e) => e.detail > 0 && el.blur());

/** Elements that use Space themselves (keyboard activation) — the player leaves it to them. */
const usesSpace = (t: EventTarget | null) =>
  t instanceof HTMLElement && t.matches('button, a[href], summary, select, textarea, input:not([type=range])');
const isTyping = (t: EventTarget | null) =>
  (t instanceof HTMLInputElement && t.type === 'text') || t instanceof HTMLTextAreaElement;

export function bindUi(player: Player): void {
  const stage = $<HTMLDivElement>('stage');
  const dropHint = $<HTMLDivElement>('drop-hint');
  const status = $<HTMLDivElement>('status');
  const errorCard = $<HTMLDivElement>('error-card');
  const errorText = $<HTMLParagraphElement>('error-text');
  const btnRetry = $<HTMLButtonElement>('btn-retry');
  const btnOpenOther = $<HTMLButtonElement>('btn-open-other');
  const banner = $<HTMLDivElement>('sync-banner');
  const syncText = $<HTMLSpanElement>('sync-text');
  const syncDismiss = $<HTMLButtonElement>('sync-dismiss');
  const notice = $<HTMLDivElement>('notice');
  const fileName = $<HTMLDivElement>('file-name');
  const fileInput = $<HTMLInputElement>('file-input');
  const btnOpen = $<HTMLButtonElement>('btn-open');
  const btnPlay = $<HTMLButtonElement>('btn-play');
  const btnBack = $<HTMLButtonElement>('btn-back');
  const btnFwd = $<HTMLButtonElement>('btn-fwd');
  const btnFirst = $<HTMLButtonElement>('btn-first');
  const btnLast = $<HTMLButtonElement>('btn-last');
  const seek = $<HTMLInputElement>('seek');
  const timeEl = $<HTMLSpanElement>('time');
  const durationEl = $<HTMLSpanElement>('duration');
  const frameEl = $<HTMLSpanElement>('frame');
  const framesEl = $<HTMLSpanElement>('frames');
  const frameGoto = $<HTMLButtonElement>('frame-goto');
  const frameInput = $<HTMLInputElement>('frame-input');
  const loop = $<HTMLInputElement>('loop');
  const mute = $<HTMLInputElement>('mute');
  const layersEl = $<HTMLOListElement>('layers');
  const layerCount = $<HTMLSpanElement>('layer-count');
  const visAll = $<HTMLInputElement>('vis-all');
  const debugList = $<HTMLDListElement>('debug-list');
  const debug = $<HTMLDetailsElement>('debug');
  const alphaFix = $<HTMLInputElement>('alpha-fix');

  // ---- opening files ------------------------------------------------------------------------
  const open = (file: File) => {
    fileName.textContent = `${file.name} · ${(file.size / 1e6).toFixed(1)} MB`;
    void player.open(file);
  };
  btnOpen.addEventListener('click', () => fileInput.click());
  btnOpenOther.addEventListener('click', () => fileInput.click());
  btnRetry.addEventListener('click', () => void player.reload());
  fileInput.addEventListener('change', () => {
    const f = fileInput.files?.[0];
    if (f) open(f);
    fileInput.value = '';
  });
  // Only file drags light up the stage, and only leaving the window turns it off (moving over
  // child elements fires dragleave too).
  window.addEventListener('dragover', (e) => {
    if (!e.dataTransfer?.types.includes('Files')) return;
    e.preventDefault();
    stage.classList.add('dragover');
  });
  window.addEventListener('dragleave', (e) => {
    if (e.relatedTarget === null) stage.classList.remove('dragover');
  });
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    stage.classList.remove('dragover');
    const f = e.dataTransfer?.files?.[0];
    if (f) open(f);
  });

  // ---- transport ------------------------------------------------------------------------------
  const goTo = (frame: number) => player.seek(frame, false);
  const lastFrame = () => Math.max(0, player.frameCount - 1);
  for (const b of [btnPlay, btnBack, btnFwd, btnFirst, btnLast, loop, mute, alphaFix, visAll]) blurAfterMouse(b);
  btnPlay.addEventListener('click', () => player.togglePlay());
  btnBack.addEventListener('click', () => player.step(-1));
  btnFwd.addEventListener('click', () => player.step(1));
  btnFirst.addEventListener('click', () => goTo(0));
  btnLast.addEventListener('click', () => goTo(lastFrame()));

  let scrubbing = false;
  seek.addEventListener('pointerdown', () => (scrubbing = true));
  window.addEventListener('pointerup', () => (scrubbing = false));
  seek.addEventListener('input', () => player.seek(Number(seek.value)));
  seek.addEventListener('change', () => player.seek(Number(seek.value)));
  loop.addEventListener('change', () => (player.loop = loop.checked));
  mute.addEventListener('change', () => player.audio?.setVolume(mute.checked ? 0 : 1));
  alphaFix.addEventListener('change', () => {
    player.compositor.alphaRangeFix = alphaFix.checked;
    player.redraw();
  });

  // Typing a frame number: the readout turns into a field; Enter jumps, Escape cancels.
  const editFrame = () => {
    if (!player.source) return;
    frameInput.value = String(player.currentFrame);
    frameGoto.hidden = true;
    frameInput.hidden = false;
    frameInput.focus();
    frameInput.select();
  };
  const endEdit = (commit: boolean) => {
    if (frameInput.hidden) return;
    if (commit && /^\d+$/.test(frameInput.value.trim())) goTo(Number(frameInput.value));
    frameInput.hidden = true;
    frameGoto.hidden = false;
  };
  frameGoto.addEventListener('click', editFrame);
  frameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') endEdit(true);
    else if (e.key === 'Escape') endEdit(false);
    else return;
    e.preventDefault();
  });
  frameInput.addEventListener('blur', () => endEdit(true));

  // ---- keyboard ---------------------------------------------------------------------------------
  // Space: play/pause. ←/→ (or , .): one frame; with Shift: BIG_STEP frames. Home/End: first/last
  // frame. G: type a frame number. 1–9 toggle the n-th layer of the panel, Shift+n solos it (again:
  // show all), 0 shows all.
  window.addEventListener(
    'keydown',
    (e) => {
      if (isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.code === 'Space') {
        if (usesSpace(e.target)) return;
        e.preventDefault();
        if (!e.repeat) player.togglePlay();
        return;
      }
      const stepKeys: Record<string, number> = { ArrowLeft: -1, ArrowRight: 1, Comma: -1, Period: 1 };
      if (e.code in stepKeys) {
        e.preventDefault();
        player.step(stepKeys[e.code] * (e.shiftKey ? BIG_STEP : 1));
        return;
      }
      if (e.code === 'Home' || e.code === 'End') {
        e.preventDefault();
        if (player.mode === 'playing' || player.mode === 'buffering') player.pause();
        goTo(e.code === 'Home' ? 0 : lastFrame());
        return;
      }
      if (e.code === 'KeyG') {
        e.preventDefault();
        editFrame();
        return;
      }
      const m = /^(Digit|Numpad)(\d)$/.exec(e.code);
      if (!m || !player.source) return;
      e.preventDefault();
      const n = Number(m[2]);
      if (n === 0) showAll();
      else if (n <= panelOrder.length) (e.shiftKey ? solo : toggle)(panelOrder[n - 1]);
    },
    { capture: true },
  );
  window.addEventListener('keyup', (e) => e.code === 'Space' && !usesSpace(e.target) && e.preventDefault(), { capture: true });

  // ---- layer panel ------------------------------------------------------------------------------
  /** Layer indices in panel order (top of the stack first); number keys follow this order. */
  let panelOrder: number[] = [];
  const setAll = (visible: (i: number) => boolean) => player.layerStates.forEach((_, i) => player.setLayerVisible(i, visible(i)));
  const has = (i: number) => i < player.layerStates.length; // rows outlive the file while it reloads
  const showAll = () => setAll(() => true);
  const toggle = (i: number) => has(i) && player.setLayerVisible(i, !player.layerStates[i].visible);
  const isSolo = (i: number) => player.layerStates.every((s, j) => s.visible === (j === i));
  /** Show only layer i (Photoshop: Alt+click the eye); doing it to the layer already alone shows everything again. */
  const solo = (i: number) => has(i) && (isSolo(i) ? showAll() : setAll((j) => j === i));
  // The master eye hides everything while anything is shown, and shows everything otherwise.
  visAll.addEventListener('click', (e) => {
    e.preventDefault();
    const any = player.layerStates.some((s) => s.visible);
    setAll(() => !any);
  });

  const describe = (L: LayerMeta) => {
    const kind = L.kind === 'video' ? `video${L.has_alpha ? '+α' : ''} ${L.coded_width}×${L.coded_height}` : 'still image';
    const parts = [kind, `frames ${L.start_frame}–${L.end_frame - 1}`, `z ${L.z}`];
    if (L.blend !== 'normal') parts.push(`blend: ${L.blend}`);
    return parts.join(' · ');
  };

  const EYE_ON = 'M12 5C6.5 5 2.5 9.5 1 12c1.5 2.5 5.5 7 11 7s9.5-4.5 11-7c-1.5-2.5-5.5-7-11-7zm0 11.5a4.5 4.5 0 1 1 0-9 4.5 4.5 0 0 1 0 9zm0-7a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5z';
  const EYE_OFF = 'M3.3 2.3 2 3.6l3.2 3.2C3.3 8.3 1.8 10.2 1 12c1.5 2.5 5.5 7 11 7 2 0 3.8-.6 5.4-1.5l3 3 1.3-1.3L3.3 2.3zM12 16.5a4.5 4.5 0 0 1-3.9-6.8l1.5 1.5a2.5 2.5 0 0 0 3.2 3.2l1.5 1.5c-.7.4-1.5.6-2.3.6zm9.5-4.5c-1-1.6-3.3-4.3-6.3-5.9l-2.4-2.4C13.6 5.2 12.8 5 12 5c-.7 0-1.4.1-2 .2l7.9 7.9c.7-.4 1.4-1 2.1-1.6L23 12z';
  const CHEVRON = 'M9 6l6 6-6 6z';
  const svg = (cls: string, d: string) => {
    const el = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    el.setAttribute('viewBox', '0 0 24 24');
    el.setAttribute('class', cls);
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', d);
    el.append(path);
    return el;
  };

  /** Thumbnail canvases by layer index; the compositor renders each layer alone into them. */
  const thumbs = new Map<number, CanvasRenderingContext2D>();
  const THUMB_W = 88;
  const THUMB_H = 52;
  /** While playing, thumbnails refresh once a second; when stepping or paused, on every frame. */
  const THUMB_PLAYING_MS = 1000;
  let thumbTimer = 0;
  let thumbsAt = 0;
  const renderThumbs = () => {
    thumbTimer = 0;
    const cf = player.current;
    if (!cf || !player.source || cf.isClosed || !thumbs.size) return;
    thumbsAt = performance.now();
    const order = [...thumbs.keys()];
    const px = player.compositor.thumbnails(cf, order, THUMB_W, THUMB_H);
    order.forEach((i, k) => {
      const p = px[k];
      if (p) thumbs.get(i)!.putImageData(new ImageData(p, THUMB_W, THUMB_H), 0, 0); // else: not in this frame, keep the last picture
    });
  };
  // Rendering reads pixels back from the GPU (a sync point), so it runs after the frame has been
  // drawn, outside the render loop's tick.
  const scheduleThumbs = () => {
    if (thumbTimer) return;
    const wait = player.isPlaying ? Math.max(0, THUMB_PLAYING_MS - (performance.now() - thumbsAt)) : 0;
    thumbTimer = window.setTimeout(renderThumbs, wait);
  };

  const buildLayers = () => {
    layersEl.textContent = '';
    thumbs.clear();
    const meta = player.source?.meta;
    if (!meta) {
      panelOrder = [];
      layerCount.textContent = '';
      visAll.disabled = true;
      return;
    }
    visAll.disabled = false;
    panelOrder = meta.layers.map((_, i) => i).sort((a, b) => meta.layers[b].z - meta.layers[a].z || b - a);
    for (const [row, i] of panelOrder.entries()) {
      const L = meta.layers[i];
      const st = player.layerStates[i];
      const li = document.createElement('li');
      li.className = 'layer';
      li.dataset.index = String(i);
      const main = document.createElement('div');
      main.className = 'row';
      // eye: a checkbox drawn as an eye; Alt+click shows only this layer
      const eye = document.createElement('label');
      eye.className = 'eye';
      eye.title = row < 9 ? `Show / hide (${row + 1}) · Alt+click: only this layer (Shift+${row + 1})` : 'Show / hide · Alt+click: only this layer';
      const vis = document.createElement('input');
      vis.type = 'checkbox';
      vis.className = 'vis';
      vis.checked = st.visible;
      vis.setAttribute('aria-label', `show layer ${L.name || L.id}`);
      vis.addEventListener('click', (e) => {
        if (!e.altKey) return;
        e.preventDefault();
        solo(i);
      });
      vis.addEventListener('change', () => has(i) && player.setLayerVisible(i, vis.checked));
      blurAfterMouse(vis);
      eye.append(vis, svg('eye-on', EYE_ON), svg('eye-off', EYE_OFF));
      const thumb = document.createElement('canvas');
      thumb.className = 'thumb';
      thumb.width = THUMB_W;
      thumb.height = THUMB_H;
      thumb.setAttribute('aria-hidden', 'true');
      thumbs.set(i, thumb.getContext('2d')!);
      const name = document.createElement('div');
      name.className = 'name';
      if (row < 9) {
        const key = document.createElement('span');
        key.className = 'key';
        key.textContent = String(row + 1);
        name.append(key);
      }
      name.append(L.name || L.id);
      name.title = L.id;
      const chev = document.createElement('span');
      chev.className = 'chev';
      chev.append(svg('', CHEVRON));
      main.append(eye, thumb, name, chev);
      // details: description, badges, opacity
      const more = document.createElement('div');
      more.className = 'more';
      more.hidden = true;
      const sub = document.createElement('div');
      sub.className = 'sub';
      sub.textContent = describe(L);
      if (L.kind === 'video' && L.lossless) {
        const badge = document.createElement('span');
        badge.className = 'badge';
        badge.textContent = 'LOSSLESS';
        sub.append(badge);
      }
      const when = document.createElement('span');
      when.className = 'when';
      when.textContent = '· not in this frame';
      sub.append(when);
      const op = document.createElement('div');
      op.className = 'opacity';
      const label = document.createElement('span');
      label.textContent = 'opacity';
      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = '0';
      slider.max = '100';
      slider.value = String(Math.round(st.opacity * 100));
      slider.setAttribute('aria-label', `opacity of layer ${L.name || L.id}`);
      const pct = document.createElement('span');
      pct.textContent = `${slider.value}%`;
      slider.addEventListener('input', () => {
        pct.textContent = `${slider.value}%`;
        if (has(i)) player.setLayerOpacity(i, Number(slider.value) / 100);
      });
      op.append(label, slider, pct);
      more.append(sub, op);
      // clicking the row (not the eye) opens and closes the details
      main.addEventListener('click', (e) => {
        if (eye.contains(e.target as Node)) return;
        li.classList.toggle('open');
        more.hidden = !li.classList.contains('open');
      });
      li.append(main, more);
      layersEl.append(li);
    }
    updateLayerRows();
    scheduleThumbs();
  };

  const updateLayerRows = () => {
    const meta = player.source?.meta;
    if (!meta) return;
    const f = player.currentFrame;
    let shown = 0;
    for (const li of layersEl.children as HTMLCollectionOf<HTMLLIElement>) {
      const i = Number(li.dataset.index);
      const st = player.layerStates[i];
      if (st.visible) shown++;
      const box = li.querySelector<HTMLInputElement>('input.vis');
      if (box && box.checked !== st.visible) box.checked = st.visible;
      li.classList.toggle('inactive', !isActive(meta.layers[i], f));
      li.classList.toggle('hidden', !st.visible);
    }
    layerCount.textContent = `${shown} / ${meta.layers.length}`;
    visAll.checked = shown === meta.layers.length;
    visAll.indeterminate = shown > 0 && shown < meta.layers.length;
  };

  // ---- readouts -----------------------------------------------------------------------------------
  const updateTransport = () => {
    const meta = player.source?.meta;
    btnPlay.classList.toggle('playing', player.isPlaying);
    btnPlay.setAttribute('aria-label', player.isPlaying ? 'Pause' : 'Play');
    if (!meta) {
      seek.max = '0';
      seek.setAttribute('aria-valuetext', 'no file');
      timeEl.textContent = durationEl.textContent = formatTime(0);
      frameEl.textContent = framesEl.textContent = '0';
      return;
    }
    // Both readouts describe the frame on screen; the slider already points at the seek target.
    const f = player.currentFrame;
    timeEl.textContent = formatTime(ptsUs(f, meta.fps));
    frameEl.textContent = String(f);
    if (!scrubbing) seek.value = String(player.mode === 'seeking' ? player.targetFrame : f);
    seek.setAttribute('aria-valuetext', `frame ${f} of ${meta.frame_count - 1}, ${formatTime(ptsUs(f, meta.fps))}`);
  };

  let statusTimer = 0;
  const updateStatus = () => {
    window.clearTimeout(statusTimer);
    const m = player.mode;
    dropHint.hidden = m !== 'empty';
    errorCard.hidden = m !== 'error';
    if (m === 'error') {
      errorText.textContent = player.error?.message ?? 'unknown error';
      status.hidden = true;
    } else if (m === 'buffering' || m === 'loading' || m === 'ended') {
      status.hidden = false;
      status.textContent = m === 'buffering' ? 'buffering…' : m === 'loading' ? 'loading…' : 'end of file';
    } else if (m === 'seeking') {
      // only show for slow seeks, to avoid flicker while stepping
      statusTimer = window.setTimeout(() => {
        if (player.mode === 'seeking') {
          status.hidden = false;
          status.textContent = `seeking to frame ${player.targetFrame}…`;
        }
      }, 250);
    } else {
      status.hidden = true;
    }
    notice.hidden = !player.notice;
    notice.textContent = player.notice ?? '';
  };

  // The sync warning can be dismissed; it comes back when a new failure happens.
  let dismissedFailures = 0;
  const failures = () => player.stats.p6Failures + player.stats.p1Failures;
  const updateBanner = () => {
    const n = failures();
    banner.hidden = n === 0 || n === dismissedFailures;
    syncText.textContent = n ? `⚠ SYNC ASSERTION FAILED ×${n} — ${player.lastSyncFailure}` : '';
  };
  syncDismiss.addEventListener('click', () => {
    dismissedFailures = failures();
    updateBanner();
  });

  const updateDebug = () => {
    if (!debug.open) return;
    const info = player.debugInfo();
    debugList.textContent = '';
    for (const [k, v] of Object.entries(info)) {
      const dt = document.createElement('dt');
      dt.textContent = k;
      const dd = document.createElement('dd');
      dd.textContent = String(v);
      if (k === 'p6Failures' || k === 'p1Failures') dd.className = Number(v) > 0 ? 'bad' : 'good';
      debugList.append(dt, dd);
    }
  };

  player.addEventListener('loaded', () => {
    const meta = player.source!.meta;
    fileName.textContent = `${player.source!.name} · ${(player.source!.bytes.size / 1e6).toFixed(1)} MB`;
    document.title = `${player.source!.name} — LVF`;
    seek.max = String(meta.frame_count - 1);
    // Like the frame readout, the time readout runs up to the last frame's timestamp.
    durationEl.textContent = formatTime(ptsUs(meta.frame_count - 1, meta.fps));
    durationEl.title = `${meta.frame_count} frames, ${formatTime(ptsUs(meta.frame_count, meta.fps))} in total`;
    framesEl.textContent = String(meta.frame_count - 1);
    dismissedFailures = 0;
    endEdit(false);
    buildLayers();
    updateStatus();
    updateBanner();
    player.audio?.setVolume(mute.checked ? 0 : 1);
  });
  player.addEventListener('state', () => {
    updateStatus();
    updateTransport();
    if (player.mode === 'error') buildLayers(); // a failed load leaves no layers; the rows stay during a reload
  });
  player.addEventListener('frame', () => {
    updateTransport();
    updateLayerRows();
    scheduleThumbs();
  });
  player.addEventListener('layers', updateLayerRows);
  player.addEventListener('sync-failure', updateBanner);
  window.setInterval(updateDebug, 250);
  debug.addEventListener('toggle', updateDebug);
  buildLayers();
  updateStatus();
  updateTransport();
}
