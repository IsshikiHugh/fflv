/**
 * DOM bindings: file open / drag & drop, transport, keyboard, layer panel, debug panel (spec 9.8).
 */
import { formatTime, ptsUs } from '../format/timing';
import { isActive } from '../format/lvf';
import type { Player } from '../player';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export function bindUi(player: Player): void {
  const stage = $<HTMLDivElement>('stage');
  const dropHint = $<HTMLDivElement>('drop-hint');
  const status = $<HTMLDivElement>('status');
  const banner = $<HTMLDivElement>('sync-banner');
  const notice = $<HTMLDivElement>('notice');
  const fileName = $<HTMLDivElement>('file-name');
  const fileInput = $<HTMLInputElement>('file-input');
  const btnPlay = $<HTMLButtonElement>('btn-play');
  const btnBack = $<HTMLButtonElement>('btn-back');
  const btnFwd = $<HTMLButtonElement>('btn-fwd');
  const seek = $<HTMLInputElement>('seek');
  const timeEl = $<HTMLSpanElement>('time');
  const durationEl = $<HTMLSpanElement>('duration');
  const frameEl = $<HTMLSpanElement>('frame');
  const framesEl = $<HTMLSpanElement>('frames');
  const loop = $<HTMLInputElement>('loop');
  const mute = $<HTMLInputElement>('mute');
  const layersEl = $<HTMLOListElement>('layers');
  const debugList = $<HTMLDListElement>('debug-list');
  const debug = $<HTMLDetailsElement>('debug');
  const alphaFix = $<HTMLInputElement>('alpha-fix');

  // ---- opening files ------------------------------------------------------------------------
  const open = (file: File) => {
    fileName.textContent = `${file.name} · ${(file.size / 1e6).toFixed(1)} MB`;
    void player.open(file);
  };
  fileInput.addEventListener('change', () => {
    const f = fileInput.files?.[0];
    if (f) open(f);
    fileInput.value = '';
  });
  window.addEventListener('dragover', (e) => {
    e.preventDefault();
    stage.classList.add('dragover');
  });
  window.addEventListener('dragleave', () => stage.classList.remove('dragover'));
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    stage.classList.remove('dragover');
    const f = e.dataTransfer?.files?.[0];
    if (f) open(f);
  });

  // ---- transport ------------------------------------------------------------------------------
  const click = (b: HTMLButtonElement, fn: () => void) =>
    b.addEventListener('click', () => {
      fn();
      b.blur(); // keep Space for play/pause instead of re-clicking the focused button
    });
  click(btnPlay, () => player.togglePlay());
  click(btnBack, () => player.step(-1));
  click(btnFwd, () => player.step(1));

  let scrubbing = false;
  seek.addEventListener('pointerdown', () => (scrubbing = true));
  window.addEventListener('pointerup', () => (scrubbing = false));
  seek.addEventListener('input', () => player.seek(Number(seek.value)));
  seek.addEventListener('change', () => {
    player.seek(Number(seek.value));
    seek.blur();
  });
  loop.addEventListener('change', () => (player.loop = loop.checked));
  mute.addEventListener('change', () => player.audio?.setVolume(mute.checked ? 0 : 1));
  alphaFix.addEventListener('change', () => {
    player.compositor.alphaRangeFix = alphaFix.checked;
    player.setLayerOpacity(0, player.layerStates[0]?.opacity ?? 1); // force a redraw
  });

  window.addEventListener(
    'keydown',
    (e) => {
      if (e.target instanceof HTMLInputElement && e.target.type === 'text') return;
      if (e.code === 'Space') {
        e.preventDefault();
        if (!e.repeat) player.togglePlay();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        player.step(-1);
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        player.step(1);
      }
    },
    { capture: true },
  );
  window.addEventListener('keyup', (e) => e.code === 'Space' && e.preventDefault(), { capture: true });

  // Number keys: 1–9 toggle the n-th layer of the panel, Shift+n solos it (again: show all), 0 shows all.
  window.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement && e.target.type === 'text') return;
    if (e.metaKey || e.ctrlKey || e.altKey || !player.source) return;
    const m = /^(Digit|Numpad)(\d)$/.exec(e.code);
    if (!m) return;
    e.preventDefault();
    const n = Number(m[2]);
    if (n === 0) {
      player.layerStates.forEach((_, i) => player.setLayerVisible(i, true));
    } else if (n <= panelOrder.length) {
      const target = panelOrder[n - 1];
      if (e.shiftKey) {
        const solo = player.layerStates.every((s, i) => s.visible === (i === target));
        player.layerStates.forEach((_, i) => player.setLayerVisible(i, solo || i === target));
      } else {
        player.setLayerVisible(target, !player.layerStates[target].visible);
      }
    }
  });

  // ---- layer panel ------------------------------------------------------------------------------
  /** Layer indices in panel order (top of the stack first); number keys follow this order. */
  let panelOrder: number[] = [];
  const buildLayers = () => {
    layersEl.textContent = '';
    const meta = player.source?.meta;
    if (!meta) return;
    panelOrder = meta.layers.map((_, i) => i).sort((a, b) => meta.layers[b].z - meta.layers[a].z || b - a);
    for (const [row, i] of panelOrder.entries()) {
      const L = meta.layers[i];
      const st = player.layerStates[i];
      const li = document.createElement('li');
      li.className = 'layer';
      li.dataset.index = String(i);
      const vis = document.createElement('input');
      vis.type = 'checkbox';
      vis.checked = st.visible;
      vis.title = 'visible';
      vis.addEventListener('change', () => player.setLayerVisible(i, vis.checked));
      const name = document.createElement('div');
      name.className = 'name';
      if (row < 9) {
        const key = document.createElement('span');
        key.className = 'key';
        key.textContent = String(row + 1);
        key.title = `press ${row + 1} to toggle, Shift+${row + 1} to solo`;
        name.append(key);
      }
      name.append(L.name || L.id);
      if (L.kind === 'video' && L.lossless) {
        const badge = document.createElement('span');
        badge.className = 'badge';
        badge.textContent = 'LOSSLESS';
        name.append(badge);
      }
      name.title = `${L.id}`;
      const sub = document.createElement('div');
      sub.className = 'sub';
      const kind = L.kind === 'video' ? `video${L.has_alpha ? '+α' : ''} ${L.coded_width}×${L.coded_height}` : 'still';
      sub.textContent = `${kind} · z ${L.z} · ${L.blend} · [${L.start_frame}, ${L.end_frame})`;
      const op = document.createElement('div');
      op.className = 'opacity';
      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = '0';
      slider.max = '100';
      slider.value = String(Math.round(st.opacity * 100));
      slider.title = 'opacity';
      const pct = document.createElement('span');
      pct.textContent = `${slider.value}%`;
      slider.addEventListener('input', () => {
        pct.textContent = `${slider.value}%`;
        player.setLayerOpacity(i, Number(slider.value) / 100);
      });
      op.append(slider, pct);
      li.append(vis, name, sub, op);
      layersEl.append(li);
    }
  };

  const syncCheckboxes = () => {
    for (const li of layersEl.children as HTMLCollectionOf<HTMLLIElement>) {
      const i = Number(li.dataset.index);
      const box = li.querySelector('input[type=checkbox]') as HTMLInputElement | null;
      if (box) box.checked = player.layerStates[i]?.visible ?? false;
    }
    updateLayerRows();
  };

  const updateLayerRows = () => {
    const meta = player.source?.meta;
    if (!meta) return;
    const f = player.currentFrame;
    for (const li of layersEl.children as HTMLCollectionOf<HTMLLIElement>) {
      const i = Number(li.dataset.index);
      li.classList.toggle('inactive', !isActive(meta.layers[i], f));
      li.classList.toggle('hidden', !player.layerStates[i].visible);
    }
  };

  // ---- readouts -----------------------------------------------------------------------------------
  const updateTransport = () => {
    const meta = player.source?.meta;
    btnPlay.classList.toggle('playing', player.isPlaying);
    if (!meta) {
      seek.max = '0';
      timeEl.textContent = durationEl.textContent = formatTime(0);
      frameEl.textContent = framesEl.textContent = '0';
      return;
    }
    const f = player.mode === 'seeking' ? player.targetFrame : player.currentFrame;
    timeEl.textContent = formatTime(ptsUs(f, meta.fps));
    frameEl.textContent = String(player.currentFrame);
    if (!scrubbing) seek.value = String(f);
  };

  let statusTimer = 0;
  const updateStatus = () => {
    window.clearTimeout(statusTimer);
    const m = player.mode;
    dropHint.hidden = m !== 'empty';
    status.classList.toggle('error', m === 'error');
    if (m === 'error') {
      status.hidden = false;
      status.textContent = `Error: ${player.error?.message ?? 'unknown'}`;
    } else if (m === 'buffering' || m === 'loading' || m === 'ended') {
      status.hidden = false;
      status.textContent = m === 'buffering' ? 'buffering…' : m === 'loading' ? 'loading…' : 'end';
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

  const updateBanner = () => {
    const n = player.stats.p6Failures + player.stats.p1Failures;
    banner.hidden = n === 0;
    banner.textContent = n ? `⚠ SYNC ASSERTION FAILED ×${n} — ${player.lastSyncFailure}` : '';
  };

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
    stage.style.aspectRatio = `${meta.canvas.width} / ${meta.canvas.height}`;
    seek.max = String(meta.frame_count - 1);
    durationEl.textContent = formatTime(ptsUs(meta.frame_count, meta.fps));
    framesEl.textContent = String(meta.frame_count - 1);
    buildLayers();
    updateStatus();
    updateBanner();
    player.audio?.setVolume(mute.checked ? 0 : 1);
  });
  player.addEventListener('state', () => {
    updateStatus();
    updateTransport();
  });
  player.addEventListener('frame', () => {
    updateTransport();
    updateLayerRows();
  });
  player.addEventListener('layers', syncCheckboxes);
  player.addEventListener('sync-failure', updateBanner);
  window.setInterval(updateDebug, 250);
  debug.addEventListener('toggle', updateDebug);
  updateStatus();
  updateTransport();
}
