import { frameStats } from './decode/frames';
import { Player } from './player';
import { readBarcodes, readRow, recordDraws } from './testing';
import { bindUi } from './ui/app';

const params = new URLSearchParams(location.search);
const hw = params.get('hw') as HardwareAcceleration | null;
// Testing aids: ?queue=N shrinks the decode queue, ?slowread=ms&readahead=N simulate slow storage.
const queue = Number(params.get('queue')) || 0;
const slowRead = Number(params.get('slowread')) || 0;
const readAhead = Number(params.get('readahead')) || 0;

if (!('VideoDecoder' in window) || !('AudioDecoder' in window)) {
  document.getElementById('drop-hint')!.innerHTML =
    '<div>This browser has no WebCodecs.</div><small>LVF needs desktop Chrome or Edge (served over https or localhost).</small>';
} else {
  const player = new Player(document.getElementById('view') as HTMLCanvasElement, {
    pipeline: {
      ...(hw ? { hardwareAcceleration: hw } : {}),
      ...(queue ? { maxQueued: queue, hardCap: queue } : {}),
      ...(slowRead ? { readDelayMs: slowRead } : {}),
      ...(readAhead ? { readAheadFrames: readAhead } : {}),
    },
    noAudio: params.has('noaudio'),
  });
  bindUi(player);
  // `fflv view` opens ?src=/media/<file>&watch=1: load it and follow changes to the file.
  const src = params.get('src');
  if (src) {
    void player.open(src, params.get('name') ?? undefined);
    if (params.get('watch') === '1') player.watch(true);
  }
  // Console / end-to-end test hook.
  Object.assign(window, { __lvf: { player, frameStats, readBarcodes, readRow, recordDraws } });
}
