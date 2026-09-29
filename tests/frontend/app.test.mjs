import test from 'node:test';
import assert from 'node:assert/strict';
import { load } from './load.mjs';

const state = { clips: [], skew: 0 };
const F = load(['esc', 'fmtBytes', 'fmtDur', 'tc', 'title', 'prettyName', 'clipName',
  'STATUS_LABEL', 'tidyLog', 'fmtClock', 'STAGE_TEXT', 'stageLabel', 'stageDetail'], { state });

test('esc escapes HTML', () => {
  assert.equal(F.esc('<b a="1">&</b>'), '&lt;b a=&quot;1&quot;&gt;&amp;&lt;/b&gt;');
  assert.equal(F.esc(null), '');
});

test('fmtBytes / fmtDur / tc', () => {
  assert.equal(F.fmtBytes(0), '—');
  assert.equal(F.fmtBytes(3_267_909), '3.3 MB');
  assert.equal(F.fmtBytes(2_500_000_000), '2.50 GB');
  assert.equal(F.fmtDur(9.4), '9.4s');
  assert.equal(F.fmtDur(125), '2:05');
  assert.equal(F.tc(6.2), '00:06.2');
});

test('prettyName strips prefix, extension, underscores', () => {
  assert.equal(F.prettyName('01_ALPHA_TITANS_20JUL_polished.mp4'), 'ALPHA TITANS 20JUL polished');
  assert.equal(F.prettyName('ALPHA_TITANS_horizontal.mp4'), 'ALPHA TITANS horizontal');
  assert.equal(F.prettyName(''), 'Untitled');
});

test('clipName drops the shared brand prefix, keeps others', () => {
  assert.equal(F.clipName('02_ALPHA_TITANS_20JUL_stopmotion.mp4'), '20JUL stopmotion');
  assert.equal(F.clipName('01_Project 1.mp4'), 'Project 1');
  assert.equal(F.clipName('02_2026-09-20 09-29-17.mp4'), '2026-09-20 09-29-17');
});

test('tidyLog shortens Windows and Unix paths (bug 7)', () => {
  const win = '$ C:\\Users\\Akansha\\Downloads\\alpha_titans\\pipeline\\build_video.py probe --out C:\\Users\\Akansha\\Downloads\\alpha_titans\\outputs\\21a974ede531 --src C:\\Users\\Akansha\\Downloads\\alpha_titans\\uploads\\21a974ede531 --orient auto';
  assert.equal(F.tidyLog(win), '$ pipeline/build_video.py probe --out outputs/… --src uploads/… --orient auto');
  assert.equal(F.tidyLog('SOURCE   /home/x/alpha_titans/uploads/21a974ede531'), 'SOURCE   uploads/…');
  assert.equal(F.tidyLog('plain line, no paths'), 'plain line, no paths');
});

test('fmtClock', () => {
  assert.equal(F.fmtClock(0), '0:00');
  assert.equal(F.fmtClock(Infinity), '0:00');   // webm voice notes report Infinity
  assert.equal(F.fmtClock(75.9), '1:15');
});

test('every pipeline stage has friendly wording', () => {
  for (const id of ['probe', 'measure', 'vision', 'plan', 'direct', 'sheets', 'cut', 'join', 'music', 'master']) {
    assert.ok(F.stageLabel({ id }) && F.stageDetail({ id }), id);
    assert.doesNotMatch(F.stageDetail({ id }), /ffprobe|YDIF|freezedetect|CRF/);
  }
  assert.equal(F.stageLabel({ id: 'new', label: 'New Stage' }), 'New Stage');
});

/* ─────────────── progress estimate (bug 1: stuck at 2%) ─────────────── */

const E = load(['serverNow', 'stageExpect', 'peak', 'estimate'], { state });
const ORDER = ['probe', 'measure', 'vision', 'plan', 'direct', 'sheets', 'cut', 'join', 'music', 'master'];
// the real stage timings of build 21a974ede531 (377 s, 439 s of 60 fps footage)
const REAL = { probe: [0, 12.1], measure: [12.1, 254.6], vision: null, plan: [254.6, 255.5],
  direct: [255.5, 255.6], sheets: [255.6, 264.4], cut: [264.4, 315.3], join: [315.3, 338.3],
  music: [338.3, 362.6], master: [362.6, 377.4] };
const W = { probe: 4, measure: 20, vision: 14, plan: 3, direct: 2, sheets: 7, cut: 31, join: 15, music: 8, master: 12 };

function jobAt(t, id = 'sim') {
  const T0 = 1_000_000;
  const stages = ORDER.map(s => {
    const r = REAL[s];
    if (!r) return { id: s, status: t >= 254.6 ? 'skipped' : 'pending' };
    const status = t >= r[1] ? 'done' : t >= r[0] ? 'running' : 'pending';
    return { id: s, status, started: status !== 'pending' ? T0 + r[0] : null, finished: status === 'done' ? T0 + r[1] : null };
  });
  let d = 0;
  for (const s of stages) d += (s.status === 'done' || s.status === 'skipped') ? W[s.id] : s.status === 'running' ? W[s.id] * .45 : 0;
  return { T0, job: { id, status: 'running', started: T0, progress: Math.round(1000 * d / 116) / 10, settings: {}, stages,
    analysis: { total_duration: 439, target: 120, planned_total: t > 255.5 ? 23.6 : null } } };
}

test('estimate keeps moving during the long motion-analysis stage', () => {
  const pcts = [];
  for (let t = 20; t <= 240; t += 20) {
    const { T0, job } = jobAt(t, 'a');
    state.skew = (T0 + t) - Date.now() / 1000;
    pcts.push(E.estimate(job).pct);
  }
  const server = jobAt(200).job.progress;
  assert.equal(server, 11.2, 'server figure really is flat here');
  assert.ok(pcts[pcts.length - 1] - pcts[0] > 25, `ring must climb, got ${pcts.map(p => p.toFixed(0))}`);
});

test('estimate never goes backwards and stays under 100 while running', () => {
  let last = 0;
  for (let t = 1; t <= 376; t += 3) {
    const { T0, job } = jobAt(t, 'b');
    state.skew = (T0 + t) - Date.now() / 1000;
    const { pct, left } = E.estimate(job);
    assert.ok(pct >= last - 1e-9, `went back at t=${t}: ${last} -> ${pct}`);
    assert.ok(pct < 100 && left >= 0);
    last = pct;
  }
  assert.ok(last > 90);
});

test('time-left estimate is close once analysis has finished', () => {
  for (const t of [280, 320, 355]) {
    const { T0, job } = jobAt(t, 'c' + t);
    state.skew = (T0 + t) - Date.now() / 1000;
    const { left } = E.estimate(job);
    const actual = 377.4 - t;
    assert.ok(Math.abs(left - actual) < Math.max(25, actual * 0.4), `t=${t}: est ${left.toFixed(0)} vs ${actual.toFixed(0)}`);
  }
});

test('estimate: done = 100%, draft uses server figure, null job safe', () => {
  assert.equal(E.estimate({ status: 'done', started: 1, finished: 11, stages: [] }).pct, 100);
  assert.equal(E.estimate({ status: 'draft', progress: 0, stages: [] }).pct, 0);
  assert.equal(E.estimate(null).pct, 0);
});

test('estimate tolerates a browser clock that disagrees with the server', () => {
  const { T0, job } = jobAt(150, 'skew');
  state.skew = (T0 + 150) - Date.now() / 1000;
  const good = E.estimate(job).pct;
  state.skew = 0;                                  // no Date header seen yet
  const bad = E.estimate({ ...job, id: 'skew2' });
  assert.ok(Number.isFinite(bad.pct) && bad.pct >= 0 && bad.pct < 100);
  assert.ok(good > 0);
});

/* ─────────────── voice recording format (bug 6) ─────────────── */

test('recFormat picks the first container the browser supports', () => {
  const plain = o => JSON.parse(JSON.stringify(o));   // sandbox objects are from another realm
  const pick = supported => plain(load(['recFormat'], { window: { MediaRecorder: { isTypeSupported: m => supported.includes(m) } } }).recFormat());
  assert.deepEqual(pick(['audio/webm;codecs=opus', 'audio/mp4']), { mime: 'audio/webm;codecs=opus', ext: 'webm' });
  assert.deepEqual(pick(['audio/mp4']), { mime: 'audio/mp4', ext: 'm4a' });              // Safari
  assert.deepEqual(pick(['audio/ogg;codecs=opus']), { mime: 'audio/ogg;codecs=opus', ext: 'ogg' });
  assert.deepEqual(plain(load(['recFormat'], { window: {} }).recFormat()), { mime: '', ext: 'webm' });
});

/* ─────────────── wording (Quality question, bug 3) ─────────────── */

test('quality options explain their use case', () => {
  const Q = load(['QUALITY_LABEL', 'QUALITY_NOTE']);
  assert.deepEqual(Object.keys(Q.QUALITY_LABEL), ['draft', 'standard', 'high', 'master']);
  assert.match(Q.QUALITY_LABEL.high[1], /YouTube/);
  assert.match(Q.QUALITY_LABEL.standard[1], /WhatsApp/);
  for (const k in Q.QUALITY_NOTE) assert.doesNotMatch(Q.QUALITY_NOTE[k], /CRF/);
});
