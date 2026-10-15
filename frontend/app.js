/* Alpha Titans — Content Engine, front end.

   Every value rendered here came from the API. There is no sample data, no
   simulated progress and no placeholder state: the stage list, the timings,
   the log and the output metadata are whatever the pipeline actually
   reported. Where a capability does not exist (speech-to-text), the UI says
   so rather than showing a control that quietly does nothing.

   Flow: 1 Upload Clips -> 2 Customize -> 3 AI Processing -> 4 Final Video. */

/* The page markup and this script ship together. If a browser or a copy
   step ever pairs this script with an older index.html, a missing #id must
   not crash the app: it gets a detached placeholder (so assignments are
   harmless), a warning is logged once, and the user is told to reload. */
// versioned like the other assets: a cached copy of this module after
// an update would leave the page running last release's studio
import { initStudio } from './studio.js?v=24';

/* The studio shell (overview, production workspace, agents, runs, health).
   Lazy: it makes no requests until a studio view is first shown. The create
   wizard below is unchanged - studio views and the wizard simply take turns
   owning the screen. */
let studio = null;

function showStudio(view) {
  const v = document.querySelector('#sxView');
  const app = document.querySelector('.app');
  if (!v || !studio) return;
  app.hidden = true;
  v.classList.add('on');
  studio.show(view);
}

function showCreate() {
  const v = document.querySelector('#sxView');
  const app = document.querySelector('.app');
  if (v) v.classList.remove('on');
  if (app) app.hidden = false;
  if (studio) studio.hide();
  goto(state.step || 1);
}

const UI_VERSION = '5';
const _missing = new Set();
const $ = (s, r = document) => {
  const e = r.querySelector(s);
  if (e || r !== document || !/^#[\w-]+$/.test(s)) return e;
  if (!_missing.has(s)) { _missing.add(s); console.warn('Alpha Titans: page is missing', s); }
  return document.createElement('div');
};
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const state = {
  health: null,
  jobId: null,
  job: null,
  clips: [],
  tracks: [],
  es: null,
  step: 1,
  reached: 1,            // furthest step unlocked for this job
  settings: {
    orientation: 'auto', resolution: '1080p', quality: 'high',
    speed: 'normal', speed_custom: 1, transition_style: 'auto',
    transition_duration: null, mood: 'auto', music_source: 'generate',
    keep_audio: 'none', sequence: 'hook',
  },
};

/* ─────────────────────────────── plumbing ─────────────────────────────── */

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const d = Date.parse(res.headers.get('date') || '');
  if (!Number.isNaN(d)) state.skew = (d / 1000) - (Date.now() / 1000);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }
  if (!res.ok) {
    const msg = (data && (data.detail || data.error)) || `${res.status} ${res.statusText}`;
    const err = new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
    if (res.status === 404 && /^No job /.test(err.message)) {
      err.jobLost = true;
      const lostId = err.message.slice(7).trim();
      if (lostId && lostId === state.jobId) jobLost();
    }
    throw err;
  }
  return data;
}

/* The server keeps jobs in memory and on disk under outputs/<id>. If it was
   restarted, or those folders were cleared, the id this page holds is gone.
   Start over cleanly instead of failing every call after it. */
function jobLost() {
  state.suppressUntil = Date.now() + 2500;   // swallow the follow-on errors
  state.es?.close();
  state.jobId = null;
  state.job = null;
  state.clips = [];
  state.reached = 1;
  setHidden('#sessionNotice', false);
  $('#outputBody').hidden = true;
  $('#outputEmpty').hidden = false;
  $('#toStep4').disabled = true;
  resetRun();
  renderClips();
  loader.fail();
  goto(1);
  toast('That session expired — please upload your clips again.', 'bad', true);
  loadBuilds();
}

/* ─────────────────────────────── loader ──────────────────────────────── */

/* One overlay with a moving ring, used for anything that makes you wait.
   It times every task: elapsed while running, remaining when progress is
   known, and "done in" once it finishes. */
const loader = (() => {
  const C = 2 * Math.PI * 50;
  let t0 = 0, tick = null, showT = null, pct = null, token = 0;
  const fmt = ms => ms < 60000 ? (ms / 1000).toFixed(1) + 's'
    : `${Math.floor(ms / 60000)}m ${String(Math.floor(ms / 1000) % 60).padStart(2, '0')}s`;
  const el = id => document.getElementById(id);
  function paint() {
    const ms = performance.now() - t0;
    el('lrElapsed').textContent = fmt(ms);
    if (pct != null && pct >= 2 && pct < 100) {
      el('lrEtaWrap').hidden = false;
      el('lrEta').textContent = fmt(ms * (100 - pct) / pct);
    } else el('lrEtaWrap').hidden = true;
  }
  const noop = { progress() {}, step() {}, done() { return 0; }, error() {} };
  function open(title, sub = '', { delay = 150 } = {}) {
    if (!el('loader') || !el('lrRing')) return noop;   // page markup out of date
    const my = ++token;
    clearTimeout(showT); clearInterval(tick);
    t0 = performance.now(); pct = null;
    const box = el('loader');
    el('lrTitle').textContent = title;
    el('lrSub').textContent = sub;
    el('lrRetry').hidden = true;
    el('lrRing').className = 'lring spin';
    el('lrFg').style.strokeDasharray = C;
    el('lrFg').style.strokeDashoffset = C * .72;
    el('lrPct').textContent = '';
    paint();
    tick = setInterval(paint, 100);
    const show = () => { box.hidden = false; box.classList.remove('out'); };
    if (!box.hidden && !box.classList.contains('out')) show();
    else if (delay) showT = setTimeout(() => my === token && show(), delay);
    else show();
    return {
      progress(p, detail) {
        if (my !== token) return;
        pct = Math.max(0, Math.min(100, p));
        el('lrRing').className = 'lring';
        el('lrFg').style.strokeDashoffset = C * (1 - pct / 100);
        el('lrPct').textContent = Math.round(pct) + '%';
        if (detail != null) el('lrSub').textContent = detail;
        paint();
      },
      step(title2, sub2 = '') {           // back to an indeterminate spin
        if (my !== token) return;
        pct = null;
        el('lrTitle').textContent = title2;
        el('lrSub').textContent = sub2;
        el('lrRing').className = 'lring spin';
        el('lrFg').style.strokeDashoffset = C * .72;
        el('lrPct').textContent = '';
        paint();
      },
      done(msg) {
        if (my !== token) return 0;
        const ms = performance.now() - t0;
        clearTimeout(showT); clearInterval(tick);
        if (box.hidden) return ms;          // too quick to have been shown
        el('lrRing').className = 'lring ok';
        el('lrFg').style.strokeDashoffset = 0;
        el('lrPct').textContent = '✓';
        el('lrEtaWrap').hidden = true;
        el('lrElapsed').textContent = fmt(ms);
        el('lrSub').textContent = (msg || 'Done') + ' in ' + fmt(ms);
        setTimeout(() => { if (my === token) hide(); }, 550);
        return ms;
      },
      error(title2, sub2) {
        if (my !== token) return;
        clearTimeout(showT); clearInterval(tick);
        show();
        el('lrRing').className = 'lring bad';
        el('lrFg').style.strokeDashoffset = 0;
        el('lrPct').textContent = '!';
        el('lrTitle').textContent = title2;
        el('lrSub').textContent = sub2 || '';
        el('lrEtaWrap').hidden = true;
        el('lrRetry').hidden = false;
      },
    };
  }
  function hide() {
    const box = el('loader');
    if (!box) return;
    box.classList.add('out');
    setTimeout(() => { if (box.classList.contains('out')) { box.hidden = true; box.classList.remove('boot'); } }, 300);
  }
  function fail() { token++; clearTimeout(showT); clearInterval(tick); hide(); }
  return { open, fail, fmt };
})();

function toast(msg, kind = '', force = false) {
  if (!force && kind === 'bad' && Date.now() < (state.suppressUntil || 0)) return;
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $('#toasts').append(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 280); }, 5200);
}

/* tolerant of an element that is missing, e.g. a stale cached index.html */
const setHidden = (sel, v) => { const e = $(sel); if (e) e.hidden = v; };

const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const fmtBytes = b => !b ? '—'
  : b > 1e9 ? (b / 1e9).toFixed(2) + ' GB'
    : b > 1e6 ? (b / 1e6).toFixed(1) + ' MB' : Math.round(b / 1e3) + ' KB';

const fmtDur = s => {
  if (s == null) return '—';
  const m = Math.floor(s / 60), r = s % 60;
  return m ? `${m}:${String(Math.floor(r)).padStart(2, '0')}` : `${(+r).toFixed(1)}s`;
};

const ago = ts => {
  if (!ts) return '';
  const d = (Date.now() / 1000) - ts;
  if (d < 60) return 'just now';
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
};

/* mm:ss.t — a duration in a bin reads as timecode, not "11.0s" */
const tc = s => {
  if (s == null) return '—';
  const m = Math.floor(s / 60), r = s - m * 60;
  return `${String(m).padStart(2, '0')}:${r.toFixed(1).padStart(4, '0')}`;
};

const title = s => String(s || '').replace(/^./, c => c.toUpperCase());

/* file and build names read as titles, not as file system entries */
const prettyName = s => String(s || 'Untitled')
  .replace(/\.[a-z0-9]{2,4}$/i, '')
  .replace(/^\d{2,3}_/, '')
  .replace(/[_]+/g, ' ')
  .replace(/\s+/g, ' ').trim() || 'Untitled';

/* clip cards drop the brand prefix every upload shares, so the part that
   tells them apart is what shows */
const clipName = n => prettyName(n).replace(/^alpha titans\s+/i, '') || prettyName(n);

const STATUS_LABEL = { done: 'Ready', running: 'Creating', error: 'Failed',
                       draft: 'Draft', cancelled: 'Cancelled', queued: 'Queued' };
const ORIENT_LABEL = { horizontal: 'Landscape', vertical: 'Vertical' };

/* ──────────────────────────────── steps ───────────────────────────────── */

function goto(step) {
  state.step = step;
  state.reached = Math.max(state.reached, step);
  $$('.step').forEach(el => el.classList.toggle('on', el.id === `step-${step}`));
  renderStepper();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function renderStepper() {
  const running = state.job?.status === 'running';
  $$('#stepper li').forEach(li => {
    const n = +li.dataset.step;
    li.classList.toggle('on', n === state.step);
    li.classList.toggle('done', n < state.step && n <= state.reached);
    li.classList.toggle('busy', n === 3 && running && state.step !== 3);
    li.onclick = (n <= state.reached && n !== state.step) ? () => goto(n) : null;
    li.style.cursor = (n <= state.reached && n !== state.step) ? 'pointer' : '';
  });
}

/* ──────────────────────────────── boot ────────────────────────────────── */

async function boot() {
  wireChrome();
  $('#lrRetry').onclick = () => location.reload();
  const pageVer = document.querySelector('meta[name="at-ui"]')?.content;
  if (pageVer !== UI_VERSION) {
    toast('This page is out of date — press Ctrl+F5 to reload it.', 'bad', true);
  }
  const task = loader.open('Starting the studio', 'Connecting to Alpha Titans…', { delay: 0 });
  try {
    state.health = await api('/api/health');
  } catch (e) {
    $('#health').className = 'pill pill-bad';
    $('#health').textContent = 'API unreachable';
    task.error("Can't reach the studio server",
      'Make sure the server is running, then try again. (' + e.message + ')');
    return;
  }
  task.step('Starting the studio', 'Loading your videos and music…');
  renderHealth();
  buildControls();
  wireStudio();
  goto(1);
  // No job is created on load. An empty job would show up in Builds as a
  // phantom draft; the first upload (or script, or voice note) creates it.
  await Promise.all([loadBuilds(), loadTracks()]);
  task.done('Ready');
}

/** Every entry point that needs somewhere to put data calls this first. */
async function ensureJob() {
  if (!state.jobId) await newBuild({ quiet: true });
  return state.jobId;
}

function wireChrome() {
  studio = initStudio({ onCreateClick: () => { showCreate(); newBuild(); } });
  // the wizard starts hidden behind the studio overview; the brand mark and
  // every sidebar entry switch between them
  if (studio) showStudio('create');
  $('#newBuild').onclick = () => { showCreate(); newBuild(); };
  $('#brandBtn').onclick = openDrawer;
  $('#libraryBtn').onclick = openDrawer;
  $('#drawerClose').onclick = closeDrawer;
  $('#drawerScrim').onclick = closeDrawer;
  $('#toStep2').onclick = () => goto(2);
  $('#backTo1').onclick = () => goto(1);
  $('#backTo2').onclick = () => goto(2);
  $('#backTo3').onclick = () => goto(3);
  $('#toStep4').onclick = () => goto(4);
  $('#againBtn').onclick = () => newBuild();
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !$('#drawer').hidden) closeDrawer();
  });
}

function renderHealth() {
  const c = state.health.capabilities;
  const el = $('#health');
  const missing = [];
  if (!c.pipeline) missing.push('pipeline');
  if (!c.ffmpeg) missing.push('ffmpeg');
  if (missing.length) {
    el.className = 'pill pill-bad';
    el.textContent = 'Setup incomplete — missing ' + missing.join(', ');
  } else {
    el.className = 'pill pill-ok';
    el.textContent = c.beat_engine ? 'Studio online · all systems ready'
      : 'Studio online · basic music engine';
    if (!c.beat_engine) el.title = 'The full soundtrack engine needs: pip install soundfile numpy scipy';
  }
  if (c.beat_engine || missing.length) el.title = Object.entries(c).map(([k, v]) => `${k}: ${v}`).join('\n');

  const noticeBox = $('#transcribeNotice');
  if (noticeBox) {
    if (c.transcription) {
      noticeBox.className = 'notice notice-info';
      const icon = $('.notice-ico', noticeBox);
      if (icon) icon.textContent = '✓';
      const strong = $('strong', noticeBox);
      if (strong) strong.textContent = 'Voice transcription is ready.';
      $('#transcribeWhy').textContent = state.health.notes?.transcription
        || 'Offline speech-to-text engine is ready.';
    } else {
      noticeBox.className = 'notice notice-warn';
      const icon = $('.notice-ico', noticeBox);
      if (icon) icon.textContent = '!';
      const strong = $('strong', noticeBox);
      if (strong) strong.textContent = 'Voice transcription is unavailable.';
      $('#transcribeWhy').textContent = state.health.notes?.transcription
        || 'No speech-to-text engine is installed, so voice notes are stored and '
         + 'played back but not transcribed.';
    }
  }

  // the bin bar states the accepted formats and the per-file cap
  const mb = state.health.limits?.clip_mb;
  const fmts = $('#binFormats');
  if (fmts && mb) fmts.textContent =
    `MOV · MP4 · M4V · AVI · MKV · up to ${mb >= 1024 ? (mb / 1024) + ' GB' : mb + ' MB'} each`;
}

/* ─────────────────────── controls built from /api/health ──────────────── */

const RES_NOTE = { '720p': 'HD', '1080p': 'Full HD',
                   '1440p': 'Quad HD', '4k': 'Ultra HD' };
const QUALITY_LABEL = {
  draft:    ['Draft',    'quick preview'],
  standard: ['Standard', 'WhatsApp · social'],
  high:     ['High',     'YouTube · recommended'],
  master:   ['Master',   'archive · re-editing'],
};
const QUALITY_NOTE = {
  draft: 'Draft: renders fastest with the smallest file. Use it to check the cut before a final render; fine detail will look soft.',
  standard: 'Standard: a smaller file that still looks good on phones. Good for WhatsApp, Instagram and quick sharing.',
  high: 'High (default): crisp bricks and studs at a sensible file size. Best for YouTube and most uploads.',
  master: 'Master: keeps the most detail, with the biggest file and the longest render. Use it for a copy you will keep or edit again later.',
};
const TRANSITION_LABEL = {
  auto: ['AI picks', 'chosen per cut'],
  cut: ['Clean', 'simple crossfades'],
  smooth: ['Smooth', 'gentle slides & fades'],
  dynamic: ['Dynamic', 'flashes & wipes'],
  flash: ['Flash', 'bright, punchy hits'],
};
const MOOD_LABEL = { auto: 'AI picks', lofi: 'Lo-fi' };

/* the pipeline's own stage names, said the way a creator would say them */
const STAGE_TEXT = {
  probe:   ['Reading your clips',     'Checking every upload and removing duplicates'],
  measure: ['Finding the action',     'Spotting real movement and skipping dead air'],
  vision:  ['Watching the footage',   'AI vision looks at real frames from each clip'],
  plan:    ['Picking the best moments','Scoring every moment and sharing out the running time'],
  direct:  ['Directing the edit',     'Choosing transitions, pacing and mood for each cut'],
  sheets:  ['Storyboarding',          'Building a preview of every planned scene'],
  cut:     ['Rendering scenes',       'Rendering each scene with its caption'],
  join:    ['Adding transitions',     'Blending the scenes into one continuous video'],
  music:   ['Composing the soundtrack','Timing the music to every cut'],
  master:  ['Final polish',           'Balancing loudness, creating a thumbnail and checking every frame'],
};
const stageLabel = s => (STAGE_TEXT[s.id] || [s.label])[0];
const stageDetail = s => (STAGE_TEXT[s.id] || [null, s.detail])[1];
const MOOD_NOTE = {
  auto: 'AI chooses a mood from the energy of your footage.',
  beat: 'Full drums, bass and a melodic hook — impacts land on every cut.',
  energetic: 'Driving and high-energy — impacts land on every cut.',
  cinematic: 'Big and dramatic — made for reveals.',
  lofi: 'Soft, relaxed bed with gentle accents on the cuts.',
  calm: 'Calm and unhurried, with gentle accents on the cuts.',
  bright: 'Light and cheerful, with bright accents on the cuts.',
  upbeat: 'Bouncy and positive, with bright accents on the cuts.',
};

const SEQUENCE_LABEL = {
  hook:     ['Hook first', 'liveliest clip opens'],
  recorded: ['Build order', 'when each was filmed'],
  uploaded: ['My order',   'exactly as arranged'],
};

const KEEP_AUDIO_LABEL = {
  none: ['Drop it', 'music bed only'],
  mix:  ['Keep + music', 'speech over a ducked score'],
  only: ['Keep it only', 'no music at all'],
};

function buildControls() {
  const o = state.health.options;

  $('#sequence').innerHTML = (o.sequence || ['hook', 'recorded', 'uploaded']).map(k => {
    const [t, sub] = SEQUENCE_LABEL[k] || [k, ''];
    return `<button data-v="${k}"><b>${t}</b><em>${sub}</em></button>`;
  }).join('');

  $('#keepAudio').innerHTML = (o.keep_audio || ['none', 'mix', 'only']).map(k => {
    const [t, s] = KEEP_AUDIO_LABEL[k] || [k, ''];
    return `<button data-v="${k}"><b>${t}</b><em>${s}</em></button>`;
  }).join('');

  $('#resolution').innerHTML = o.resolution.map(r =>
    `<button data-v="${r}"><b>${r === '4k' ? '4K' : r}</b><em>${RES_NOTE[r] || ''}</em></button>`
  ).join('');

  $('#quality').innerHTML = o.quality.map(q => {
    const [n, sub] = QUALITY_LABEL[q] || [title(q), ''];
    return `<button data-v="${q}"><b>${n}</b><em>${sub}</em></button>`;
  }).join('');

  $('#transition').innerHTML = o.transition_style.map(t => {
    const [n, sub] = TRANSITION_LABEL[t] || [title(t), ''];
    return `<button data-v="${t}"><b>${n}</b><em>${sub}</em></button>`;
  }).join('');

  $('#mood').innerHTML = o.mood.map(m =>
    `<button data-v="${m}">${MOOD_LABEL[m] || title(m)}</button>`).join('');

  qualityHint();
  moodHint();
  keepAudioHint();
  sequenceHint();
  visionNotice();
  renderAgents();
}

const qualityHint = () =>
  $('#qualityHint').textContent = QUALITY_NOTE[state.settings.quality] || '';
const moodHint = () =>
  $('#moodHint').textContent = MOOD_NOTE[state.settings.mood] || '';
const sequenceHint = () =>
  $('#sequenceHint').textContent =
    (state.health?.sequence_detail || {})[state.settings.sequence] || '';
const keepAudioHint = () =>
  $('#keepAudioHint').textContent =
    (state.health?.keep_audio_detail || {})[state.settings.keep_audio] || '';

/* a picker is any container whose children carry data-v */
function picker(sel, fn) {
  const root = $(sel);
  const sync = () => $$('button', root).forEach(b =>
    b.classList.toggle('on', b.dataset.v === root.dataset.value));
  $$('button', root).forEach(b => b.onclick = () => {
    root.dataset.value = b.dataset.v;
    sync();
    fn(b.dataset.v);
    summarise();
  });
  sync();
}

function setPicker(sel, value) {
  const root = $(sel);
  if (!root || value == null) return;
  root.dataset.value = String(value);
  $$('button', root).forEach(b =>
    b.classList.toggle('on', b.dataset.v === String(value)));
}

/* ────────────────────────────── builds list ───────────────────────────── */

function openDrawer() { $('#drawer').hidden = false; loadBuilds(); }
function closeDrawer() { $('#drawer').hidden = true; }

async function loadBuilds() {
  const host = $('#builds');
  let list = [];
  try { list = (await api('/api/jobs')).jobs || []; }
  catch (e) { host.innerHTML = `<div class="empty">Could not load builds: ${esc(e.message)}</div>`; return; }

  const done = list.filter(j => j.status === 'done').length;
  $('#buildsSub').textContent = list.length
    ? `${done} finished · ${list.length} total`
    : 'Nothing here yet';

  if (!list.length) {
    host.innerHTML = `<div class="empty">No videos yet. Upload some LEGO clips to
      create your first one.</div>`;
    return;
  }
  host.innerHTML = list.map(j => `
    <article class="buildcard" data-id="${j.id}">
      <div class="thumb" ${j.thumbnail
        ? `style="background-image:url(/api/jobs/${j.id}/file/${encodeURIComponent(j.thumbnail)})"`
        : ''}>${j.thumbnail ? '' : '<svg class="brick"><use href="#i-brick"/></svg>'}
        <span class="badge ${j.status}">${STATUS_LABEL[j.status] || j.status}${j.status === 'running' ? ' ' + j.progress + '%' : ''}</span>
      </div>
      <div class="body">
        <b>${esc(j.title === 'Untitled build' ? 'Untitled video' : prettyName(j.title))}</b>
        <small>${j.status === 'done' || j.clips
          ? `${j.clips} clip${j.clips === 1 ? '' : 's'}${j.duration ? ' · ' + fmtDur(j.duration) : ''}`
          : 'Not created yet'} · ${ago(j.created)}</small>
      </div>
    </article>`).join('');
  $$('.buildcard', host).forEach(c => c.onclick = () => {
    closeDrawer();
    openJob(c.dataset.id);
  });
}

/* ────────────────────────────── new build ─────────────────────────────── */

async function newBuild({ quiet = false } = {}) {
  const task = quiet ? null : loader.open('Starting a new video');
  try {
    const r = await api('/api/jobs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state.settings),
    });
    state.es?.close();
    state.jobId = r.id;
    state.job = r.job;
    state.clips = [];
    state.reached = 1;
    $('#script').value = '';
    $('#title').value = '';
    $('#voiceBox').hidden = true;
    $('#captionPreview').hidden = true;
    $('#log').textContent = '';
    delete $('#log').dataset.err;
    $('#outputBody').hidden = true;
    $('#outputEmpty').hidden = false;
    $('#toStep4').disabled = true;
    $('#startBuild').disabled = false;
    resetRun();
    renderClips();
    goto(1);
    setHidden('#sessionNotice', true);
    task?.done('New video ready');
  } catch (e) { task && loader.fail(); toast('Could not create a video: ' + e.message, 'bad'); }
}

/* ────────────────────────────── step 1: clips ─────────────────────────── */

function wireStudio() {
  $('#browseClips').onclick = e => { e.stopPropagation(); $('#clipInput').click(); };
  $('#uploader').onclick = e => {
    if (e.target.closest('#uploadProgress')) return;
    $('#clipInput').click();
  };
  $('#stageToggle').onclick = () => {
    const on = $('#stageRun').classList.toggle('all');
    $('#stageToggle').textContent = on ? 'Hide steps' : 'Show all steps';
  };
  $('#clipInput').onchange = e => uploadClips([...e.target.files]);

  /* The whole window is the drop target, not one box. dragenter/dragleave
     fire constantly as the pointer crosses child elements, so they are
     counted rather than toggled - otherwise the overlay flickers. */
  let dragDepth = 0;
  const overlay = $('#dropOverlay');
  const isFiles = e => [...(e.dataTransfer?.types || [])].includes('Files');
  const videoish = f => f.type.startsWith('video/')
                     || /\.(mov|mp4|m4v|avi|mkv)$/i.test(f.name);

  window.addEventListener('dragenter', e => {
    if (!isFiles(e)) return;
    e.preventDefault();
    if (++dragDepth === 1) {
      const n = e.dataTransfer.items ? e.dataTransfer.items.length : 0;
      $('#dropCount').textContent = n > 1
        ? `Release to upload ${n} clips`
        : 'Release to upload';
      overlay.hidden = false;
    }
  });
  window.addEventListener('dragover', e => { if (isFiles(e)) e.preventDefault(); });
  window.addEventListener('dragleave', e => {
    if (!isFiles(e)) return;
    if (--dragDepth <= 0) { dragDepth = 0; overlay.hidden = true; }
  });
  window.addEventListener('drop', e => {
    if (!isFiles(e)) return;
    e.preventDefault();
    dragDepth = 0; overlay.hidden = true;
    const files = [...(e.dataTransfer?.files || [])].filter(videoish);
    if (!files.length) return toast('No video files in that drop.', 'bad');
    if (state.step !== 1) goto(1);
    uploadClips(files);
  });

  $('#script').addEventListener('change', saveScript);
  $('#scriptFile').onchange = uploadScriptFile;
  $('#recordBtn').onclick = toggleRecord;

  picker('#orientation', v => state.settings.orientation = v);
  picker('#resolution', v => state.settings.resolution = v);
  picker('#quality', v => { state.settings.quality = v; qualityHint(); });
  picker('#transition', v => state.settings.transition_style = v);
  picker('#xfade', v => state.settings.transition_duration =
    v === 'auto' ? null : parseFloat(v));
  picker('#mood', v => { state.settings.mood = v; moodHint(); });
  picker('#sequence', v => {
    state.settings.sequence = v; sequenceHint(); timestampNotice();
  });
  picker('#keepAudio', v => {
    state.settings.keep_audio = v; keepAudioHint(); audioNotice();
  });
  picker('#speed', v => {
    state.settings.speed = v;
    $('#speedCustom').hidden = v !== 'custom';
  });
  picker('#musicSource', v => {
    state.settings.music_source = v;
    $('#moodWrap').hidden = v !== 'generate';
    $('#libraryWrap').hidden = v !== 'upload';
  });

  $('#speedRange').oninput = e => {
    state.settings.speed_custom = parseFloat(e.target.value);
    $('#speedOut').textContent = parseFloat(e.target.value).toFixed(2) + '×';
    summarise();
  };
  $('#target').onchange = e => {
    state.settings.target_seconds = e.target.value || null; summarise();
  };
  $('#title').onchange = e => state.settings.title = e.target.value.trim() || undefined;
  $('#musicFile').onchange = uploadTrack;
  $('#startBuild').onclick = startBuild;
  $('#cancelBuild').onclick = cancelBuild;
  $('#regenBtn').onclick = regenerate;
  $('#copyLog').onclick = () => {
    navigator.clipboard?.writeText($('#log').textContent)
      .then(() => toast('Pipeline output copied', 'ok'))
      .catch(() => toast('Clipboard refused', 'bad'));
  };
  summarise();
}

async function uploadClips(files) {
  if (!files.length) return;
  await ensureJob();
  const box = $('#uploadProgress');
  box.hidden = false;
  const bar = $('i', box), label = $('span', box);
  const fd = new FormData();
  files.forEach(f => fd.append('files', f, f.name));

  const n = files.length;
  const task = loader.open(`Uploading ${n} clip${n === 1 ? '' : 's'}`, 'Starting upload…', { delay: 0 });
  const tUp = performance.now();
  let lost = false;
  await new Promise(resolve => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api/jobs/${state.jobId}/clips`);
    xhr.upload.onprogress = e => {
      if (!e.lengthComputable) return;
      const pct = Math.round(100 * e.loaded / e.total);
      bar.style.width = pct + '%';
      label.textContent = pct < 100 ? `Uploading… ${pct}%` : 'Preparing previews…';
      const secs = (performance.now() - tUp) / 1000;
      const rate = secs > 0.2 ? e.loaded / secs : 0;
      if (pct < 100) task.progress(100 * e.loaded / e.total,
        `${fmtBytes(e.loaded)} of ${fmtBytes(e.total)}${rate ? ' · ' + fmtBytes(rate) + '/s' : ''}`);
      else task.step('Preparing previews', 'Reading your clips and making thumbnails…');
    };
    xhr.onload = () => {
      if (xhr.status === 404 && /No job/.test(xhr.responseText)) {
        lost = true; jobLost(); resolve(); return;
      }
      try {
        const r = JSON.parse(xhr.responseText);
        (r.rejected || []).forEach(x => toast(`${x.name}: ${x.why}`, 'bad'));
        if (r.saved?.length) toast(`Added ${r.saved.length} clip${r.saved.length > 1 ? 's' : ''}`, 'ok');
      } catch { toast('Upload failed', 'bad'); }
      resolve();
    };
    xhr.onerror = () => { toast('Upload failed — network error', 'bad'); resolve(); };
    xhr.send(fd);
  });
  box.hidden = true;
  bar.style.width = '0';
  $('#clipInput').value = '';
  if (lost) return;
  task.step('Preparing previews', 'Reading your clips and making thumbnails…');
  await refreshClips();
  setHidden('#sessionNotice', true);
  task.done(`${n} clip${n === 1 ? '' : 's'} uploaded`);
}

async function refreshClips() {
  if (!state.jobId) return;
  try { state.clips = (await api(`/api/jobs/${state.jobId}/clips`)).clips || []; }
  catch (e) { toast('Could not list clips: ' + e.message, 'bad'); }
  renderClips();
}

/* A <video> that has streamed a clip keeps the file open on the server, and
   Windows refuses to rename an open file — which is what reorder does. */
function releaseClipVideos() {
  $$('#clipList video').forEach(v => {
    try { v.pause(); v.removeAttribute('src'); v.load(); } catch { /* ignore */ }
  });
}

function renderClips() {
  const host = $('#clipList');
  const n = state.clips.length;
  $('#clipsEmpty').hidden = n > 0;
  $('#step-1').classList.toggle('has-clips', n > 0);
  $('.bin').classList.toggle('has', n > 0);

  $('#clipTally').textContent = n
    ? `${n} clip${n === 1 ? '' : 's'} ready for AI editing`
    : 'Nothing uploaded yet';

  audioNotice();
  timestampNotice();
  summarise();
  $('#toStep2').disabled = !n;
  $('#step1Hint').textContent = n
    ? `${n} clip${n === 1 ? '' : 's'} ready — add more any time.`
    : 'Add at least one clip to continue.';
  $('#startBuild').disabled = !n;
  $('#startHint').textContent = n ? '' : 'Add clips first.';

  host.innerHTML = state.clips.map((c, i) => {
    const poster = `/api/jobs/${state.jobId}/poster/${encodeURIComponent(c.name)}`;
    return `
    <div class="clip ${c.orientation === 'vertical' ? 'v' : ''}" draggable="true"
         data-name="${esc(c.name)}" data-i="${i}" style="animation-delay:${Math.min(i, 12) * 40}ms">
      <span class="strip" style="--poster:url('${poster}')">
        <video src="/api/jobs/${state.jobId}/clip/${encodeURIComponent(c.name)}"
               poster="${poster}" preload="none" playsinline></video>
        <span class="ord">${i + 1}</span>
        <span class="playhint" aria-hidden="true"><svg class="i-play"><use href="#i-play"/></svg><svg class="i-pause" viewBox="0 0 24 24"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z" fill="currentColor"/></svg></span>
        <button class="iconbtn rm" title="Remove clip">×</button>
      </span>
      <div class="cbody">
        <div class="nm" title="${esc(c.name)}">${esc(clipName(c.name))}</div>
        <button class="infobtn" title="Clip details" aria-expanded="false">i</button>
      </div>
      <dl class="details">
        <dt>Length</dt><dd>${tc(c.duration)}</dd>
        <dt>Format</dt><dd>${ORIENT_LABEL[c.orientation] || '—'}${c.rotation ? ` · rot ${c.rotation}°` : ''}</dd>
        <dt>Resolution</dt><dd>${c.width || '?'}×${c.height || '?'}</dd>
        <dt>Codec</dt><dd>${esc(c.codec || '?')}</dd>
        <dt>Audio</dt><dd>${c.has_audio ? 'Yes' : 'Silent'}</dd>
        ${c.recorded ? `<dt>Recorded</dt><dd>${esc(String(c.recorded).slice(0, 16).replace('T', ' '))}</dd>` : ''}
        <dt>Size</dt><dd>${fmtBytes(c.size)}</dd>
        <dt>File</dt><dd>${esc(c.name)}</dd>
      </dl>
    </div>`;
  }).join('');

  $$('.clip', host).forEach(el => {
    const vid = $('video', el);
    const strip = $('.strip', el);
    const sync = () => el.classList.toggle('playing', !vid.paused);
    vid.addEventListener('play', sync);
    vid.addEventListener('pause', sync);
    vid.addEventListener('ended', sync);
    strip.onclick = ev => {
      if (ev.target.closest('.rm')) return;
      ev.stopPropagation();
      if (vid.paused) {
        // only one preview at a time
        $$('.clip video').forEach(o => { if (o !== vid) o.pause(); });
        vid.preload = 'auto';
        vid.play().catch(() => toast('This clip cannot be previewed in the browser.', 'bad'));
      } else vid.pause();
    };
    $('.infobtn', el).onclick = ev => {
      ev.stopPropagation();
      const open = el.classList.toggle('open');
      ev.currentTarget.setAttribute('aria-expanded', String(open));
    };
    $('.rm', el).onclick = async ev => {
      ev.stopPropagation();
      releaseClipVideos();
      try {
        await api(`/api/jobs/${state.jobId}/clips/${encodeURIComponent(el.dataset.name)}`,
          { method: 'DELETE' });
        await refreshClips();
      } catch (e) { toast('Remove failed: ' + e.message, 'bad'); }
    };
    el.ondragstart = e => {
      el.classList.add('drag');
      e.dataTransfer.setData('text/plain', el.dataset.i);
    };
    el.ondragend = () => el.classList.remove('drag');
    el.ondragover = e => { e.preventDefault(); el.classList.add('over'); };
    el.ondragleave = () => el.classList.remove('over');
    el.ondrop = async e => {
      e.preventDefault();
      el.classList.remove('over');
      const from = +e.dataTransfer.getData('text/plain');
      const to = +el.dataset.i;
      if (from === to || Number.isNaN(from)) return;
      const names = state.clips.map(c => c.name);
      names.splice(to, 0, names.splice(from, 1)[0]);
      releaseClipVideos();
      await new Promise(r => setTimeout(r, 250));
      try {
        await api(`/api/jobs/${state.jobId}/clips/order`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ order: names }),
        });
        await refreshClips();
      } catch (err) {
        toast('Could not reorder: ' + err.message, 'bad');
        await refreshClips();
      }
    };
  });
}

/* ──────────────────────────── step 2: settings ────────────────────────── */

/* "Build order" needs recording timestamps in the files. Screen recordings,
   re-encoded clips and anything stripped by an editor often have none, so say
   so up front rather than silently falling back to upload order. */
function timestampNotice() {
  const box = $('#noTimestamps');
  if (!box) return;
  const withTime = state.clips.filter(c => c.recorded).length;
  const show = state.settings.sequence === 'recorded'
            && state.clips.length > 0 && withTime < 2;
  box.hidden = !show;
  if (show) {
    box.innerHTML = `<span class="notice-ico">i</span><div>
      <strong>${withTime === 0 ? 'None of your clips carry' : 'Only one clip carries'} a recording timestamp.</strong>
      <span>Build order needs at least two to sort by. These files will fall
      back to the order you arranged them in — drag them into build order on
      step 1 and pick <em>My order</em> instead.</span></div>`;
  }
}

/* If the footage has audio and it is about to be thrown away, say so once,
   here, rather than letting someone find out from a silent tutorial. */
function audioNotice() {
  const box = $('#audioDetected');
  if (!box) return;
  const withAudio = state.clips.filter(c => c.has_audio).length;
  const show = withAudio > 0 && state.settings.keep_audio === 'none';
  box.hidden = !show;
  if (show) {
    box.innerHTML = `<span class="notice-ico">i</span><div>
      <strong>${withAudio} of ${state.clips.length} clip${state.clips.length === 1 ? '' : 's'} carry an audio track.</strong>
      <span>Right now that audio is dropped and replaced by the score. If this is
      a tutorial, an interview or anything where someone is speaking, pick
      <em>Keep + music</em> or <em>Keep it only</em>.</span></div>`;
  }
}

/* Whether the vision layer is live, said once, from /api/health - never
   implied. No key is a supported mode, not a fault. */
function visionNotice() {
  const box = $('#visionState');
  if (!box) return;
  const v = state.health?.vision || {};
  box.className = 'notice ' + (v.enabled ? 'notice-info' : 'notice-muted');
  box.innerHTML = v.enabled
    ? `<span class="notice-ico">◉</span><div>
        <strong>AI vision is on</strong>
        <span>${esc(v.provider)} / ${esc(v.model)} looks at ${v.frames_per_clip} real
        frames from each clip to understand the build, spot the best shots and
        choose a strong opening.</span>
       </div>`
    : `<span class="notice-ico">○</span><div>
        <strong>AI vision is off</strong>
        <span>${esc(v.why_disabled || 'No API key is configured.')}
        Your video is still planned from measured motion and timing. Set
        <code>AT_VISION_API_KEY</code> to let a model look at the footage too.</span>
       </div>`;
}

/* The five agents, named, with which ones use a model stated rather than
   implied. Built from /api/health so it cannot drift from the server. */
function renderAgents() {
  const host = $('#agentRoster');
  if (!host) return;
  const agents = state.health?.agents || [];
  const on = state.health?.vision?.enabled;
  host.innerHTML = agents.map(a => `
    <div class="agentcard ${a.uses_model ? 'model' : ''} ${a.uses_model && !on ? 'off' : ''}">
      <div class="acard-top">
        <b>${esc(a.name)}</b>
        ${a.uses_model
          ? `<span class="amodel">${on ? 'vision model' : 'needs API key'}</span>`
          : '<span class="arule">rule-based</span>'}
      </div>
      <p>${esc(a.does)}</p>
      <small>${a.stages.length} stage${a.stages.length === 1 ? '' : 's'}: ${a.stages.join(' · ')}</small>
    </div>`).join('');
}

function summarise() {
  const host = $('#customSummary');
  if (!host) return;
  const s = state.settings;
  const speed = s.speed === 'custom' ? (+s.speed_custom).toFixed(2) + '×'
    : ({ slow: 'Relaxed', normal: 'Natural', fast: 'Punchy' }[s.speed] || title(s.speed));
  const music = s.music_source === 'generate'
      ? 'AI · ' + (MOOD_LABEL[s.mood] || title(s.mood))
    : s.music_source === 'upload' ? (s.music_track ? prettyName(s.music_track) : 'My track') : 'None';
  const fmt = { auto: 'Auto', horizontal: 'Landscape', vertical: 'Vertical' }[s.orientation] || s.orientation;
  const trans = (TRANSITION_LABEL[s.transition_style] || [title(s.transition_style)])[0];
  const n = state.clips.length;
  const thumbs = state.clips.slice(0, 4).map(c =>
    `<i style="background-image:url('/api/jobs/${state.jobId}/poster/${encodeURIComponent(c.name)}')"></i>`).join('');
  host.innerHTML = `
    ${n ? `<div class="sumclips"><span class="thumbs">${thumbs}</span>
      <div><b>${n} clip${n === 1 ? '' : 's'}</b><small>ready to edit</small></div></div>` : ''}
    <div class="sumlist">
      <div><span>Format</span><b>${esc(fmt)}</b></div>
      <div><span>Resolution</span><b>${s.resolution === '4k' ? '4K' : esc(s.resolution)}</b></div>
      <div><span>Pace</span><b>${esc(speed)}</b></div>
      <div><span>Transitions</span><b>${esc(trans)}</b></div>
      <div><span>Music</span><b>${esc(String(music).slice(0, 22))}</b></div>
    </div>`;
}

async function saveScript() {
  if (!state.jobId && !$('#script').value.trim()) return;
  await ensureJob();
  try {
    const r = await api(`/api/jobs/${state.jobId}/script`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: $('#script').value }),
    });
    if (r.preview?.length) renderCaptionPreview(r.preview);
  } catch (e) { toast('Could not save the script: ' + e.message, 'bad'); }
}

function renderCaptionPreview(caps) {
  const box = $('#captionPreview');
  if (!caps?.length) { box.hidden = true; return; }
  box.hidden = false;
  box.innerHTML = `<small>Caption preview, scene by scene</small>`
    + caps.map((c, i) => `<div class="cp ${c ? '' : 'blank'}"><i>${i + 1}</i>
        <span>${c ? esc(c) : 'no caption'}</span></div>`).join('');
}

async function uploadScriptFile(e) {
  const f = e.target.files?.[0];
  if (!f || !state.jobId) return;
  const fd = new FormData();
  fd.append('file', f, f.name);
  try {
    const r = await api(`/api/jobs/${state.jobId}/script-file`, { method: 'POST', body: fd });
    $('#script').value = r.script;
    toast('Script loaded from ' + r.filename, 'ok');
    saveScript();
  } catch (err) { toast('Script upload failed: ' + err.message, 'bad'); }
  e.target.value = '';
}

/* voice note — recorded in the browser, stored and played back by the API.
   No transcription is claimed: /api/health reports whether an engine exists,
   and the notice above the recorder states plainly that none does. */
let rec = null, chunks = [], recBusy = false, recT0 = 0, lastRecSecs = 0;

/* The first container the browser can actually record into. Chrome and Edge
   do webm/opus, Firefox ogg or webm, Safari only mp4/aac. Hard-coding webm is
   what made recording work on one browser and silently fail on another. */
function recFormat() {
  const MR = window.MediaRecorder;
  const opts = [['audio/webm;codecs=opus', 'webm'], ['audio/webm', 'webm'],
                ['audio/ogg;codecs=opus', 'ogg'], ['audio/mp4', 'm4a']];
  for (const [mime, ext] of opts) if (MR?.isTypeSupported?.(mime)) return { mime, ext };
  return { mime: '', ext: 'webm' };
}

function setRecUI(mode, text = '') {
  const b = $('#recordBtn');
  b.classList.toggle('recording', mode === 'rec');
  b.disabled = mode === 'busy';
  b.innerHTML = mode === 'rec'
    ? '<span class="rec-dot"></span> Stop recording'
    : '<span class="rec-dot"></span> Record voice note';
  $('#recordState').textContent = text;
}

async function toggleRecord() {
  if (rec && rec.state === 'recording') { rec.stop(); return; }
  if (recBusy) return;                       // a double-click, or still uploading
  if (!window.isSecureContext) {
    return toast('The microphone only works on https:// or on 127.0.0.1 — open the app from one of those.', 'bad');
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    return toast('This browser cannot record audio. Try Chrome or Edge.', 'bad');
  }
  recBusy = true;
  setRecUI('busy', 'Starting microphone…');
  let stream;
  try {
    await ensureJob();
    if (!state.jobId) throw Object.assign(new Error('no job'), { name: 'NoJob' });
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    recBusy = false;
    setRecUI('idle');
    const why = e.name === 'NotAllowedError' ? 'Microphone permission was refused — allow it in the address bar and try again.'
      : e.name === 'NotFoundError' ? 'No microphone was found on this device.'
        : e.name === 'NotReadableError' ? 'The microphone is being used by another app.'
          : 'Could not start recording: ' + e.message;
    return toast(why, 'bad');
  }

  const fmt = recFormat();
  chunks = [];
  try { rec = fmt.mime ? new MediaRecorder(stream, { mimeType: fmt.mime }) : new MediaRecorder(stream); }
  catch { rec = new MediaRecorder(stream); }
  rec.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
  rec.onerror = e => toast('Recording failed: ' + (e.error?.message || 'unknown error'), 'bad');
  rec.onstop = async () => {
    stream.getTracks().forEach(t => t.stop());
    const secs = (performance.now() - recT0) / 1000;
    const type = rec.mimeType || fmt.mime || 'audio/webm';
    const blob = new Blob(chunks, { type });
    if (secs < 0.8 || blob.size < 1500) {
      recBusy = false; setRecUI('idle');
      return toast('That recording was too short — hold it for at least a second.', 'bad');
    }
    setRecUI('busy', 'Saving voice note…');
    const ext = /mp4|aac/.test(type) ? 'm4a' : /ogg/.test(type) ? 'ogg' : 'webm';
    const fd = new FormData();
    fd.append('file', blob, `voice_${Date.now()}.${ext}`);
    try {
      lastRecSecs = secs;
      const v = await api(`/api/jobs/${state.jobId}/voice`, { method: 'POST', body: fd });
      renderVoice(v, secs);
      toast('Voice note saved', 'ok');
    } catch (e) {
      toast('Voice upload failed: ' + e.message, 'bad');
    } finally {
      recBusy = false; setRecUI('idle');
    }
  };
  rec.start(250);                            // chunks every 250 ms, so a stop never loses it all
  recT0 = performance.now();
  recBusy = false;
  setRecUI('rec', '0.0s');
  const tick = setInterval(() => {
    if (!rec || rec.state !== 'recording') return clearInterval(tick);
    $('#recordState').textContent = ((performance.now() - recT0) / 1000).toFixed(1) + 's';
  }, 100);
}

/* A small custom player instead of the browser's <audio controls>, which
   restyles itself on hover and hid the label next to it. The src carries a
   version so a new recording is never served from the old one's cache. */
function renderVoice(v, knownSecs) {
  const box = $('#voiceBox');
  box.hidden = false;
  const src = `/api/jobs/${state.jobId}/voice-file?v=${encodeURIComponent(v.filename || '')}-${v.bytes || 0}`;
  const fallback = v.duration || knownSecs || (v.filename === state.job?.voice?.filename ? lastRecSecs : 0);
  box.innerHTML = `
    <div class="vplayer">
      <button type="button" class="vplay" aria-label="Play voice note">
        <svg class="i-play"><use href="#i-play"/></svg>
        <svg class="i-pause" viewBox="0 0 24 24"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z" fill="currentColor"/></svg>
      </button>
      <div class="vmain">
        <div class="vhead"><strong>Voice note</strong><span class="vtime">0:00 / ${fmtClock(fallback)}</span></div>
        <div class="vbar"><i></i></div>
      </div>
      <audio preload="metadata" src="${src}"></audio>
    </div>
    ${v.transcript ? `
      <div class="transcript">
        <div class="muted small" style="margin-bottom:4px"><b>Transcript</b></div>
        <div>${esc(v.transcript)}</div>
        <button type="button" class="btn btn-ghost btn-sm mt" id="useTranscriptBtn">Insert into captions</button>
      </div>` : ''}`;

  const au = $('audio', box), btn = $('.vplay', box), fill = $('.vbar i', box),
        time = $('.vtime', box), bar = $('.vbar', box);
  const dur = () => (Number.isFinite(au.duration) && au.duration > 0) ? au.duration : fallback;
  const paint = () => {
    const d = dur();
    fill.style.width = d ? Math.min(100, 100 * au.currentTime / d) + '%' : '0';
    time.textContent = `${fmtClock(au.currentTime)} / ${fmtClock(d)}`;
    btn.classList.toggle('on', !au.paused);
  };
  btn.onclick = () => {
    if (au.paused) au.play().catch(() => toast('Could not play the voice note.', 'bad'));
    else au.pause();
  };
  bar.onclick = e => {
    const d = dur(); if (!d) return;
    const r = bar.getBoundingClientRect();
    au.currentTime = d * Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  };
  ['timeupdate', 'play', 'pause', 'loadedmetadata', 'durationchange'].forEach(ev => au.addEventListener(ev, paint));
  au.addEventListener('ended', () => { au.currentTime = 0; paint(); });
  au.addEventListener('error', () => { time.textContent = 'could not load'; });

  if (v.transcript) {
    const tb = $('#useTranscriptBtn');
    if (tb) tb.onclick = () => {
      const sc = $('#script');
      sc.value = sc.value.trim() ? sc.value.trim() + '\n' + v.transcript : v.transcript;
      saveScript();
      toast('Transcript added to captions', 'ok');
    };
  }
}

const fmtClock = s => {
  if (!s || !Number.isFinite(s)) return '0:00';
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
};

async function loadTracks() {
  try { state.tracks = (await api('/api/music')).tracks || []; }
  catch { state.tracks = []; }
  renderTracks();
}

function renderTracks() {
  const host = $('#trackList');
  if (!state.tracks.length) {
    host.innerHTML = `<div class="empty">No tracks in the library yet — upload one
      to use it as the bed.</div>`;
    return;
  }
  host.innerHTML = state.tracks.map(t => `
    <div class="track ${state.settings.music_track === t.name ? 'on' : ''}"
         data-name="${esc(t.name)}">
      <span>${esc(t.name)}</span>
      ${t.mood ? `<span class="fact">${esc(t.mood)}</span>` : '<span></span>'}
      <span class="dur">${fmtDur(t.duration)}</span>
    </div>`).join('');
  $$('.track', host).forEach(el => el.onclick = () => {
    state.settings.music_track = el.dataset.name;
    renderTracks();
    summarise();
  });
}

async function uploadTrack(e) {
  const f = e.target.files?.[0];
  if (!f) return;
  const fd = new FormData();
  fd.append('file', f, f.name);
  fd.append('mood', state.settings.mood || '');
  try {
    const t = await api('/api/music', { method: 'POST', body: fd });
    toast(`Added ${t.name} (${fmtDur(t.duration)})`, 'ok');
    state.settings.music_track = t.name;
    await loadTracks();
    summarise();
  } catch (err) { toast('Track upload failed: ' + err.message, 'bad'); }
  e.target.value = '';
}

/* ─────────────────────────── step 3: processing ───────────────────────── */

function resetRun() {
  $('#stageRun').innerHTML = (state.health?.stages || []).map((s, i) => `
    <li class="agentgroup pending"><div class="agenthead">
      <span class="dot"></span>
      <div class="ainfo"><b>${esc(stageLabel(s))}</b><span class="adoes">${esc(stageDetail(s))}</span></div>
      <span class="t"></span>
    </div></li>`).join('');
  setRing(0, 'Queued', '');
  $('#procSub').textContent = 'Getting your clips ready…';
  $('#logCount').textContent = 'Raw pipeline output';
}

function setRing(pct, label, cls) {
  const c = 2 * Math.PI * 52;
  $('#ringFg').style.strokeDasharray = c;
  $('#ringFg').style.strokeDashoffset = c * (1 - Math.max(0, Math.min(100, pct)) / 100);
  $('#ringPct').textContent = Math.round(pct) + '%';
  $('#ringState').textContent = label;
  $('#ring').className = 'ring ' + (cls || '');
}

async function startBuild() {
  if (!state.jobId) { goto(1); return toast('Upload your clips first.', 'bad'); }
  const task = loader.open('Sending your clips to the AI', 'Checking your clips…', { delay: 0 });
  // Make sure the server still has this job and its clips before anything else
  try {
    const on = (await api(`/api/jobs/${state.jobId}/clips`)).clips || [];
    if (!on.length) {
      loader.fail();
      state.clips = []; renderClips(); goto(1);
      return toast('The server has no clips for this video — upload them again.', 'bad', true);
    }
  } catch (e) {
    if (!e.jobLost) { loader.fail(); toast('Could not start: ' + e.message, 'bad'); }
    return;
  }
  task.step('Sending your clips to the AI', 'Saving captions and settings…');
  await saveScript();
  if (!state.jobId) return;          // the session was lost while saving
  state.settings.target_seconds = $('#target').value || null;
  state.settings.title = $('#title').value.trim() || undefined;
  $('#startBuild').disabled = true;
  try {
    state.job = await api(`/api/jobs/${state.jobId}/start`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state.settings),
    });
    $('#log').textContent = '';
    delete $('#log').dataset.err;
    $('#toStep4').disabled = true;
    $('#procTitle').textContent = 'AI processing';
    resetRun();
    renderRun(state.job);
    goto(3);
    listen(state.jobId);
    task.done('AI started');
  } catch (e) {
    loader.fail();
    toast('Could not start: ' + e.message, 'bad');
    $('#startBuild').disabled = !state.jobId;
  }
}

async function cancelBuild() {
  try { await api(`/api/jobs/${state.jobId}/cancel`, { method: 'POST' }); }
  catch (e) { toast(e.message, 'bad'); }
}

/* runs once per build, from whichever of the event stream or the poll sees
   the end first */
const finished = {};
function finishRun(job) {
  if (finished[job.id]) return;
  finished[job.id] = true;
  state.es?.close();
  clearInterval(procTick);
  $('#startBuild').disabled = !state.clips.length;
  if (job.status === 'done') {
    toast('Your video is ready', 'ok');
    renderOutput(job);
    $('#toStep4').disabled = false;
    state.reached = 4;
    goto(4);
  } else if (job.status === 'error') {
    toast('Build failed: ' + (job.error || 'unknown'), 'bad');
    $('#logDetails').open = true;
  }
  loadBuilds();
}

function listen(jobId) {
  delete finished[jobId];
  state.es?.close();
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  state.es = es;
  es.onmessage = ev => {
    let m;
    try { m = JSON.parse(ev.data); } catch { return; }
    if (m.kind === 'snapshot' || m.kind === 'final') {
      state.job = m.job;
      renderRun(m.job);
      if (m.kind === 'final') finishRun(m.job);
      return;
    }
    if (m.kind === 'line') appendLog(m);
    if (m.kind === 'stage' || m.kind === 'status') refreshJob();
  };
  es.onerror = () => { /* EventSource retries on its own */ };
}

async function refreshJob() {
  try {
    const was = state.job?.status;
    state.job = await api(`/api/jobs/${state.jobId}`);
    renderRun(state.job);
    if (was === 'running' && state.job.status !== 'running') finishRun(state.job);
  } catch { /* transient */ }
}

let logLines = 0;
/* Full Windows/Unix paths make every line wrap. Show them relative to the
   project instead: uploads/<id>/..., outputs/<id>/..., pipeline/... */
function tidyLog(t) {
  return t
    .replace(/(?:[A-Za-z]:)?[\\/](?:[^\s\\/]+[\\/])*?(uploads|outputs|pipeline|data)[\\/]/g, '$1/')
    .replace(/\\/g, '/')
    .replace(/(uploads|outputs)\/[0-9a-f]{12}(?=\/|\s|$)/g, '$1/…');
}
function appendLog(m) {
  const pre = $('#log');
  const line = document.createElement('span');
  const t = tidyLog(m.text || '');
  line.className = t.startsWith('$') ? 'cmd'
    : /fail|error|refus|missing|could not/i.test(t) ? 'bad'
      : /warn|note:/i.test(t) ? 'warn' : '';
  line.textContent = `${String(m.t).padStart(6)}  ${t}\n`;
  pre.append(line);
  logLines++;
  $('#logCount').textContent = `${logLines} line${logLines === 1 ? '' : 's'}`;
  if ($('#autoscroll').checked && $('#logDetails').open) pre.scrollTop = pre.scrollHeight;
}

/* A clip the planner left out must never disappear quietly - say which, and
   say why, in the planner's own words. */
function renderExcluded(job) {
  const box = $('#excluded');
  if (!box) return;
  const out = job.analysis?.rejected || [];
  box.hidden = !out.length;
  if (!out.length) return;
  box.innerHTML = `<span class="notice-ico">!</span><div>
    <strong>${out.length} clip${out.length === 1 ? ' was' : 's were'} left out of the video</strong>
    ${out.map(r => `<span style="display:block">${esc(clipName(r.clip))} — ${esc(r.why)}</span>`).join('')}
    </div>`;
}

/* The edit plan: what was decided, by whom, and why. `decided_by` is either
   "measurement" (the director worked it out) or "you" (your setting won). */
function renderPlan(job) {
  const card = $('#planCard'), host = $('#planList');
  const plan = job.edit_plan;
  if (!card || !host) return;
  card.hidden = !plan || !plan.decisions?.length;
  if (card.hidden) return;
  const names = Object.fromEntries((state.health?.agents || [])
    .map(a => [a.id, a.name.replace(/ Agent$/, '')]));
  host.innerHTML = plan.decisions.map(d => {
    const by = d.decided_by;
    const cls = by === 'vision' ? 'vision' : by === 'measurement' ? 'auto' : 'manual';
    const label = by === 'vision' ? 'saw it'
                : by === 'measurement' ? 'decided' : 'yours';
    const who = names[d.agent] || '';
    return `<div class="plandec ${cls}">
      <div class="planhead">
        ${who ? `<span class="planagent">${esc(who)}</span>` : ''}
        <span class="planby">${label}</span>
        <b>${esc(d.what)}</b>
        <span class="planchose">${esc(d.chose ?? '—')}</span>
      </div>
      <p class="planwhy">${esc(d.why || '')}</p>
    </div>`;
  }).join('');
}

/* ─────────────────────── progress & time estimate ──────────────────────
   The server reports progress in fixed stage weights, and a running stage
   counts as 45% of its weight until it ends. Motion analysis alone can take
   most of a build on long 60 fps footage, so the ring used to sit on one
   number for minutes. Instead the time each stage should take is predicted
   from the footage and the planned length, corrected by how fast the stages
   that already finished actually ran on this machine, and the ring follows
   the clock between server updates. It never moves backwards. */

const serverNow = () => Date.now() / 1000 + (state.skew || 0);

function stageExpect(job) {
  const a = job.analysis || {};
  const F = a.total_duration
    || (a.clips || []).reduce((t, c) => t + (c.duration || 0), 0)
    || state.clips.reduce((t, c) => t + (c.duration || 0), 0) || 60;
  const tgt = a.target || +(job.settings?.target_seconds) || 60;
  const O = a.planned_total || Math.min(tgt, Math.max(15, F * 0.06));
  const sp = { slow: 0.75, fast: 1.35 }[job.settings?.speed] || 1;
  return {
    probe: 2 + 0.03 * F, measure: 3 + 0.55 * F, vision: 20, plan: 1, direct: 0.5,
    sheets: 3 + 0.3 * O, cut: 3 + 2.2 * O, join: 2 + 1.0 * O,
    music: 3 + 1.0 * O / sp, master: 3 + 0.65 * O / sp,
  };
}

const peak = {};
function estimate(job) {
  if (!job) return { pct: 0, left: null, elapsed: 0 };
  if (job.status === 'done') return { pct: 100, left: 0, elapsed: (job.finished - job.started) || 0 };
  const exp = stageExpect(job), now = serverNow();
  let actual = 0, predicted = 0;              // for the speed correction
  for (const st of job.stages || []) {
    const e = exp[st.id] ?? 5;
    if (st.status === 'done' && st.started && st.finished && e >= 3) {
      actual += st.finished - st.started; predicted += e;
    }
  }
  const k = predicted > 5 ? Math.min(3, Math.max(0.3, actual / predicted)) : 1;
  let total = 0, got = 0, left = 0;
  for (const st of job.stages || []) {
    if (st.status === 'skipped') continue;
    const done = st.status === 'done';
    const e = done && st.started && st.finished ? (st.finished - st.started) : (exp[st.id] ?? 5) * k;
    total += e;
    if (done) { got += e; continue; }
    if (st.status === 'running') {
      const ran = Math.max(0, now - (st.started || now));
      const f = ran < e ? 0.92 * ran / e : 0.92 + 0.07 * (1 - Math.exp(-(ran - e) / e));
      got += e * f;
      left += Math.max(e - ran, e * 0.08);
    } else left += e;
  }
  let pct = job.status === 'running' ? Math.min(99, 100 * got / (total || 1)) : (job.progress || 0);
  if (job.status === 'running') {
    pct = Math.max(pct, job.progress || 0, peak[job.id] || 0);
    peak[job.id] = pct;
  }
  const elapsed = job.started ? Math.max(0, now - job.started) : 0;
  return { pct, left: job.status === 'running' ? left : null, elapsed };
}

let procTick = null;
function renderProcTime(job) {
  const box = $('#procTime');
  clearInterval(procTick);
  let n = 0;
  const paint = () => {
    const j = state.job && state.job.id === job.id ? state.job : job;
    const est = estimate(j);
    if (j.status === 'running') {
      const cur = (j.stages || []).find(x => x.status === 'running');
      setRing(est.pct, cur ? stageLabel(cur) : 'Working', 'live');
    }
    if (!box) return;
    if (j.status === 'running') {
      box.innerHTML = `<span>Elapsed <b>${loader.fmt(est.elapsed * 1000)}</b></span>`
        + `<span>About <b>${loader.fmt(est.left * 1000)}</b> left</span>`;
      box.hidden = false;
    } else if (j.status === 'done' && j.started && j.finished) {
      box.innerHTML = `<span>Finished in <b>${loader.fmt((j.finished - j.started) * 1000)}</b></span>`;
      box.hidden = false;
    } else box.hidden = true;
  };
  paint();
  if (job.status === 'running') procTick = setInterval(() => {
    paint();
    // the event stream can be buffered by proxies and tunnels; poll as well
    if (++n % 4 === 0 && state.jobId === job.id) refreshJob();
  }, 1000);
}

function renderRun(job) {
  const a = job.analysis || {};
  const cur = (job.stages || []).find(x => x.status === 'running');
  const label = job.status === 'running' ? (cur ? stageLabel(cur) : (job.current || 'Working'))
    : job.status === 'done' ? 'Complete'
      : job.status === 'draft' ? 'Not started' : title(job.status);
  setRing(estimate(job).pct, label,
    job.status === 'done' ? 'done' : job.status === 'error' ? 'bad'
      : job.status === 'running' ? 'live' : '');

  const bits = [];
  if (a.clips?.length) bits.push(`Analysing ${a.clips.length} clip${a.clips.length === 1 ? '' : 's'}`);
  if (a.segments?.length) bits.push(`${a.segments.length} scenes planned`);
  if (a.planned_total) bits.push(`${fmtDur(+a.planned_total)} video`);
  if (a.orientation) bits.push(ORIENT_LABEL[a.orientation] || a.orientation);
  let sub = bits.join(' · ');
  if (job.status === 'running' && !sub) sub = 'The AI is reviewing your footage…';
  if (job.status === 'done') sub = 'Your video has been edited, scored and checked.';
  if (job.status === 'error') sub = 'Something went wrong: ' + (job.error || 'unknown error')
    + '. The technical log below has the details.';
  if (job.status === 'cancelled') sub = 'This build was cancelled.';
  $('#procSub').textContent = sub || 'Getting your clips ready…';
  $('#cancelBuild').hidden = job.status !== 'running';
  $('#procTitle').textContent = job.status === 'done' ? 'Your video is ready'
    : job.status === 'error' ? 'Processing failed'
      : job.status === 'cancelled' ? 'Processing cancelled'
        : job.status === 'draft' ? 'Ready when you are' : 'AI is creating your video';

  // Stages are grouped under the agent that owns them, so the run reads as
  // five agents doing their jobs rather than ten anonymous steps.
  const agents = state.health?.agents || [];
  const byId = Object.fromEntries(job.stages.map((s, i) => [s.id, { s, i }]));
  // whether a model ran is a fact about THIS build, not about the server as it
  // is configured right now - an old build keeps its badge when the key goes
  const ranModel = job.status === 'draft'
    ? !!state.health?.vision?.enabled
    : !!job.vision;
  $('#stageRun').innerHTML = agents.map(a => {
    const mine = a.stages.map(id => byId[id]).filter(Boolean);
    if (!mine.length) return '';
    const fin = x => x.s.status === 'done' || x.s.status === 'skipped';
    const st = mine.some(x => x.s.status === 'error') ? 'error'
      : mine.some(x => x.s.status === 'running') ? 'running'
        : mine.every(x => x.s.status === 'skipped') ? 'skipped'
          : mine.every(fin) ? 'done' : 'pending';
    const secs = mine.reduce((t, x) =>
      t + ((x.s.started && x.s.finished) ? x.s.finished - x.s.started : 0), 0);
    return `<li class="agentgroup ${st}">
      <div class="agenthead">
        <span class="dot"></span>
        <div class="ainfo">
          <b>${esc(a.name)}${a.uses_model
            ? (ranModel ? ' <span class="amodel">AI vision</span>'
                        : ' <span class="amodel off">vision off</span>')
            : ''}</b>
          <span class="adoes">${esc(a.does)}</span>
        </div>
        <span class="t">${secs ? secs.toFixed(1) + 's' : (st === 'running' ? 'working…' : st === 'skipped' ? 'skipped' : '')}</span>
      </div>
      <ol class="agentstages">${mine.map(({ s, i }) => {
        const t = (s.started && s.finished) ? ((s.finished - s.started).toFixed(1) + 's')
          : s.status === 'running' ? 'running…' : '';
        const note = s.status === 'skipped' ? 'Not needed for this video'
          : s.status === 'done' ? 'Done' : stageDetail(s);
        return `<li class="${s.status}">
          <span class="dot"></span>
          <div><b>${esc(stageLabel(s))}</b><small>${esc(note)}</small></div>
          <span class="t">${t}</span></li>`;
      }).join('')}</ol>
    </li>`;
  }).join('');

  renderProcTime(job);
  renderExcluded(job);
  renderPlan(job);
  if (job.captions?.length) renderCaptionPreview(job.captions);
  if (job.output) { $('#toStep4').disabled = false; state.reached = 4; }
  if (job.status === 'error' && job.error) {
    const pre = $('#log');
    if (!pre.dataset.err) {
      pre.dataset.err = '1';
      appendLog({ t: '', text: 'BUILD FAILED: ' + job.error });
    }
  }
  renderStepper();
}

/* ───────────────────────────── step 4: output ─────────────────────────── */

async function openJob(id) {
  const task = loader.open('Opening your video');
  try {
    const job = await api(`/api/jobs/${id}`);
    setHidden('#sessionNotice', true);
    state.es?.close();
    state.jobId = id;
    state.job = job;
    state.settings = { ...state.settings, ...(job.settings || {}) };
    hydrateSettings();
    await refreshClips();
    $('#script').value = job.script || '';
    $('#voiceBox').hidden = !job.voice;
    if (job.voice) renderVoice(job.voice);
    renderCaptionPreview(job.captions);
    $('#log').textContent = '';
    logLines = 0;
    delete $('#log').dataset.err;
    resetRun();
    renderRun(job);
    state.reached = job.output ? 4 : (job.status === 'running' ? 3 : 2);

    if (job.status === 'running') {
      goto(3);
      listen(id);
    } else if (job.output) {
      renderOutput(job);
      goto(4);
    } else {
      goto(3);
      toast(job.status === 'error'
        ? 'That video failed — details are in the technical log.'
        : 'That video has not been created yet.');
      if (job.status === 'error') $('#logDetails').open = true;
    }
    task.done('Opened');
  } catch (e) { loader.fail(); toast('Could not open that video: ' + e.message, 'bad'); }
}

function hydrateSettings() {
  const s = state.settings;
  setPicker('#orientation', s.orientation);
  setPicker('#resolution', s.resolution);
  setPicker('#quality', s.quality);
  setPicker('#transition', s.transition_style);
  setPicker('#xfade', s.transition_duration == null ? 'auto' : s.transition_duration);
  setPicker('#speed', s.speed);
  setPicker('#musicSource', s.music_source);
  setPicker('#mood', s.mood);
  setPicker('#keepAudio', s.keep_audio || 'none');
  setPicker('#sequence', s.sequence || 'hook');
  $('#speedCustom').hidden = s.speed !== 'custom';
  $('#moodWrap').hidden = s.music_source !== 'generate';
  $('#libraryWrap').hidden = s.music_source !== 'upload';
  if (s.speed_custom) {
    $('#speedRange').value = s.speed_custom;
    $('#speedOut').textContent = (+s.speed_custom).toFixed(2) + '×';
  }
  $('#target').value = s.target_seconds || '';
  $('#title').value = s.title || '';
  qualityHint(); moodHint(); keepAudioHint(); sequenceHint(); visionNotice();
  renderAgents();
  audioNotice(); timestampNotice();
  renderTracks(); summarise();
}

function renderOutput(job) {
  const o = job.output;
  $('#outputEmpty').hidden = !!o;
  $('#outputBody').hidden = !o;
  if (!o) return;

  const player = $('#player');
  player.src = `/api/jobs/${job.id}/file/${encodeURIComponent(o.file)}`;
  // the pipeline's own thumbnail, so the player shows a real frame before play
  if (o.thumbnail) player.poster = `/api/jobs/${job.id}/file/${encodeURIComponent(o.thumbnail)}`;
  else player.removeAttribute('poster');
  $('#playerShell').classList.toggle('vertical', o.orientation === 'vertical');
  $('#outName').textContent = prettyName(job.settings?.title || job.title || o.file);
  const nClips = job.analysis?.clips?.length;
  const nOut = (job.analysis?.rejected || []).length;
  const used = nClips ? nClips - nOut : null;
  const m = String(o.music || 'silent');
  const bedName = b => b === 'library' ? 'your music track'
    : `${(MOOD_LABEL[b] || title(b)).toLowerCase()} AI soundtrack`;
  const musicLine = m === 'silent' ? 'no music'
    : m === 'original audio' ? 'the original clip audio'
      : m.startsWith('original audio + ') ? `the original audio over a ${bedName(m.slice(17))}`
        : m === 'library' ? 'your music track'
          : `${/^[aeiou]/i.test(bedName(m)) ? 'an' : 'a'} ${bedName(m)}`;
  $('#outLine').textContent =
    `${o.segments} scene${o.segments === 1 ? '' : 's'}`
    + (used != null ? ` from ${used} clip${used === 1 ? '' : 's'}` : '')
    + `, cut together with ${musicLine}.`
    + (nOut ? ` ${nOut} clip${nOut === 1 ? ' was' : 's were'} left out — see AI Processing for why.` : '');
  $('#downloadBtn').href = `/api/jobs/${job.id}/download`;

  $('#outChips').innerHTML = [
    fmtDur(o.duration),
    ORIENT_LABEL[o.orientation] || o.orientation,
    o.height ? (o.height >= 2000 || o.width >= 3800 ? '4K' : Math.min(o.width, o.height) + 'p') : null,
    o.speed && o.speed !== 1 ? `${o.speed}× pace` : null,
    job.started && job.finished ? `Made in ${loader.fmt((job.finished - job.started) * 1000)}` : null,
  ].filter(Boolean).map(x => `<span>${esc(x)}</span>`).join('');

  renderTimeline(job);

  // A build the operator asked to be silent is judged against that, not
  // against a loudness range it was never meant to hit.
  const silent = o.music === 'silent';
  const checks = [
    ['Frame count', o.frames_ok, `${o.frames} of ${o.frames_expected} expected`],
    ['No black frames', !o.black_frames, o.black_frames ? 'detected' : 'clean'],
    silent
      ? ['Silent, as requested', o.lufs != null && o.lufs < -50,
        o.lufs == null ? 'not measured' : o.lufs + ' LUFS']
      : ['Loudness in range', o.lufs != null && o.lufs > -20 && o.lufs < -11,
        o.lufs == null ? 'not measured' : o.lufs + ' LUFS'],
    ['Audio track present', !!o.audio_codec, o.audio_codec || 'none'],
  ];
  const passed = checks.filter(c => c[1]).length;
  const qc = $('#verifySummary');
  qc.className = 'qc ' + (passed === checks.length ? 'ok' : 'warn');
  qc.innerHTML = passed === checks.length
    ? `<svg class="ico"><use href="#i-check"/></svg>All quality checks passed`
    : `${passed} of ${checks.length} quality checks passed`;

  $('#verify').innerHTML = checks.map(([label, ok, val]) => `
    <div class="vrow ${ok ? 'ok' : 'bad'}">
      <span class="ico">${ok ? '✓' : '!'}</span>
      <span>${label}</span><span class="val">${esc(val)}</span>
    </div>`).join('');

  $('#meta').innerHTML = [
    ['File', o.file], ['Size', fmtBytes(o.size)], ['Duration', o.duration + 's'],
    ['Resolution', `${o.width}×${o.height}`], ['Frame rate', o.fps],
    ['Video', o.video_codec],
    ['Audio', `${o.audio_codec || '—'} ${o.sample_rate ? o.sample_rate + ' Hz' : ''}`],
    ['Speed', o.speed + '×'], ['Orientation', o.orientation],
    ['Transitions', (o.transitions || []).map(t => t.toFixed(2) + 's').join(', ') || '—'],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${esc(v ?? '—')}</dd>`).join('');

  const segs = job.analysis?.segments || [];
  $('#cutsheet').innerHTML = segs.length ? `
    <thead><tr><th>#</th><th>Clip</th><th>In–out</th><th>Length</th>
      <th>Motion</th><th>Caption</th></tr></thead>
    <tbody>${segs.map(s => `<tr>
      <td>${s.index}</td><td>${esc(s.clip)}</td>
      <td>${s.start}–${s.end}</td><td>${s.duration}s</td>
      <td>${s.mean_ydif ?? '—'}</td>
      <td class="wrap">${esc(s.caption || '—')}</td></tr>`).join('')}</tbody>`
    : '<tbody><tr><td class="wrap">No scene list for this video.</td></tr></tbody>';
}

/* segment strip — proportional to the planner's own durations */
function renderTimeline(job) {
  const segs = job.analysis?.segments || [];
  const host = $('#timeline');
  if (!segs.length) { host.innerHTML = ''; return; }
  const total = segs.reduce((a, s) => a + (s.duration || 0), 0) || 1;
  const BRICKS = ['#EF3B3B', '#FFD23F', '#3B82F6', '#22C55E', '#FF8A1F', '#38BDF8', '#F472B6'];
  const hue = i => BRICKS[i % BRICKS.length];
  const marks = (job.output?.transitions || []).length;
  host.innerHTML = `
    <div class="tl-bar">${segs.map((s, i) => `
      <div class="tl-seg" style="flex:${(s.duration || 0) / total};background:${hue(i)}"
           title="Scene ${s.index} · ${esc(clipName(s.clip))} · ${s.duration}s">
        ${(s.duration || 0) / total > 0.07 ? s.index : ''}
      </div>`).join('')}</div>
    <div class="tl-legend">${segs.length} scenes · ${marks}
      transition${marks === 1 ? '' : 's'}, with the music timed to every cut.</div>`;
}

async function regenerate() {
  const task = loader.open('Regenerating your video', 'Sending your settings to the AI…');
  try {
    const r = await api(`/api/jobs/${state.jobId}/regenerate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...state.settings }),
    });
    state.jobId = r.id;
    state.job = r.job;
    $('#log').textContent = '';
    logLines = 0;
    delete $('#log').dataset.err;
    $('#toStep4').disabled = true;
    resetRun();
    renderRun(r.job);
    goto(3);
    listen(r.id);
    task.done('AI started');
  } catch (e) { loader.fail(); toast('Regenerate failed: ' + e.message, 'bad'); }
}

boot();
