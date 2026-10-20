/* Alpha Titans — the cloud studio. Vanilla ES module, no build.
 *
 * Built from the studio's own pieces (frontend/studio.js on main): the same
 * shell and sidebar, the same top bar, the same Create screen with its
 * pills and examples, the same five-phase production screen and the same
 * finished-video screen, styled by the same studio.css. What differs is
 * only where things happen: footage goes up to the control plane, a cloud
 * worker is woken to produce it, and the results come back as downloads.
 *
 * WHAT THE PROGRESS BAR MEANS
 *   Bytes the SERVER confirmed it holds, plus the chunk in flight as the
 *   browser reports sending it. When the server acknowledges less than was
 *   sent, the bar steps back to the server's figure.
 *
 * WHAT THIS PAGE NEVER DOES
 *   * put the access token in a URL - it lives in this tab's sessionStorage
 *     and travels in a header. Downloads use one-time links instead;
 *   * pull a large file through fetch() into memory - the browser's own
 *     download manager fetches it;
 *   * invent an ETA.
 */

const API = '/api/cloud';
const TOKEN_KEY = 'at.cloud.token';
const SALT_KEY = 'at.cloud.salt';
const JOBS_KEY = 'at.cloud.jobs';     // the videos THIS browser made
const TERMINAL = new Set(['completed', 'failed', 'cancelled']);

/* ───────────────────────── studio helpers ─────────────────────────── */

const el = (t, c, h) => { const n = document.createElement(t);
  if (c) n.className = c; if (h != null) n.innerHTML = h; return n; };

const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const svg = (d, cls = '') => {
  if (!d) return '';
  const p = d.split('|').map(x => `<path d="${x}" fill="none" stroke="currentColor"
    stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>`).join('');
  return `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true">${p}</svg>`;
};

const ICON = {   // the studio's own paths
  nav_create: 'M12 5v14M5 12h14',
  nav_prod: 'M4 5h16v14H4zM9 9l6 3-6 3z',
  nav_assets: 'M4 7l8-3 8 3-8 3zM4 12l8 3 8-3M4 17l8 3 8-3',
  upload: 'M12 16V4m0 0-4.5 4.5M12 4l4.5 4.5M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3',
  spark: 'M12 3l1.7 5L19 10l-5.3 2L12 17l-1.7-5L5 10l5.3-2z',
  trash: 'M4 7h16M9 7V5h6v2M7 7l1 13h8l1-13',
  burger: 'M4 7h16M4 12h16M4 17h16',
  play: 'M8 5.5v13l10.5-6.5z',
  down: 'M12 4v11m0 0-4.5-4.5M12 15l4.5-4.5M5 20h14',
};

const NA = why => `<span class="sx-na">Unavailable${why ? ` · ${esc(why)}` : ''}</span>`;

/* ───────────────────────────── labels ─────────────────────────────── */
// Identical to the studio's Create screen.

const DURATIONS = [['Auto', null], ['15s', 15], ['30s', 30], ['60s', 60]];
const STYLE_LABEL = { auto: 'Let AI decide', cinematic: 'Cinematic',
  fast: 'Fast-paced', minimal: 'Minimal', social: 'Social', story: 'Story-driven' };
const PACE_LABEL = { auto: 'Let AI decide', relaxed: 'Relaxed',
  balanced: 'Balanced', fast: 'Fast' };
const MUSIC_LABEL = { auto: 'Let AI choose', none: 'No music', lofi: 'Lo-fi',
  calm: 'Calm', bright: 'Bright', upbeat: 'Upbeat', beat: 'Beat-driven',
  energetic: 'Energetic', cinematic: 'Cinematic' };
const CAP_LABEL = { auto: 'Auto', off: 'Off', key_moments: 'Key moments' };
const VO_LABEL = { none: 'None', text: 'Write narration text' };
const FRAMING_LABEL = { fit: 'Show the whole clip (blurred sides)',
  fill: 'Crop to fill the frame' };
const ORIENT_LABEL = { horizontal: 'Horizontal · YouTube',
  vertical: 'Vertical · Shorts, Reels', auto: 'Auto' };
const EXAMPLES = [
  'Make a fast-paced cinematic reel with energetic music.',
  'A 30-second highlight of the most interesting moments, with captions.',
  'Keep the mechanical detail visible and use a calm score.',
];

// The studio's five phases, mapped onto the real graph nodes, with the
// cloud's own steps before and after them.
const UX_PHASES = [
  ['Analyzing your footage', ['clip_analysis', 'vision', 'research']],
  ['Understanding your request', ['content_strategy']],
  ['Building the story', ['story', 'voiceover', 'music', 'animation',
                          'transition', 'caption']],
  ['Editing your video', ['editing_director', 'render']],
  ['Final quality check', ['critic']],
];
const WORKER_WORD = { stopped: 'Worker asleep', starting: 'Worker starting',
  ready: 'Worker awake', stopping: 'Worker switching off',
  start_failed: 'Worker unavailable' };
const KIND_LABEL = { video: 'Video', thumbnail: 'Thumbnail',
  youtube: 'YouTube upload sheet', report: 'Technical report',
  plan: 'Edit plan', log: 'Production log' };

/* ───────────────────────────── state ──────────────────────────────── */

const S = { cfg: null, token: '', view: 'create', jobId: null, job: null,
  trace: null, jobs: [], worker: null, options: null, error: null,
  timer: null, sideOpen: false, watched: null, media: {} };
const C = { files: [], instruction: '', advanced: false, note: null,
  creating: false, sent: {}, upTimer: null,
  // the background upload: it starts as soon as clips are added
  up: { run: 0, state: 'idle', jobId: null, error: null, xhr: null },
  prefs: { style: 'auto', pacing: 'auto', music: 'auto', captions: 'auto',
           voiceover: 'none', resolution: '1080p', quality: 'high',
           target_seconds: null, orientation: 'horizontal', framing: 'fit' } };   // Auto, as on main

let root = null, sideEl = null;

function getItem(kind, key) {
  try { return (kind === 'session' ? sessionStorage : localStorage).getItem(key) || ''; }
  catch { return ''; }
}
function setItem(kind, key, v) {
  try { (kind === 'session' ? sessionStorage : localStorage).setItem(key, v); }
  catch { /* private mode: the page still works, it just forgets */ }
}

// With no sign-in, a browser sees the videos it made: their ids are kept
// here, and a job id is the key to that one job and nothing else.
function myJobs() {
  try {
    return JSON.parse(getItem('local', JOBS_KEY) || '[]')
      .filter(x => /^[a-f0-9]{12}$/.test(x));
  } catch { return []; }
}
function rememberJob(id) {
  const ids = [id, ...myJobs().filter(x => x !== id)].slice(0, 50);
  setItem('local', JOBS_KEY, JSON.stringify(ids));
}

/* ────────────────────────────── api ───────────────────────────────── */

class ApiError extends Error {
  constructor(status, body) {
    super((body && body.message) || `HTTP ${status}`);
    this.status = status; this.body = body || {};
  }
}

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (S.token) headers.Authorization = `Bearer ${S.token}`;
  if (opts.json !== undefined) {
    headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(opts.json);
  }
  const r = await fetch(path, { method: opts.method || 'GET', headers,
                                body: opts.body, cache: 'no-store' });
  let body = null;
  try { body = await r.json(); } catch { body = null; }
  if (!r.ok) throw new ApiError(r.status, body);
  return body;
}

/* ──────────────────────────── chrome ──────────────────────────────── */

const NAV = [
  { group: 'Create', items: [
    ['create', 'Create video', 'nav_create'],
    ['projects', 'My videos', 'nav_assets'],
  ] },
  { group: 'Production', items: [
    ['job', 'Current production', 'nav_prod'],
  ] },
];

function buildSidebar() {
  sideEl.textContent = '';
  const brand = el('button', 'sx-brand');
  brand.innerHTML = `<svg class="mark" aria-hidden="true"><use href="#i-brick"/></svg>
    <span class="brand-text"><b>Alpha Titans</b><small>Cloud</small></span>`;
  brand.addEventListener('click', () => go('create'));
  sideEl.append(brand);

  NAV.forEach(g => {
    const items = g.items.filter(([id]) => id !== 'job' || S.jobId);
    if (!items.length) return;
    const wrap = el('div', 'sx-navgroup');
    wrap.append(el('div', 'sx-navlabel', esc(g.group)));
    items.forEach(([id, label, ico]) => {
      const b = el('button', 'sx-nav' + (S.view === id ? ' is-on' : ''));
      const tag = id === 'projects' ? (S.jobs.length || '') : '';
      b.innerHTML = `${svg(ICON[ico])}<span>${esc(label)}</span>
        ${tag ? `<span class="sx-nav-tag">${tag}</span>` : ''}`;
      b.addEventListener('click', () => { closeSide(); go(id); });
      wrap.append(b);
    });
    sideEl.append(wrap);
  });

  const foot = el('div', 'sx-side-foot');
  const w = S.worker;
  const dot = !w ? '' : w === 'ready' || w === 'starting' ? 'is-ok'
    : w === 'start_failed' ? 'is-warn' : '';
  foot.innerHTML = `<div class="sx-sysline">
    <i class="sx-sysdot ${dot}"></i>
    <span>${w ? esc(WORKER_WORD[w] || w) : 'Connecting…'}</span>
  </div>`;
  sideEl.append(foot);
}

function closeSide() {
  S.sideOpen = false;
  sideEl.classList.remove('is-open');
  document.querySelector('.sx-scrim')?.remove();
}
function toggleSide() {
  S.sideOpen = !S.sideOpen;
  sideEl.classList.toggle('is-open', S.sideOpen);
  document.querySelector('.sx-scrim')?.remove();
  if (S.sideOpen) {
    const sc = el('div', 'sx-scrim');
    sc.addEventListener('click', closeSide);
    document.body.append(sc);
  }
}

function topBar(title, sub, right) {
  const t = el('header', 'sx-top');
  const burger = el('button', 'sx-burger');
  burger.setAttribute('aria-label', 'Menu');
  burger.innerHTML = svg(ICON.burger);
  burger.addEventListener('click', toggleSide);
  t.append(burger);
  t.append(el('div', null,
    `<h1>${esc(title)}</h1>${sub ? `<p class="sx-top-sub">${sub}</p>` : ''}`));
  if (right) { const r = el('div', 'sx-top-right'); r.append(right); t.append(r); }
  return t;
}

function pickRow(options, current, onPick, labels) {
  const row = el('div', 'cx-pills');
  options.forEach(o => {
    const val = Array.isArray(o) ? o[1] : o;
    const txt = Array.isArray(o) ? o[0] : (labels ? labels[o] || o : o);
    const b = el('button', 'cx-pill' + (current === val ? ' is-on' : ''),
                 esc(String(txt)));
    b.type = 'button';
    b.disabled = C.creating;
    b.addEventListener('click', () => { onPick(val); render(); });
    row.append(b);
  });
  return row;
}

const mb = n => `${(Number(n || 0) / 1048576).toFixed(1)} MB`;
const size = n => { n = Number(n || 0);
  return n < 1024 ? `${n} B` : n < 1048576 ? `${Math.round(n / 1024)} KB` : mb(n); };
const when = t => (t ? new Date(t * 1000).toLocaleString() : '');

/* ═══════════════════════════ SIGN IN ═══════════════════════════════ */

function viewToken() {
  const f = document.createDocumentFragment();
  f.append(topBar('Sign in', 'This studio needs its access token'));
  const page = el('div', 'cx-page');
  const block = el('section', 'cx-block');
  block.append(el('h2', 'cx-h', 'Access token'));
  const box = el('div', 'cx-ask');             // the studio's framed field
  const input = el('input', 'cx-ta cl-token');
  input.type = 'password'; input.autocomplete = 'off'; input.spellcheck = false;
  input.placeholder = 'Paste the access token';
  input.setAttribute('aria-label', 'Access token');
  box.append(input);
  block.append(box);
  block.append(el('p', 'cx-note', 'It is kept only in this tab and sent in '
    + 'a request header, never in a link.'));
  if (S.error) block.append(el('p', 'cx-note is-warn', esc(S.error)));
  page.append(block);
  const go_ = el('section', 'cx-go');
  const btn = el('button', 'cx-cta', `${svg(ICON.spark)}<span>Continue</span>`);
  btn.type = 'button';
  btn.addEventListener('click', async () => {
    S.token = input.value.trim(); input.value = '';
    try {
      await api(`${API}/jobs?limit=1`);
      setItem('session', TOKEN_KEY, S.token);
      S.error = null;
      S.view = 'create';
      await boot();
    } catch {
      S.token = ''; S.error = 'That token was not accepted.'; render();
    }
  });
  input.addEventListener('keydown', e => { if (e.key === 'Enter') btn.click(); });
  go_.append(btn);
  page.append(go_);
  f.append(page);
  return f;
}

/* ═══════════════════════════ CREATE ════════════════════════════════ */
//
// The clips start uploading the moment they are added, so the person
// chooses the options while the bytes travel. "Create my video" unlocks at
// 100% and sends those choices with it.

const pct = (a, b) => (b ? Math.min(100, Math.floor(100 * a / b)) : 0);

function totals() {
  const total = C.files.reduce((a, x) => a + x.file.size, 0);
  const got = C.files.reduce((a, x, i) => a + Math.min(C.sent[i] || 0, x.file.size), 0);
  return { total, got, p: pct(got, total) };
}

function addFiles(list) {
  const all = Array.from(list || []);
  const ok = all.filter(f => /\.(mp4|mov|m4v|mkv|avi)$/i.test(f.name));
  C.note = ok.length < all.length
    ? `${all.length - ok.length} file(s) skipped: use MP4, MOV, M4V, MKV or AVI.`
    : null;
  const most = S.cfg?.max_files || 8;
  if (C.files.length + ok.length > most) {
    C.note = `At most ${most} clips per video.`;
    return render();
  }
  if (!ok.length) return render();
  C.files.push(...ok.map(f => {
    const url = URL.createObjectURL(f);
    // ONE preview per clip, made once and moved into each redraw: a new
    // <video> per redraw reloads the clip and flickers
    const thumb = el('video');
    thumb.muted = true; thumb.playsInline = true; thumb.preload = 'metadata';
    const x = { file: f, url, thumb, meta: null };
    // its shape and length, for the orientation hint
    thumb.addEventListener('loadedmetadata', () => {
      if (thumb.videoWidth && thumb.videoHeight) {
        x.meta = { vertical: thumb.videoHeight > thumb.videoWidth,
                   seconds: thumb.duration || 0 };
        if (S.view === 'create') render();
      }
    }, { once: true });
    thumb.src = `${url}#t=1`;
    return x;
  }));
  restartUpload();
}

function removeFile(i) {
  const [x] = C.files.splice(i, 1);
  if (x) URL.revokeObjectURL(x.url);
  restartUpload();
}

function uploadLine() {
  const t = totals();
  if (C.up.state === 'uploading') {
    return `Uploading your clips… ${t.p}% (${mb(t.got)} of ${mb(t.total)})`;
  }
  if (C.up.state === 'done') {
    return '✓ All clips uploaded. Choose your options, then press Create my video.';
  }
  if (C.up.state === 'error') return `Upload stopped: ${C.up.error}`;
  return '';
}

function clipLine(x, i) {
  const sent = Math.min(C.sent[i] || 0, x.file.size);
  if (C.up.state === 'done') return `✓ Uploaded · ${mb(x.file.size)}`;
  return `${pct(sent, x.file.size)}% · ${mb(sent)} of ${mb(x.file.size)}`;
}

function ctaState() {
  const t = totals();
  if (C.creating) return { label: 'Starting your video…', on: false,
    note: 'Sending your choices…' };
  if (!C.files.length) return { label: 'Create my video', on: false,
    note: 'Add at least one clip to begin.' };
  if (C.up.state === 'uploading') return { label: `Uploading… ${t.p}%`, on: false,
    note: 'You can choose your options while your clips upload.' };
  if (C.up.state === 'error') return { label: 'Create my video', on: false,
    note: 'The upload has to finish first: press Try again.' };
  return { label: 'Create my video', on: C.up.state === 'done',
    note: 'A video worker starts only for this, and switches itself off '
      + 'when there is nothing left to do.' };
}

// What each orientation choice will do with THESE clips, in words.
function orientationHint() {
  const known = C.files.filter(x => x.meta);
  const mins = s => (s < 90 ? `${Math.max(1, Math.round(s))} s`
                            : `${Math.round(s / 60)} min`);
  const v = known.filter(x => x.meta.vertical).reduce((a, x) => a + x.meta.seconds, 0);
  const h = known.filter(x => !x.meta.vertical).reduce((a, x) => a + x.meta.seconds, 0);
  const o = C.prefs.orientation;
  if (!known.length || known.length < C.files.length) {
    return 'Horizontal suits YouTube; vertical suits Shorts and Reels.';
  }
  const mixed = v > 0 && h > 0;
  const shapes = mixed
    ? `Your clips: ${mins(v)} vertical, ${mins(h)} horizontal. `
    : `Your clips are all ${v > 0 ? 'vertical' : 'horizontal'}. `;
  const out = o === 'auto' ? (v > h ? 'vertical' : 'horizontal') : o;
  let what = o === 'auto'
    ? `Auto will make a ${out} video, because most of your footage is ${out}.`
    : `The video will be ${out}.`;
  const other = out === 'vertical' ? h : v;
  if (other > 0) {
    what += C.prefs.framing === 'fill'
      ? ` ${out === 'vertical' ? 'Horizontal' : 'Vertical'} clips will be cropped to fill the frame.`
      : ` ${out === 'vertical' ? 'Horizontal' : 'Vertical'} clips will be shown whole, with blurred sides.`;
  }
  return shapes + what;
}

function viewCreate() {
  const f = document.createDocumentFragment();
  f.append(topBar('Create', 'Upload your clips and tell Alpha Titans what you want'));
  const page = el('div', 'cx-page');

  // ── 1. upload ──
  const up = el('section', 'cx-block');
  up.append(el('h2', 'cx-h', 'Upload your clips'));
  const drop = el('div', 'cx-drop' + (C.files.length ? ' is-compact' : ''));
  drop.innerHTML = `
    <div class="cx-drop-in">
      <span class="cx-drop-ico">${svg(ICON.upload)}</span>
      <p class="cx-drop-t">Drop your clips here</p>
      <p class="cx-drop-s">or click to browse from your computer</p>
      <p class="cx-drop-f">MP4 · MOV · M4V · MKV · AVI · multiple clips</p>
    </div>
    <input type="file" accept=".mp4,.mov,.m4v,.mkv,.avi,video/*" multiple hidden>`;
  const input = drop.querySelector('input');
  if (!C.creating) {
    drop.addEventListener('click', () => input.click());
    input.addEventListener('change', () => addFiles(input.files));
    ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => {
      e.preventDefault(); drop.classList.add('is-over'); }));
    ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => {
      e.preventDefault(); drop.classList.remove('is-over'); }));
    drop.addEventListener('drop', e => addFiles(e.dataTransfer.files));
  }
  up.append(drop);
  if (C.note) up.append(el('p', 'cx-note is-warn', esc(C.note)));

  if (C.files.length) {
    const t = totals();
    const status = el('div', 'cl-up' + (C.up.state === 'error' ? ' is-error' : '')
      + (C.up.state === 'done' ? ' is-done' : ''));
    status.innerHTML = `<p class="cl-up-line">${esc(uploadLine())}</p>
      <div class="cl-bar cl-bar-lg"><i style="width:${
        C.up.state === 'done' ? 100 : t.p}%"></i></div>`;
    if (C.up.state === 'error') {
      const again = el('button', 'cx-ex', 'Try again');
      again.type = 'button';
      again.addEventListener('click', restartUpload);
      status.append(again);
    }
    status.append(el('p', 'cx-pref-hint', 'Adding or removing a clip starts '
      + 'the upload again.'));
    up.append(status);

    const grid = el('div', 'cx-clips');
    C.files.forEach((x, i) => {
      const sent = Math.min(C.sent[i] || 0, x.file.size);
      const card = el('div', 'cx-clip');
      card.dataset.clip = i;
      card.innerHTML = `
        <div class="cx-clip-th"></div>
        <div class="cx-clip-meta">
          <p class="cx-clip-n">${esc(x.file.name)}</p>
          <p class="cx-clip-d">${esc(clipLine(x, i))}</p>
          <div class="cl-bar"><i style="width:${
            C.up.state === 'done' ? 100 : pct(sent, x.file.size)}%"></i></div>
        </div>
        ${C.creating ? '' : `<button class="cx-clip-x" title="Remove">${svg(ICON.trash)}</button>`}`;
      card.querySelector('.cx-clip-th').append(x.thumb);
      card.querySelector('.cx-clip-x')?.addEventListener('click', () => removeFile(i));
      grid.append(card);
    });
    up.append(grid);
  }
  page.append(up);

  // ── 2. the instruction ──
  const ask = el('section', 'cx-block');
  ask.append(el('h2', 'cx-h', 'Tell Alpha Titans what you want'));
  const box = el('div', 'cx-ask');
  const ta = el('textarea', 'cx-ta');
  ta.placeholder = 'Tell Alpha Titans what you want to create…';
  ta.rows = 4;
  ta.value = C.instruction;
  ta.disabled = C.creating;
  ta.addEventListener('input', () => { C.instruction = ta.value; });
  box.append(ta);
  ask.append(box);
  const ex = el('div', 'cx-examples');
  EXAMPLES.forEach(t => {
    const b = el('button', 'cx-ex', esc(t));
    b.type = 'button';
    b.disabled = C.creating;
    b.addEventListener('click', () => { C.instruction = t; render(); });
    ex.append(b);
  });
  ask.append(ex);
  page.append(ask);

  // ── 3. preferences, from what the engine actually supports ──
  const o = S.options;
  const prefs = el('section', 'cx-block');
  prefs.append(el('h2', 'cx-h', 'Preferences'));
  const grid = el('div', 'cx-prefs');
  const group = (label, node, hint) => {
    const g = el('div', 'cx-pref');
    g.append(el('h3', 'cx-pref-h', esc(label)));
    g.append(node);
    if (hint) g.append(el('p', 'cx-pref-hint', hint));
    return g;
  };
  grid.append(group('Length', pickRow(DURATIONS, C.prefs.target_seconds,
    v => C.prefs.target_seconds = v)));
  if (o) {
    grid.append(group('Orientation', pickRow(
      ['horizontal', 'vertical', 'auto'].filter(v => o.orientation.includes(v)),
      C.prefs.orientation, v => C.prefs.orientation = v, ORIENT_LABEL),
      esc(orientationHint())));
    grid.append(group('Style', pickRow(o.style, C.prefs.style,
      v => C.prefs.style = v, STYLE_LABEL)));
    grid.append(group('Pacing', pickRow(o.pacing, C.prefs.pacing,
      v => C.prefs.pacing = v, PACE_LABEL)));
    grid.append(group('Music', pickRow(o.music, C.prefs.music,
      v => C.prefs.music = v, MUSIC_LABEL)));
    grid.append(group('Captions', pickRow(o.captions, C.prefs.captions,
      v => C.prefs.captions = v, CAP_LABEL)));
    grid.append(group('Voiceover', pickRow(o.voiceover, C.prefs.voiceover,
      v => C.prefs.voiceover = v, VO_LABEL),
      'Narration is written as text. Nothing here synthesises a voice.'));
    grid.append(group('Quality', pickRow(o.resolution, C.prefs.resolution,
      v => C.prefs.resolution = v)));
  } else {
    grid.append(el('p', 'cx-note', 'The other preferences could not be '
      + 'loaded; the engine will use its defaults.'));
  }
  prefs.append(grid);
  if (o) {
    const adv = el('button', 'cx-adv');
    adv.type = 'button';
    adv.textContent = C.advanced ? 'Hide advanced preferences' : 'Advanced preferences';
    adv.addEventListener('click', () => { C.advanced = !C.advanced; render(); });
    prefs.append(adv);
  }
  if (C.advanced && o) {
    const a = el('div', 'cx-prefs');
    a.append(group('Render quality', pickRow(o.quality, C.prefs.quality,
      v => C.prefs.quality = v), 'Higher settings take longer to render.'));
    a.append(group('Mixed orientations', pickRow(o.framing, C.prefs.framing,
      v => C.prefs.framing = v, FRAMING_LABEL),
      'What to do with a clip shot the other way up from the others.'));
    prefs.append(a);
  }
  page.append(prefs);

  // ── 4. the one button ──
  const cta = ctaState();
  const go_ = el('section', 'cx-go');
  const btn = el('button', 'cx-cta', `${svg(ICON.spark)}<span>${esc(cta.label)}</span>`);
  btn.type = 'button';
  btn.disabled = !cta.on;
  btn.addEventListener('click', createVideo);
  go_.append(btn);
  go_.append(el('p', 'cx-go-note', esc(cta.note)));
  page.append(go_);
  f.append(page);
  return f;
}

/* ───────────────────────────── upload ─────────────────────────────── */

// Progress arrives many times a second. A full redraw for each one rebuilt
// every card, so the figures are updated where they stand instead.
function paintProgress() {
  const t = totals();
  const line = root.querySelector('.cl-up-line');
  if (line) line.textContent = uploadLine();
  const all = root.querySelector('.cl-up .cl-bar > i');
  if (all) all.style.width = `${t.p}%`;
  C.files.forEach((x, i) => {
    const card = root.querySelector(`.cx-clip[data-clip="${i}"]`);
    if (!card) return;
    card.querySelector('.cx-clip-d').textContent = clipLine(x, i);
    const bar = card.querySelector('.cl-bar > i');
    if (bar) bar.style.width = `${pct(C.sent[i] || 0, x.file.size)}%`;
  });
  const cta = ctaState();
  const label = root.querySelector('.cx-go .cx-cta span');
  if (label) label.textContent = cta.label;
}

class Superseded extends Error {}

async function idempotencyKey(files, n = 0) {
  // the same files chosen again in this browser -> the same job, so a
  // closed tab resumes rather than uploading twice; `n` moves past a job
  // those same files already finished
  let salt = getItem('local', SALT_KEY);
  if (!salt) {
    salt = Array.from(crypto.getRandomValues(new Uint8Array(12)),
                      b => b.toString(16).padStart(2, '0')).join('');
    setItem('local', SALT_KEY, salt);
  }
  const sig = files.map(f => `${f.name}|${f.size}|${f.lastModified}`).join('/');
  const digest = await crypto.subtle.digest('SHA-256',
    new TextEncoder().encode(`${salt}/${sig}${n ? `#${n}` : ''}`));
  return 'up-' + Array.from(new Uint8Array(digest).slice(0, 16),
    b => b.toString(16).padStart(2, '0')).join('');
}

function putChunk(url, blob, start, end, total, onSent) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    C.up.xhr = x;
    x.open('PUT', url);
    if (S.token) x.setRequestHeader('Authorization', `Bearer ${S.token}`);
    x.setRequestHeader('Content-Range', `bytes ${start}-${end}/${total}`);
    x.timeout = 180000;
    x.upload.onprogress = e => { if (e.lengthComputable) onSent(e.loaded); };
    x.onload = () => {
      let b = {};
      try { b = JSON.parse(x.responseText || '{}'); } catch { b = {}; }
      if (x.status === 200 || x.status === 409) {
        resolve({ received: Number(b.received) || 0, complete: !!b.complete });
      } else reject(new ApiError(x.status, b));
    };
    x.onabort = () => reject(new Superseded());
    x.onerror = () => reject(new Error('the network dropped'));
    x.ontimeout = () => reject(new Error('the upload timed out'));
    x.send(blob);
  });
}

async function sendFile(jobId, file, t, onProgress, live) {
  const chunk = S.cfg.chunk_bytes;
  let off = t.received || 0, failures = 0;
  onProgress(off);
  while (true) {
    live();
    if (off >= file.size) {
      const s = await api(`${API}/uploads/${jobId}/${t.idx}`);
      if (s.complete) return;
      off = s.received;
    }
    const end = Math.min(off + chunk, file.size) - 1;
    try {
      const ack = await putChunk(t.url, file.slice(off, end + 1), off, end,
                                 file.size, sent => onProgress(off + sent));
      failures = 0;
      if (ack.complete) { onProgress(file.size); return; }
      off = Math.min(ack.received, file.size);           // the SERVER's figure
      onProgress(off);
    } catch (e) {
      if (e instanceof Superseded) throw e;
      if (e instanceof ApiError && e.status >= 400 && e.status < 500
          && e.status !== 429) throw e;
      if (++failures > 8) throw e;
      await new Promise(r => setTimeout(r, Math.min(30000, 1000 * 2 ** failures)));
      live();
      try {
        const s = await api(`${API}/uploads/${jobId}/${t.idx}`);
        if (s.complete) { onProgress(file.size); return; }
        off = s.received;
      } catch { /* keep the last known offset */ }
    }
  }
}

function restartUpload() {
  // A different set of clips is a different upload. The old one stops, and
  // if it never became a video it is cancelled rather than left behind.
  const old = C.up.jobId;
  C.up.run += 1;
  try { C.up.xhr?.abort(); } catch { /* already finished */ }
  if (old) api(`${API}/jobs/${old}/cancel`, { method: 'POST' }).catch(() => {});
  clearTimeout(C.upTimer);
  C.up = { run: C.up.run, state: C.files.length ? 'uploading' : 'idle',
           jobId: null, error: null, xhr: null };
  C.sent = {};
  render();
  const run = C.up.run;
  // a moment's pause, so several clips added at once start one upload
  if (C.files.length) C.upTimer = setTimeout(() => uploadClips(run), 500);
}

async function uploadClips(run) {
  const files = C.files.map(x => x.file);
  const live = () => { if (run !== C.up.run) throw new Superseded(); };
  try {
    let made = null;
    for (let n = 0; n < 6 && !made; n++) {
      // an unfinished upload of these clips is resumed; one that already
      // became a video is left alone and a new one is started
      const r = await api(`${API}/uploads`, { method: 'POST', json: {
        idempotency_key: await idempotencyKey(files, n),
        files: files.map(f => ({ name: f.name, size: f.size,
                                 mime: f.type || 'video/mp4' })),
        settings: {} } });
      live();
      if (r.job.state === 'uploading') made = r;
    }
    if (!made) throw new Error('these clips could not be prepared');
    C.up.jobId = made.job.job_id;
    for (const t of made.files) {
      if (t.verified) { C.sent[t.idx] = files[t.idx].size; paintProgress(); continue; }
      await sendFile(made.job.job_id, files[t.idx], t, n => {
        C.sent[t.idx] = n;
        paintProgress();
      }, live);
    }
    live();
    C.up.state = 'done';
    render();
  } catch (e) {
    if (e instanceof Superseded || run !== C.up.run) return;
    if (e.status === 401) return needToken();
    C.up.state = 'error';
    C.up.error = e.message || 'the upload failed';
    render();
  }
}

async function createVideo() {
  if (C.up.state !== 'done' || C.creating || !C.up.jobId) return;
  const jobId = C.up.jobId;
  const settings = { ...C.prefs, instruction: C.instruction.trim() };
  if (settings.target_seconds == null) delete settings.target_seconds;
  C.creating = true; C.note = null; render();
  try {
    await api(`${API}/uploads/${jobId}/complete`, { method: 'POST',
                                                     json: { settings } });
    rememberJob(jobId);
    C.files.forEach(x => URL.revokeObjectURL(x.url));
    C.files = []; C.sent = {}; C.instruction = ''; C.creating = false;
    C.up = { run: C.up.run + 1, state: 'idle', jobId: null, error: null, xhr: null };
    S.watched = jobId;
    openJob(jobId);
  } catch (e) {
    C.creating = false;
    if (e.status === 401) return needToken();
    C.note = `Could not start the video: ${e.message}`;
    render();
  }
}

/* ═══════════════════════════ PRODUCTION ════════════════════════════ */

function phaseState(nodes, done, now) {
  const ran = nodes.filter(n => done.has(n)).length;
  if (nodes.includes(now)) return 'active';
  if (!ran) return 'pending';
  return ran < nodes.length ? 'active' : 'done';
}

// the order a job moves through; `retrying` sits wherever it last was
const ORDER = ['uploading', 'queued', 'starting_vm', 'downloading',
               'processing', 'uploading_results', 'completed'];

function timeLeft(est) {
  const s = Number(est.remaining_s) || 0;
  if (est.percent >= 100) return 'Done';
  if (s <= 0) return 'Finishing…';
  if (s < 60) return 'Less than a minute left';
  const m = Math.ceil(s / 60);
  return `About ${m} minute${m > 1 ? 's' : ''} left`;
}

function workingScreen(j) {
  const wrap = el('section', 'wx-wrap');
  const stopped = j.state === 'failed' || j.state === 'cancelled';
  wrap.innerHTML = stopped
    ? `<p class="sx-kicker">Stopped</p>
       <h2 class="wx-h">This video was not made</h2>`
    : `<p class="sx-kicker">In production</p>
       <h2 class="wx-h">Creating your video…</h2>
       <p class="wx-sub">This takes a few minutes. You can close this page; the
         work carries on, and opening it again picks it up.</p>`;

  if (stopped) {
    const box = el('div', 'wx-failed');
    box.innerHTML = j.state === 'cancelled'
      ? `<h3>This production was cancelled</h3>
         <p class="sx-dim">Nothing more will run for it.</p>`
      : `<h3>This production stopped</h3>
         <p class="sx-dim">It was tried ${esc(j.attempt)} time(s) of
           ${esc(j.max_attempts)}. Nothing was delivered.</p>
         <pre class="wx-failed-why">${esc(j.error?.message || 'no reason reported')}</pre>`;
    wrap.append(box);
    const again = el('button', 'wx-send', `${svg(ICON.spark)}<span>Start a new video</span>`);
    again.type = 'button';
    again.addEventListener('click', () => go('create'));
    wrap.append(again);
    return wrap;
  }
  const est = j.estimate;
  if (est) {
    const box = el('div', 'cl-progress');
    box.innerHTML = `
      <div class="cl-progress-top"><b>${esc(est.percent)}% done</b>
        <span>${esc(timeLeft(est))}</span></div>
      <div class="cl-bar cl-bar-lg"><i style="width:${esc(est.percent)}%"></i></div>
      <p class="cx-pref-hint">${esc(est.based_on
        ? `Estimated from the last ${est.based_on} video${est.based_on > 1 ? 's' : ''} `
          + 'made here; long footage takes longer to measure.'
        : 'A rough estimate: no video has finished here yet.')}</p>`;
    wrap.append(box);
  }
  if (j.state === 'retrying') {
    wrap.append(el('p', 'wx-count', 'Something went wrong; it will be tried '
      + `again${j.retry_at ? ` at ${esc(when(j.retry_at))}` : ''} — attempt `
      + `${esc(j.attempt)} of ${esc(j.max_attempts)}.`));
  }

  const p = j.progress || {};
  const done = new Set(p.nodes_done || []);
  const now = j.state === 'processing' ? p.current_node : null;
  const at = ORDER.indexOf(j.state);
  const step = (label, idx) => ({ label,
    state: at > idx ? 'done' : at === idx ? 'active' : 'pending' });
  const rows = [
    step('Uploading your footage', 0),
    { label: 'Starting the cloud worker',
      state: at > 2 ? 'done' : (at === 1 || at === 2) ? 'active' : 'pending' },
    step('Fetching your footage', 3),
    ...UX_PHASES.map(([label, nodes]) => ({ label,
      state: at > 4 ? 'done' : at < 4 ? 'pending'
        : phaseState(nodes, done, now) })),
    step('Saving your video', 5),
  ];
  const list = el('ol', 'wx-phases');
  rows.forEach(r => list.append(el('li', `wx-phase is-${r.state}`,
    `<span class="wx-dot"></span><span class="wx-label">${esc(r.label)}</span>
     <span class="wx-tick">${r.state === 'done' ? '✓' : ''}</span>`)));
  wrap.append(list);
  const steps = UX_PHASES.flatMap(([, n]) => n);
  const ran = steps.filter(n => done.has(n)).length;
  if (ran) wrap.append(el('p', 'wx-count',
    `${ran} of ${steps.length} production steps complete`));
  if (j.state === 'uploading') {
    const s = (j.sources || []).reduce((a, x) => a + (x.received || 0), 0);
    const t = (j.sources || []).reduce((a, x) => a + (x.size || 0), 0);
    wrap.append(el('p', 'wx-count', `${mb(s)} of ${mb(t)} uploaded — choose the `
      + 'same files on Create to carry on.'));
  }
  const c = el('button', 'cx-ex', 'Cancel this production');
  c.type = 'button';
  c.addEventListener('click', async () => {
    if (!confirm('Cancel this production?')) return;
    try { S.job = await api(`${API}/jobs/${S.jobId}/cancel`, { method: 'POST' }); }
    catch (e) { S.error = e.message; }
    render();
  });
  wrap.append(c);
  return wrap;
}

async function ticket(o, mode) {
  return (await api(`${o.link}/ticket${mode === 'view' ? '?mode=view' : ''}`,
                    { method: 'POST' })).url;
}

async function save(o) {
  // the browser's own download manager fetches it; the token stays in a
  // header, and the one-time link it gets in exchange lives for a minute
  const a = el('a');
  a.href = await ticket(o, 'download');
  a.download = o.name;
  document.body.append(a); a.click(); a.remove();
}

async function saveAll(outputs) {
  for (const o of outputs) {
    try { await save(o); } catch (e) { S.error = `Download failed: ${e.message}`; }
    await new Promise(r => setTimeout(r, 800));            // one at a time
  }
  if (S.error) render();
}

async function loadMedia(j) {
  // view links for the player and thumbnail; the sheet and report as text
  if (S.media.job === j.job_id) return;
  S.media = { job: j.job_id };
  const by = k => (j.outputs || []).find(o => o.kind === k);
  const text = async o => (await fetch(await ticket(o, 'view'))).text();
  try { if (by('video')) S.media.video = await ticket(by('video'), 'view'); } catch {}
  try { if (by('thumbnail')) S.media.thumb = await ticket(by('thumbnail'), 'view'); } catch {}
  try { if (by('youtube')) S.media.sheet = await text(by('youtube')); } catch {}
  try { if (by('report')) S.media.report = JSON.parse(await text(by('report'))); } catch {}
  try { S.trace = await api(`${API}/jobs/${j.job_id}/trace`); } catch { S.trace = null; }
  if (S.jobId === j.job_id && S.view === 'job') render();
}

function finishedScreen(j) {
  const sec = el('section', 'sx-screen');
  const n = S.media.report || {};
  const sum = S.trace?.summary || {};
  const v = String(sum.verdict || '');
  const inner = el('div', 'sx-screen-in');
  const vertical = Number(n.height) > Number(n.width);
  const player = el('div', 'sx-player' + (vertical ? ' is-vertical' : ''));
  player.innerHTML = S.media.video
    ? `<video controls preload="metadata" ${S.media.thumb
        ? `poster="${esc(S.media.thumb)}"` : ''} src="${esc(S.media.video)}"></video>`
    : '<div class="sx-player-empty">Loading the finished video…</div>';
  inner.append(player);

  const strip = el('dl', 'sx-filmstrip');
  const cell = (k, val, cls = '', small = false) =>
    `<div class="sx-fs ${cls}"><dt>${esc(k)}</dt>
       <dd class="${small ? 'sx-small' : ''}">${val}</dd></div>`;
  strip.innerHTML =
    cell('Title', j.title ? esc(j.title) : NA(''), '', true)
    + cell('Duration', n.duration != null ? `${Number(n.duration).toFixed(1)}s` : NA(''))
    + cell('Resolution', n.height ? `${esc(n.height)}p` : NA(''))
    + cell('Audio', n.audio_codec ? `✓ ${esc(n.audio_codec)}` : NA('none'),
        n.audio_codec ? 'is-ok' : '')
    + cell('Critic', v ? `${v === 'PASS' ? '✓ ' : ''}${esc(v)}${sum.score != null
        ? ` · ${esc(sum.score)}` : ''}` : NA(''),
        v === 'PASS' ? 'is-ok' : v ? 'is-warn' : '');
  inner.append(strip);
  sec.append(inner);

  const after = el('div', 'sx-after');
  const tw = el('div', 'sx-thumbwrap');
  tw.innerHTML = `<h4>Thumbnail</h4>${S.media.thumb
    ? `<img alt="Chosen thumbnail" src="${esc(S.media.thumb)}">`
    : '<p class="sx-dim">No thumbnail recorded.</p>'}`;
  after.append(tw);

  const pack = el('div', 'sx-pack');
  pack.innerHTML = '<h4>Download to this computer</h4>';
  const all = el('button', 'cx-cta', `${svg(ICON.down)}<span>Download everything</span>`);
  all.type = 'button';
  all.addEventListener('click', () => saveAll(j.outputs || []));
  pack.append(all);
  const ul = el('ul', 'cl-files');
  (j.outputs || []).forEach(o => {
    const li = el('li');
    const a = el('a', null, `${svg(ICON.down)}<span>${esc(KIND_LABEL[o.kind] || o.kind)}</span>`);
    a.href = '#';
    a.addEventListener('click', ev => {
      ev.preventDefault();
      save(o).catch(e => { S.error = `Download failed: ${e.message}`; render(); });
    });
    li.append(a, el('span', 'sx-dim', `${esc(o.name)} · ${size(o.size)}`));
    ul.append(li);
  });
  pack.append(ul);
  after.append(pack);
  sec.append(after);

  if (S.media.sheet) {
    const yt = el('div', 'sx-pack cl-yt');
    yt.innerHTML = `<h4>For YouTube</h4><pre class="cl-sheet">${esc(S.media.sheet)}</pre>`;
    sec.append(yt);
  }
  return sec;
}

function viewJob() {
  const f = document.createDocumentFragment();
  const j = S.job;
  f.append(topBar(j?.title || 'Your production', j
    ? `${esc(j.job_id)} · ${esc(j.state.replace(/_/g, ' '))}` : 'Loading…'));
  if (S.error) f.append(el('div', 'sx-err', `<p class="sx-dim">${esc(S.error)}</p>`));
  if (!j) { f.append(el('div', 'sx-skel')); return f; }
  if (j.state === 'completed') {
    loadMedia(j);
    f.append(finishedScreen(j));
    // watched it finish in this tab: the files come down without asking, once
    const key = `at.cloud.got.${j.job_id}`;
    if (S.watched === j.job_id && !getItem('session', key)) {
      setItem('session', key, '1');
      saveAll(j.outputs || []);
    }
  } else {
    f.append(workingScreen(j));
  }
  return f;
}

/* ═══════════════════════════ MY VIDEOS ═════════════════════════════ */

function viewProjects() {
  const f = document.createDocumentFragment();
  f.append(topBar('My videos', 'Everything produced in the cloud'));
  const list = el('div', 'cx-page');
  if (!S.jobs.length) {
    list.append(el('p', 'sx-dim', 'Nothing yet. Create your first video.'));
  }
  S.jobs.forEach(j => {
    const wrap = el('div', 'sx-prod-wrap');
    const b = el('button', 'sx-prod');
    const chip = j.state === 'completed' ? 'pass'
      : j.state === 'failed' || j.state === 'cancelled' ? 'escalate' : 'revise';
    b.innerHTML = `
      <div class="sx-prod-thumb"><span class="sx-play">${svg(ICON.play)}</span></div>
      <div class="sx-prod-main">
        <div class="sx-prod-name">${esc(j.title || 'Untitled video')}</div>
        <div class="sx-prod-meta">${esc(j.job_id)} · ${esc(when(j.created_at))}
          · attempt ${esc(j.attempt)} of ${esc(j.max_attempts)}</div>
      </div>
      <div class="sx-prod-side"><span class="sx-chip ${chip}">${
        esc(j.state.replace(/_/g, ' '))}</span></div>`;
    b.addEventListener('click', () => openJob(j.job_id));
    wrap.append(b);
    list.append(wrap);
  });
  f.append(list);
  return f;
}

/* ─────────────────────────────── shell ────────────────────────────── */

const VIEWS = { create: viewCreate, job: viewJob, projects: viewProjects,
                token: viewToken };

function render() {
  if (!root) return;
  const y = window.scrollY;
  root.textContent = '';
  if (S.error && !['token', 'job'].includes(S.view)) {
    root.append(el('div', 'sx-err', `<h4>Something went wrong</h4>
      <p class="sx-dim">${esc(S.error)}</p>`));
  }
  root.append((VIEWS[S.view] || viewCreate)());
  buildSidebar();
  window.scrollTo(0, y);
}

function writeHash() {
  const h = S.view === 'job' && S.jobId ? `#job=${S.jobId}`
    : S.view === 'projects' ? '#videos' : '#create';
  if (location.hash !== h) history.replaceState(null, '', h);
}

async function go(view) {
  S.error = null;
  if (view === 'job' && S.jobId) return openJob(S.jobId);
  clearTimeout(S.timer);
  S.view = view;
  if (view === 'projects') await refreshList();
  writeHash();
  render();
}

async function refreshList() {
  try {
    const open = S.cfg?.open_access && !S.token;
    const d = await api(`${API}/jobs?${open
      ? `ids=${myJobs().join(',')}` : 'limit=30'}`);
    S.jobs = d.jobs || []; S.worker = d.worker?.state || S.worker;
  } catch (e) { if (e.status === 401) needToken(); }
}

function openJob(id) {
  if (S.jobId !== id) { S.job = null; S.media = {}; S.trace = null; }
  S.jobId = id; S.view = 'job'; S.error = null;
  writeHash();
  render();
  poll();
}

// A redraw would restart a playing <video> and can swallow a click that
// lands mid-redraw, so the page redraws only when what it shows changed
// (a heartbeat alone moves `updated_at` and nothing on screen).
const shown = j => j && JSON.stringify([j.state, j.title, j.progress,
  j.attempt, j.retry_at, j.error, j.sources, (j.outputs || []).length,
  j.estimate && [j.estimate.percent, Math.ceil(j.estimate.remaining_s / 60)]]);

async function poll() {
  clearTimeout(S.timer);
  if (S.view !== 'job' || !S.jobId) return;
  try {
    const j = await api(`${API}/jobs/${S.jobId}`);
    rememberJob(j.job_id);      // a video opened by its link joins My videos
    if (!TERMINAL.has(j.state)) S.watched = j.job_id;
    const worker = j.worker?.state || S.worker;
    const changed = shown(S.job) !== shown(j) || S.error;
    const moved = worker !== S.worker;
    S.job = j; S.worker = worker; S.error = null;
    if (changed) render(); else if (moved) buildSidebar();
    if (TERMINAL.has(j.state)) return;
  } catch (e) {
    if (e.status === 401) return needToken();
    S.error = `Could not reach the server (${e.message}). Retrying.`;
    render();
  }
  S.timer = setTimeout(poll, 3000);
}

function needToken() {
  clearTimeout(S.timer);
  S.token = ''; S.view = 'token'; render();
}

async function boot() {
  S.token = getItem('session', TOKEN_KEY);
  try { S.cfg = await api(`${API}/config`); } catch { S.cfg = null; }
  if (!S.cfg || !S.cfg.enabled) {
    root.textContent = '';
    root.append(topBar('Cloud studio', 'Not available'));
    root.append(el('div', 'sx-err', `<h4>Cloud production is not enabled on this
      server</h4>${((S.cfg && S.cfg.problems) || []).map(p =>
      `<p class="sx-dim">${esc(p)}</p>`).join('')}`));
    buildSidebar();
    return;
  }
  if (S.cfg.auth_required && !S.token) return needToken();
  try { S.options = await api(`${API}/options`); } catch { S.options = null; }
  await refreshList();
  if (S.view === 'token') return;
  const m = /#job=([a-f0-9]{12})/.exec(location.hash);
  if (m) return openJob(m[1]);
  S.view = location.hash === '#videos' ? 'projects' : 'create';
  writeHash();
  render();
}

window.addEventListener('hashchange', () => {
  if (S.view === 'token' || !S.cfg?.enabled) return;
  const m = /#job=([a-f0-9]{12})/.exec(location.hash);
  if (m) { if (m[1] !== S.jobId || S.view !== 'job') openJob(m[1]); }
  else if (location.hash === '#videos' && S.view !== 'projects') go('projects');
  else if (location.hash === '#create' && S.view !== 'create') go('create');
});

root = document.querySelector('#sxView');
sideEl = document.querySelector('#sxSide');
buildSidebar();
boot();
