/* Alpha Titans — Studio.
 *
 * The workspace shell: sidebar, Overview, Production workspace (agent
 * canvas + inspector + video + timeline + critic + packaging), Agents
 * directory, Runs and System Health.
 *
 * DATA HONESTY
 * ------------
 * Every value on screen comes from one of these, unchanged:
 *
 *   /api/v2/health            agent registry, model, reasoning on/off
 *   /api/v2/jobs              builds, verdict, score, attempts
 *   /api/v2/jobs/{id}/status  verdict, totals, artifacts, revisions
 *   /api/v2/jobs/{id}/trace   per-node spans: timings, tools, fallbacks
 *   /api/jobs/{id}            analysis -> real segment times for the timeline
 *   /api/jobs/{id}/plan       decisions, with produced_by and confidence
 *   /api/jobs/{id}/notes      verified figures for the delivered file
 *
 * Where a value does not exist, the UI prints "Unavailable" and says why.
 * Nothing is estimated, averaged into existence, or carried over from
 * another run. In particular: trace spans carry no confidence and no
 * decisions, so only the four agents that write into the edit plan have
 * those; the rest say so.
 *
 * TOPOLOGY
 * --------
 * The canvas draws the graph the backend actually runs (server/agents/
 * graph.py). Note that music -> animation -> transition -> caption is a
 * CHAIN, not a parallel fan-out: each briefs the next. The only genuine
 * parallelism is vision || research. Drawing it any other way would be a
 * picture of a system we do not have.
 */

const V2 = '/api/v2';

/* ───────────────────────────── topology ───────────────────────────── */

const ROWS = [
  { nodes: ['clip_analysis'] },
  { nodes: ['vision', 'research'], parallel: true },
  { nodes: ['content_strategy'] },
  { nodes: ['story'] },
  { nodes: ['voiceover'] },
  { nodes: ['music', 'animation', 'transition', 'caption'], chain: true },
  { nodes: ['editing_director'] },
  { nodes: ['render'] },
  { nodes: ['critic'] },
  { nodes: ['__pass', '__revise'], branch: true },
  { nodes: ['thumbnail', 'title', 'publishing', 'performance_analysis', 'learning'],
    chain: true },
];

const EDGES = [
  ['clip_analysis', 'vision'], ['clip_analysis', 'research'],
  ['vision', 'content_strategy'], ['research', 'content_strategy'],
  ['content_strategy', 'story'], ['story', 'voiceover'],
  ['voiceover', 'music'],
  ['music', 'animation'], ['animation', 'transition'], ['transition', 'caption'],
  ['caption', 'editing_director'],
  ['editing_director', 'render'], ['render', 'critic'],
  ['critic', '__pass'], ['critic', '__revise'],
  ['__revise', 'editing_director', 'cycle'],
  ['__pass', 'thumbnail'],
  ['thumbnail', 'title'], ['title', 'publishing'],
  ['publishing', 'performance_analysis'], ['performance_analysis', 'learning'],
];

const LABEL = {
  clip_analysis: 'Clip Analysis', vision: 'Vision', research: 'Research',
  content_strategy: 'Content Strategy', story: 'Story', voiceover: 'Voiceover',
  music: 'Music', animation: 'Animation', transition: 'Transition',
  caption: 'Caption', editing_director: 'Editing Director', render: 'Rendering',
  critic: 'Critic', thumbnail: 'Thumbnail', title: 'Title',
  publishing: 'Publishing', performance_analysis: 'Performance',
  learning: 'Learning',
};

const ROLE = {
  clip_analysis: 'Measures every clip: motion, dead air, rotation, codec.',
  vision: 'Looks at real frames and reports what each clip shows.',
  research: 'What the world currently says about this subject.',
  content_strategy: 'What this video is saying, and to whom.',
  story: 'Orders the moments into a narrative with a hook and a payoff.',
  voiceover: 'Writes narration against the story beats.',
  music: 'Chooses the mood and the shape of the score.',
  animation: 'Framing and emphasis, within what the renderer can do.',
  transition: 'Picks the join at every cut.',
  caption: 'The words that appear on screen.',
  editing_director: 'Turns all of it into one edit plan.',
  render: 'Runs the real pipeline: cut, join, score, master.',
  critic: 'Inspects the rendered MP4 itself, not the plan that made it.',
  thumbnail: 'Chooses the frame that represents the video.',
  title: 'Names it.',
  publishing: 'Pre-flight checks and a plan for the operator.',
  performance_analysis: 'How the published video performed.',
  learning: 'What this run changes about the next one.',
};

const ICON = {
  clip_analysis: 'M4 7h16v10H4zM9 7v10M15 7v10',
  vision: 'M2 12s3.8-6 10-6 10 6 10 6-3.8 6-10 6-10-6-10-6z|M12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5z',
  research: 'M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM20 20l-4-4',
  content_strategy: 'M4 6h16M4 12h10M4 18h7',
  story: 'M5 4h9l5 5v11H5zM14 4v5h5',
  voiceover: 'M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3zM6 12a6 6 0 0 0 12 0M12 18v3',
  music: 'M9 18V5l11-2v13|M9 18a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0zM20 16a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z',
  animation: 'M4 5h16v14H4zM4 10h16M9 5v14',
  transition: 'M3 12h7M14 12h7M10 8l4 4-4 4',
  caption: 'M4 6h16v12H4zM7 11h4M7 14h8M14 11h3',
  editing_director: 'M4 6h7v5H4zM13 6h7v5h-7zM4 13h16v5H4z',
  render: 'M5 4l14 8-14 8z',
  critic: 'M12 4l2.3 5.5L20 11l-5.7 1.5L12 18l-2.3-5.5L4 11l5.7-1.5z',
  thumbnail: 'M4 5h16v14H4zM8.5 10a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM4 16l5-5 4 4 3-3 4 4',
  title: 'M5 6h14M9 6v12M15 6v12',
  publishing: 'M4 12l16-8-6 16-2.5-6.5z',
  performance_analysis: 'M4 19V9M10 19V5M16 19v-7M22 19H2',
  learning: 'M12 4L3 9l9 5 9-5zM6 11.5V16c0 1.5 2.7 3 6 3s6-1.5 6-3v-4.5',
  nav_overview: 'M4 13h7V4H4zM13 20h7v-9h-7zM13 8h7V4h-7zM4 20h7v-4H4z',
  nav_prod: 'M4 5h16v14H4zM9 9l6 3-6 3z',
  nav_agents: 'M5 6h3M16 6h3M5 12h14M5 18h3M16 18h3|M12 3.6a2.4 2.4 0 1 0 0 4.8 2.4 2.4 0 0 0 0-4.8zM12 15.6a2.4 2.4 0 1 0 0 4.8 2.4 2.4 0 0 0 0-4.8z',
  nav_runs: 'M4 7h16M4 12h16M4 17h10',
  nav_create: 'M12 5v14M5 12h14',
  mic: 'M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3zM6 12a6 6 0 0 0 12 0M12 18v3',
  upload: 'M12 16V4m0 0-4.5 4.5M12 4l4.5 4.5M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3',
  spark: 'M12 3l1.7 5L19 10l-5.3 2L12 17l-1.7-5L5 10l5.3-2z',
  trash: 'M4 7h16M9 7V5h6v2M7 7l1 13h8l1-13',
  send: 'M4 12l16-8-6 16-2.5-6.5z',
  nav_health: 'M3 12h4l2-5 3 10 2.5-5H21',
  nav_assets: 'M4 7l8-3 8 3-8 3zM4 12l8 3 8-3M4 17l8 3 8-3',
  burger: 'M4 7h16M4 12h16M4 17h16',
  check: 'M4 12.5l5 5L20 6.5',
  cross: 'M6 6l12 12M18 6L6 18',
  play: 'M8 5.5v13l10.5-6.5z',
};

const TYPE_LABEL = { reasoning: 'Reasoning agent', hybrid: 'Hybrid agent',
  deterministic: 'Deterministic agent' };

/* one glyph per category - the canvas reads by shape before it reads words */
const GLYPH = { reasoning: '✦', hybrid: '◈', deterministic: '◇' };

/* what each worker does, in the plainest words that are still true */
const SHORT = {
  clip_analysis: 'Measuring your footage',
  vision: 'Understanding your footage',
  research: 'Checking the wider subject',
  content_strategy: 'Deciding what to say',
  story: 'Shaping the narrative',
  voiceover: 'Writing the narration',
  music: 'Scoring the cut',
  animation: 'Framing every shot',
  transition: 'Choosing the joins',
  caption: 'Writing on-screen words',
  editing_director: 'Composing the edit',
  render: 'Building the video',
  critic: 'Watching the finished file',
  thumbnail: 'Picking the cover frame',
  title: 'Naming the video',
  publishing: 'Pre-flight for release',
  performance_analysis: 'Reading the outcome',
  learning: 'Improving the next run',
};

/* the directory reads as a production team, grouped by craft */
const CRAFT = [
  ['Creative intelligence', 'What the video is about',
   ['vision', 'research', 'content_strategy', 'story']],
  ['Media production', 'How it looks and sounds',
   ['voiceover', 'music', 'animation', 'transition', 'caption']],
  ['Execution', 'Building, judging and shipping it',
   ['clip_analysis', 'editing_director', 'render', 'critic',
    'thumbnail', 'title', 'publishing', 'performance_analysis', 'learning']],
];

const glyphFor = n => {
  const m = metaFor(n);
  return GLYPH[m?.agency] || GLYPH.deterministic;
};

/* ───────────────────────────── helpers ────────────────────────────── */

const el = (t, c, h) => { const n = document.createElement(t);
  if (c) n.className = c; if (h != null) n.innerHTML = h; return n; };

const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const dur = n => {
  const v = Number(n) || 0;
  if (!isFinite(v) || v <= 0) return '0ms';
  if (v < 1) return `${Math.round(v * 1000)}ms`;
  if (v < 60) return `${v.toFixed(2)}s`;
  return `${Math.floor(v / 60)}m ${Math.round(v % 60)}s`;
};

// A span that is still RUNNING has no `ended_at`, and this used to return 0
// for it - so an agent four minutes into measuring an eight-minute clip
// displayed "0ms" and read as frozen. While it runs, count from its start
// to now instead. The server writes `started_at` as a unix time and the
// server is this machine, so the two clocks are the same one; a nonsense
// difference (a clock that moved, a span from another host) is treated as
// unknown rather than shown as a wild number.
const MAX_PLAUSIBLE_RUN_S = 6 * 60 * 60;

const wallOf = s => {
  const a = Number(s?.started_at), b = Number(s?.ended_at);
  if (isFinite(a) && isFinite(b) && b >= a) return b - a;
  if (isFinite(a) && s?.status === 'running') {
    const live = Date.now() / 1000 - a;
    if (live >= 0 && live < MAX_PLAUSIBLE_RUN_S) return live;
  }
  return 0;
};

const svg = (d, cls = '') => {
  if (!d) return '';
  const p = d.split('|').map(x => `<path d="${x}" fill="none" stroke="currentColor"
    stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>`).join('');
  return `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true">${p}</svg>`;
};

const short = v => {
  let s = typeof v === 'object' ? JSON.stringify(v) : String(v);
  if (/[\\/]/.test(s) && s.length > 34) s = '…' + s.split(/[\\/]/).pop();
  return s.length > 64 ? s.slice(0, 61) + '…' : s;
};

const NA = why => `<span class="sx-na">Unavailable${why ? ` · ${esc(why)}` : ''}</span>`;

async function get(p) {
  const r = await fetch(p, { headers: { accept: 'application/json' } });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}
const tryGet = async p => { try { return await get(p); } catch { return null; } };

/* ─────────────────────────────── state ────────────────────────────── */

const S = {
  view: 'create',
  health: null, jobs: [], tracks: {}, thumbs: {}, loaded: false, error: null,
  jobId: null, status: null, spans: [], decisions: [], notes: null, job: null,
  selected: null, timer: null, sideOpen: false, loadNote: null,
  activityOpen: false, produced: {}, armedDelete: null,
  artifacts: {},
  published: null,
  // polls taken since the verdict landed, while the file is published
  settled: 0,
};

let root = null, sideEl = null, onCreate = null;

const meta = id => (S.health?.agent_detail || []).find(a => a.id === id) || null;
const metaFor = n => meta(n === 'render' ? 'rendering' : n);
const spanOf = n => S.spans.find(s => s.node === n) || null;
const decsOf = n => {
  const id = n === 'render' ? 'rendering' : n;
  return S.decisions.filter(d => (d.produced_by || d.agent) === id);
};

function stateOf(n) {
  if (n === '__pass') return S.status?.verdict === 'PASS' ? 'ok' : 'queued';
  if (n === '__revise') return (S.status?.attempts > 1
    || S.status?.verdict === 'REVISE') ? 'ok' : 'queued';
  const sp = spanOf(n);
  if (!sp) return S.jobId ? 'queued' : 'queued';
  if (sp.status === 'error') return 'error';
  if (S.status?.current_node === n) return 'running';
  if (sp.fallback_used) return 'degraded';
  return 'ok';
}
const STATE_WORD = { ok: 'Completed', degraded: 'Degraded', error: 'Failed',
  running: 'Running', queued: 'Queued' };

/* ───────────────────────────── sidebar ────────────────────────────── */

/* Create is the product. The agent architecture is the engine, and lives
   below the divider for anyone who wants to look at it. */
const NAV = [
  { group: 'Create', items: [
    ['create', 'Create video', 'nav_create'],
    ['overview', 'Projects', 'nav_prod'],
    ['runs', 'My videos', 'nav_assets'],
  ] },
  { group: 'Production details', items: [
    ['production', 'AI activity', 'nav_overview'],
    ['agents', 'Agents', 'nav_agents'],
    ['health', 'System health', 'nav_health'],
  ] },
];

function buildSidebar() {
  sideEl.textContent = '';
  const brand = el('button', 'sx-brand');
  brand.innerHTML = `<svg class="mark" aria-hidden="true"><use href="#i-brick"/></svg>
    <span class="brand-text"><b>Alpha Titans</b><small>Production</small></span>`;
  brand.addEventListener('click', () => go('overview'));
  sideEl.append(brand);

  NAV.forEach(g => {
    const wrap = el('div', 'sx-navgroup');
    wrap.append(el('div', 'sx-navlabel', esc(g.group)));
    g.items.forEach(([id, label, ico]) => {
      const b = el('button', 'sx-nav' + (S.view === id ? ' is-on' : ''));
      const tag = id === 'agents' && S.health ? S.health.agent_count
        : id === 'runs' ? (S.jobs.length || '') : '';
      b.innerHTML = `${svg(ICON[ico])}<span>${esc(label)}</span>
        ${tag ? `<span class="sx-nav-tag">${tag}</span>` : ''}`;
      b.addEventListener('click', () => {
        closeSide();
        if (id === '__create') { if (onCreate) onCreate(); return; }
        go(id);
      });
      wrap.append(b);
    });
    sideEl.append(wrap);
  });

  const foot = el('div', 'sx-side-foot');
  const live = S.health?.reasoning_enabled;
  foot.innerHTML = `<div class="sx-sysline">
    <i class="sx-sysdot ${S.health ? (live ? 'is-ok' : 'is-warn') : ''}"></i>
    <span>${S.health ? (live ? 'Reasoning on' : 'Deterministic mode') : 'Connecting…'}</span>
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

/* ────────────────────────── shared chrome ─────────────────────────── */

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

const sectionHead = (title, note, action) => {
  const h = el('div', 'sx-sec-h');
  h.innerHTML = `<h3>${esc(title)}</h3>${note ? `<p>${note}</p>` : ''}`;
  if (action) { const a = el('div', 'sx-sec-act'); a.append(action); h.append(a); }
  return h;
};

/* ═══════════════════════════ CREATE ═══════════════════════════════════
 *
 * The product. Upload footage, say what you want, pick a few preferences,
 * press one button. The eighteen agents are the engine and are not
 * mentioned here beyond a single line of fine print.
 *
 * Everything on this screen is wired to real endpoints:
 *   POST /api/jobs                      create the job
 *   POST /api/jobs/{id}/clips           upload (multipart)
 *   GET  /api/jobs/{id}/clips           list
 *   DELETE /api/jobs/{id}/clips/{name}  remove
 *   GET  /api/v2/produce/options        what the engine can honour
 *   POST /api/v2/jobs/{id}/produce      start a real agent run
 *
 * The option lists are fetched, never hardcoded, so this screen cannot
 * offer a style or a mood the backend would reject.
 */

const C = {
  jobId: null, clips: [], options: null,
  instruction: '', transcript: '', transcribing: false,
  prefs: { style: 'auto', pacing: 'auto', music: 'auto', captions: 'auto',
           voiceover: 'none', resolution: '1080p', quality: 'high',
           target_seconds: null },
  advanced: false, listening: false, busy: null, speechError: null,
  uploadError: null,
  rec: null,
};

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
// A clip shot the other way up from the rest: crop it to the canvas so
// the finished video reads as one piece, or scale it in whole with
// blurred sides and keep every pixel.
const FRAMING_LABEL = { fill: 'Fill the frame', fit: 'Fit with blurred sides' };

const EXAMPLES = [
  'Make a fast-paced cinematic reel with energetic music.',
  'A 30-second highlight of the most interesting moments, with captions.',
  'Keep the mechanical detail visible and use a calm score.',
];

/* ───────────────────────────── speech ─────────────────────────────── */

// WHY THIS DOES NOT USE THE BROWSER'S SPEECH API
// ----------------------------------------------
// `webkitSpeechRecognition` streams the microphone to a Google cloud
// service. A packaged browser ships without the key for it, so every
// attempt failed with the error code `network` - "the speech service could
// not be reached" - and the Speak button was dead for everyone. It also
// meant the person's voice left the machine.
//
// This project already ships an OFFLINE transcriber: faster-whisper, wired
// into `POST /api/jobs/{id}/voice`, which stores the note and returns a
// transcript with it. Recording here and posting there needs no cloud
// service, works in any browser with a microphone, and the audio never
// leaves this computer.

function recorderSupport() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia
            && window.MediaRecorder);
}

function pickAudioMime() {
  const want = ['audio/webm;codecs=opus', 'audio/webm',
                'audio/ogg;codecs=opus', 'audio/mp4'];
  for (const m of want) {
    if (window.MediaRecorder?.isTypeSupported?.(m)) return m;
  }
  return '';                      // let the browser choose
}

async function startRecording(onStop) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mime = pickAudioMime();
  const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  const chunks = [];
  rec.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
  rec.onstop = () => {
    // release the mic as soon as recording ends, not when the upload does
    stream.getTracks().forEach(t => t.stop());
    onStop(new Blob(chunks, { type: mime || 'audio/webm' }));
  };
  rec.start();
  return rec;
}

const MIC_SAY = {
  NotAllowedError: 'microphone access was blocked — allow it and try again',
  NotFoundError: 'no microphone was found',
  NotReadableError: 'the microphone is in use by another app',
};

async function transcribeNote(blob) {
  const jobId = await ensureJob();
  const t = blob.type || '';
  const ext = t.includes('ogg') ? 'ogg' : t.includes('mp4') ? 'm4a' : 'webm';
  const fd = new FormData();
  fd.append('file', blob, `voice_note.${ext}`);
  const r = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/voice`,
                        { method: 'POST', body: fd });
  if (!r.ok) {
    let why = `${r.status}`;
    try { why = (await r.json()).detail || why; } catch { /* keep the code */ }
    throw new Error(why);
  }
  return r.json();         // { transcript, transcript_status, duration, … }
}

// `transcript_status` comes straight from extras.transcribe_voice_file
function sayTranscriptProblem(status) {
  if (status === 'empty') return 'nothing was heard — try again, a little closer';
  if (status === 'no_engine') {
    return 'the offline transcriber is not installed — run '
           + '`pip install faster-whisper`';
  }
  return String(status || 'the transcription failed');
}

/* ───────────────────────────── uploads ────────────────────────────── */

async function ensureJob() {
  if (C.jobId) return C.jobId;
  const r = await fetch('/api/jobs', { method: 'POST',
    headers: { 'content-type': 'application/json' }, body: '{}' });
  if (!r.ok) throw new Error(`could not create a project (${r.status})`);
  C.jobId = (await r.json()).id;
  return C.jobId;
}

async function refreshClips() {
  if (!C.jobId) { C.clips = []; return; }
  const d = await tryGet(`/api/jobs/${encodeURIComponent(C.jobId)}/clips`);
  C.clips = Array.isArray(d) ? d : (d?.clips || []);
}

async function uploadFiles(files) {
  const all = [...files];
  const list = all.filter(f => /\.(mp4|mov|webm|m4v|avi|mkv)$/i.test(f.name));
  const skipped = all.length - list.length;
  C.uploadError = null;
  if (!list.length) {
    C.uploadError = all.length
      ? 'Those files are not video containers. Use MP4, MOV or WebM.'
      : null;
    C.busy = null; render(); return;
  }
  C.busy = `Uploading ${list.length} clip${list.length > 1 ? 's' : ''}…`;
  render();
  try {
    await ensureJob();
    const fd = new FormData();
    list.forEach(f => fd.append('files', f, f.name));
    const r = await fetch(`/api/jobs/${encodeURIComponent(C.jobId)}/clips`,
      { method: 'POST', body: fd });
    if (!r.ok) {
      // say what the server actually said; "(400)" on its own is useless
      let why = `${r.status}`;
      try {
        const d = await r.json();
        why = d.detail || d.message || JSON.stringify(d).slice(0, 160);
      } catch { why = (await r.text().catch(() => '')).slice(0, 160) || why; }
      throw new Error(why);
    }
    const d = await r.json().catch(() => null);
    if (d && Array.isArray(d.rejected) && d.rejected.length) {
      C.uploadError = d.rejected
        .map(x => `${x.name}: ${x.why}`).join(' · ');
    }
    await refreshClips();
  } catch (e) {
    // an upload that fails must not take the whole page down with it: the
    // form, the clips already uploaded and the typed brief all survive.
    C.uploadError = `Upload failed — ${String(e.message || e)}`;
  }
  if (skipped && !C.uploadError) {
    C.uploadError = `${skipped} file(s) skipped: not a video container.`;
  }
  C.busy = null;
  render();
}

async function removeClip(name) {
  C.busy = 'Removing…'; render();
  await fetch(`/api/jobs/${encodeURIComponent(C.jobId)}/clips/${encodeURIComponent(name)}`,
    { method: 'DELETE' }).catch(() => {});
  await refreshClips();
  C.busy = null; render();
}

/* ───────────────────────────── the screen ─────────────────────────── */

function pickRow(name, options, current, onPick, labels) {
  const row = el('div', 'cx-pills');
  options.forEach(o => {
    const val = Array.isArray(o) ? o[1] : o;
    const txt = Array.isArray(o) ? o[0] : (labels ? labels[o] || o : o);
    const on = Array.isArray(o) ? current === val : current === val;
    const b = el('button', 'cx-pill' + (on ? ' is-on' : ''), esc(String(txt)));
    b.type = 'button';
    b.addEventListener('click', () => { onPick(val); render(); });
    row.append(b);
  });
  return row;
}

function viewCreate() {
  const f = document.createDocumentFragment();
  f.append(topBar('Create', 'Upload your clips and tell Alpha Titans what you want'));

  const page = el('div', 'cx-page');

  // ── 1. upload ──
  const up = el('section', 'cx-block');
  up.append(el('h2', 'cx-h', 'Upload your clips'));
  const drop = el('div', 'cx-drop' + (C.clips.length ? ' is-compact' : ''));
  drop.innerHTML = `
    <div class="cx-drop-in">
      <span class="cx-drop-ico">${svg(ICON.upload)}</span>
      <p class="cx-drop-t">Drop your clips here</p>
      <p class="cx-drop-s">or click to browse from your computer</p>
      <p class="cx-drop-f">MP4 · MOV · WebM · multiple clips</p>
    </div>
    <input type="file" accept="video/*" multiple hidden>`;
  const input = drop.querySelector('input');
  drop.addEventListener('click', () => input.click());
  input.addEventListener('change', () => uploadFiles(input.files));
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => {
    e.preventDefault(); drop.classList.add('is-over'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => {
    e.preventDefault(); drop.classList.remove('is-over'); }));
  drop.addEventListener('drop', e => uploadFiles(e.dataTransfer.files));
  up.append(drop);

  if (C.busy) up.append(el('p', 'cx-busy', esc(C.busy)));
  if (C.uploadError) {
    const warn = el('p', 'cx-note is-warn', esc(C.uploadError));
    const retry = el('button', 'cx-ex', 'Dismiss');
    retry.addEventListener('click', () => { C.uploadError = null; render(); });
    warn.append(document.createTextNode(' '));
    warn.append(retry);
    up.append(warn);
  }

  if (C.clips.length) {
    const grid = el('div', 'cx-clips');
    C.clips.forEach(c => {
      const name = c.name || c;
      const card = el('div', 'cx-clip');
      card.innerHTML = `
        <div class="cx-clip-th">
          <img alt="" loading="lazy"
            src="/api/jobs/${encodeURIComponent(C.jobId)}/poster/${encodeURIComponent(name)}">
        </div>
        <div class="cx-clip-meta">
          <p class="cx-clip-n">${esc(String(name).replace(/^\d+_/, ''))}</p>
          <p class="cx-clip-d">${c.duration != null
            ? `${Number(c.duration).toFixed(1)}s` : ''}${c.width
            ? ` · ${c.width}×${c.height}` : ''}</p>
        </div>
        <button class="cx-clip-x" title="Remove">${svg(ICON.trash)}</button>`;
      card.querySelector('.cx-clip-x')
        .addEventListener('click', () => removeClip(name));
      grid.append(card);
    });
    up.append(grid);
  }
  page.append(up);

  // ── 2. the instruction: the hero interaction ──
  const ask = el('section', 'cx-block');
  ask.append(el('h2', 'cx-h', 'Tell Alpha Titans what you want'));
  const box = el('div', 'cx-ask' + (C.listening ? ' is-listening' : ''));
  const ta = el('textarea', 'cx-ta');
  ta.placeholder = 'Tell Alpha Titans what you want to create…';
  ta.rows = 4;
  ta.value = C.instruction;
  ta.addEventListener('input', () => { C.instruction = ta.value; });
  box.append(ta);

  const tools = el('div', 'cx-ask-tools');
  const supported = recorderSupport();
  const label = C.transcribing ? 'Transcribing…'
              : C.listening ? 'Stop' : 'Speak';
  const mic = el('button', 'cx-mic' + (C.listening ? ' is-on' : ''));
  mic.type = 'button';
  mic.innerHTML = `${svg(ICON.mic)}<span>${label}</span>`;
  mic.disabled = !supported || C.transcribing;
  mic.addEventListener('click', async () => {
    if (C.transcribing) return;
    if (C.listening) { C.rec?.stop(); return; }   // onstop does the rest
    C.instruction = ta.value;
    C.speechError = null;
    try {
      C.rec = await startRecording(async blob => {
        C.listening = false; C.rec = null; C.transcribing = true; render();
        try {
          const v = await transcribeNote(blob);
          if (v.transcript_status === 'ok' && v.transcript) {
            C.transcript = v.transcript;
            ta.value = (C.instruction ? C.instruction.trim() + ' ' : '')
              + v.transcript;
            C.instruction = ta.value;
          } else {
            C.speechError = sayTranscriptProblem(v.transcript_status);
          }
        } catch (e) {
          C.speechError = String(e.message || e);
        }
        C.transcribing = false; render();
      });
      C.listening = true; render();
    } catch (e) {
      C.speechError = MIC_SAY[e?.name] || String(e?.message || e);
      C.listening = false; C.rec = null; render();
    }
  });
  tools.append(mic);
  if (C.transcript) {
    tools.append(el('span', 'cx-dictated',
      `${svg(ICON.mic)} dictated text included`));
  }
  box.append(tools);
  ask.append(box);

  if (C.listening) {
    ask.append(el('p', 'cx-note',
      'Recording — say what you want, then press Stop. Your voice is '
      + 'transcribed on this computer and never sent to a cloud service.'));
  } else if (C.transcribing) {
    ask.append(el('p', 'cx-note', 'Transcribing your note…'));
  }

  if (!supported) {
    ask.append(el('p', 'cx-note',
      'Voice input needs a microphone and MediaRecorder, which this browser '
      + 'does not provide. Typing works exactly the same — nothing is lost.'));
  } else if (C.speechError) {
    ask.append(el('p', 'cx-note is-warn',
      `Voice input unavailable — ${esc(C.speechError)}. `
      + 'Everything you typed is untouched.'));
  }

  const ex = el('div', 'cx-examples');
  EXAMPLES.forEach(t => {
    const b = el('button', 'cx-ex', esc(t));
    b.type = 'button';
    b.addEventListener('click', () => {
      C.instruction = t; C.transcript = ''; render();
    });
    ex.append(b);
  });
  ask.append(ex);
  page.append(ask);

  // ── 3. preferences, from what the engine actually supports ──
  const o = C.options;
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

  grid.append(group('Length', pickRow('len', DURATIONS, C.prefs.target_seconds,
    v => C.prefs.target_seconds = v)));
  if (o) {
    grid.append(group('Style', pickRow('style', o.style, C.prefs.style,
      v => C.prefs.style = v, STYLE_LABEL)));
    grid.append(group('Pacing', pickRow('pace', o.pacing, C.prefs.pacing,
      v => C.prefs.pacing = v, PACE_LABEL)));
    grid.append(group('Music', pickRow('music', o.music, C.prefs.music,
      v => C.prefs.music = v, MUSIC_LABEL)));
    grid.append(group('Captions', pickRow('cap', o.captions, C.prefs.captions,
      v => C.prefs.captions = v, CAP_LABEL)));
    grid.append(group('Voiceover', pickRow('vo', o.voiceover, C.prefs.voiceover,
      v => C.prefs.voiceover = v, VO_LABEL),
      'Narration is written as text. Nothing here synthesises a voice.'));
    grid.append(group('Quality', pickRow('res', o.resolution, C.prefs.resolution,
      v => C.prefs.resolution = v)));
  }
  prefs.append(grid);

  const adv = el('button', 'cx-adv');
  adv.type = 'button';
  adv.textContent = C.advanced ? 'Hide advanced preferences'
                               : 'Advanced preferences';
  adv.addEventListener('click', () => { C.advanced = !C.advanced; render(); });
  prefs.append(adv);
  if (C.advanced && o) {
    const a = el('div', 'cx-prefs');
    a.append(group('Render quality', pickRow('q', o.quality, C.prefs.quality,
      v => C.prefs.quality = v),
      'Higher settings take longer to render.'));
    a.append(group('Orientation', pickRow('or', o.orientation,
      C.prefs.orientation || 'auto', v => C.prefs.orientation = v)));
    if (o.framing) {
      a.append(group('Mixed orientations', pickRow('fr', o.framing,
        C.prefs.framing || 'fill', v => C.prefs.framing = v, FRAMING_LABEL),
        'What to do with a clip shot the other way up from the others.'));
    }
    prefs.append(a);
  }
  page.append(prefs);

  // ── 4. the one button ──
  const go = el('section', 'cx-go');
  const btn = el('button', 'cx-cta');
  btn.type = 'button';
  btn.innerHTML = `${svg(ICON.spark)}<span>Create my video</span>`;
  const ready = C.clips.length > 0;
  btn.disabled = !ready || !!C.busy;
  btn.addEventListener('click', startProduction);
  go.append(btn);
  go.append(el('p', 'cx-go-note', ready
    ? 'Alpha Titans will analyse your footage and coordinate its production agents.'
    : 'Upload at least one clip to begin.'));
  page.append(go);

  f.append(page);
  return f;
}

/* ───────────────────────────── start it ───────────────────────────── */

async function startProduction() {
  const body = {
    instruction: C.instruction.trim(),
    voice_transcript: C.transcript.trim(),
    style: C.prefs.style, pacing: C.prefs.pacing, music: C.prefs.music,
    captions: C.prefs.captions, voiceover: C.prefs.voiceover,
    resolution: C.prefs.resolution, quality: C.prefs.quality,
  };
  if (C.prefs.target_seconds) body.target_seconds = C.prefs.target_seconds;
  if (C.prefs.orientation) body.orientation = C.prefs.orientation;
  if (C.prefs.framing) body.framing = C.prefs.framing;

  C.busy = 'Starting production…'; render();
  try {
    const r = await fetch(`${V2}/jobs/${encodeURIComponent(C.jobId)}/produce`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body) });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.detail || `could not start (${r.status})`);
    }
    C.busy = null;
    S.jobs = [{ id: C.jobId, verdict: null, attempts: 1 }, ...S.jobs];
    await openProduction(C.jobId);
  } catch (e) {
    // keep the person on Create with their clips, brief and choices intact
    C.busy = null;
    C.uploadError = `Could not start — ${String(e.message || e)}`;
    render();
  }
}

async function loadCreateOptions() {
  if (C.options) return;
  C.options = await tryGet(`${V2}/produce/options`);
  if (C.options) render();
}

/* ═══════════════════════════ view: overview ════════════════════════ */

function viewOverview() {
  const f = document.createDocumentFragment();
  f.append(topBar('Overview', 'Everything this system has produced'));

  const h = S.health;
  const hero = el('section', 'sx-hero');
  hero.innerHTML = `
    <div class="sx-hero-in">
      <p class="sx-kicker">Alpha Titans</p>
      <h2>AI video production<br><em>by eighteen specialized agents</em></h2>
      <p class="sx-hero-lede">Footage goes in. A reasoning layer decides what
        the video is saying, specialists decide how it is cut, scored and
        captioned, and a critic watches the finished file before it ships.</p>
      <div class="sx-spine">
        ${[['01','Raw footage'],['02','AI analysis'],['03','Creative intelligence'],
           ['04','Production'],['05','Quality control'],['06','Final video']]
          .map(([n, t]) => `<div class="sx-spine-step">
             <div class="sx-spine-dot"></div>
             <div class="sx-spine-n">${n}</div>
             <p class="sx-spine-t">${t}</p></div>`).join('')}
      </div>
    </div>`;
  f.append(hero);

  const done = S.jobs.filter(j => j.verdict).length;
  const metrics = el('dl', 'sx-metrics');
  metrics.innerHTML = `
    <div class="sx-metric is-accent"><dt>Agents</dt>
      <dd>${h ? h.agent_count : '—'}<small>${h
        ? `${h.reasoning_agents} reasoning · ${h.agent_count - h.reasoning_agents} measured`
        : ''}</small></dd></div>
    <div class="sx-metric"><dt>Productions</dt>
      <dd>${S.jobs.length}<small>${S.jobs.length
        ? 'built by the agent graph' : 'none yet'}</small></dd></div>
    <div class="sx-metric"><dt>Completed runs</dt>
      <dd>${done}<small>${done ? 'reached a verdict' : 'nothing finished'}</small></dd></div>
    <div class="sx-metric ${h ? (h.reasoning_enabled ? 'is-ok' : 'is-warn') : ''}">
      <dt>System</dt><dd>${h ? (h.reasoning_enabled ? 'Live' : 'Fallback')
        : '—'}<small>${h ? (h.reasoning_enabled ? esc(h.model || '')
        : 'no model key configured') : ''}</small></dd></div>`;
  f.append(metrics);

  const prods = el('section', 'sx-sec');
  prods.append(sectionHead('Recent productions',
    S.jobs.length ? 'Select one to open its workspace' : ''));
  if (!S.jobs.length) {
    prods.append(el('div', 'sx-empty',
      `<h4>No productions yet</h4><p>Run <code class="sx-mono">python -m agents.runner run --src &lt;folder&gt;</code>
       to put footage through the graph.</p>`));
  } else {
    const list = el('div', 'sx-prods');
    S.jobs.forEach(j => list.append(prodRow(j)));
    prods.append(list);
  }
  f.append(prods);

  const act = el('section', 'sx-sec');
  act.append(sectionHead('Agent activity',
    S.jobId ? `Latest trace · <span class="sx-mono">${esc(S.jobId)}</span>`
            : 'Open a production to see its trace'));
  act.append(S.spans.length ? activityFeed() : el('div', 'sx-empty',
    '<p>No run selected.</p>'));
  f.append(act);
  return f;
}

function prodRow(j) {
  const b = el('button', 'sx-prod');
  const v = String(j.verdict || '—');
  // the real filename, read from the thumbnail agent's own frame_grab call.
  // Guessing a name here produced 404s for runs made before that agent existed.
  const tn = S.thumbs[j.id];
  b.innerHTML = `
    <div class="sx-prod-thumb">
      ${tn ? `<img src="/api/jobs/${encodeURIComponent(j.id)}/file/${encodeURIComponent(tn)}"
           alt="" loading="lazy">` : ''}
      <span class="sx-play">${svg(ICON.play)}</span>
    </div>
    <div class="sx-prod-main">
      <div class="sx-prod-name">${esc(j.id)}</div>
      <div class="sx-prod-meta">${j.attempts || 1} attempt${(j.attempts || 1) > 1 ? 's' : ''}
        ${j.score != null ? ` · score ${j.score}` : ''}</div>
      <div class="sx-ticks">${tickRow(j.id)}</div>
    </div>
    <div class="sx-prod-side">
      <span class="sx-chip ${v.toLowerCase()}">${esc(v)}</span>
    </div>`;
  b.addEventListener('click', () => openProduction(j.id));

  // The row itself is a <button>, so the delete cannot live inside it.
  // Deleting takes the video, its clips, its artifacts and its trace, and
  // none of that comes back - so it asks first, in place. One stray click
  // only arms it; the second one, on a button that now says what it will
  // do, carries it out.
  const wrap = el('div', 'sx-prod-wrap');
  wrap.append(b);
  const armed = S.armedDelete === j.id;
  const del = el('button', 'sx-prod-del' + (armed ? ' is-armed' : ''));
  del.type = 'button';
  del.title = armed ? 'This cannot be undone' : 'Delete this production';
  del.textContent = armed ? 'Delete for good' : '×';
  del.addEventListener('click', async ev => {
    ev.stopPropagation();
    if (!armed) { S.armedDelete = j.id; render(); return; }
    S.armedDelete = null;
    del.disabled = true;
    del.textContent = 'Deleting…';
    try {
      const r = await fetch(`${V2}/jobs/${encodeURIComponent(j.id)}`,
                            { method: 'DELETE' });
      if (!r.ok) {
        let why = `${r.status}`;
        try { why = (await r.json()).detail || why; } catch { /* keep it */ }
        throw new Error(why);
      }
      if (S.jobId === j.id) { S.jobId = null; S.status = null; S.spans = []; }
      await load();                       // the list is re-read, not guessed
    } catch (e) {
      S.error = `Could not delete ${j.id}: ${e.message || e}`;
      render();
    }
  });
  wrap.append(del);
  return wrap;
}

/* One tick per agent that ran, coloured by what the trace recorded. Grey
   means the backend has not been asked yet; the row fills in as the
   per-job traces arrive. Nothing is drawn for an agent that did not run. */
function tickRow(id) {
  const t = S.tracks[id];
  if (!t) return Array.from({ length: 18 },
    () => '<i class="sx-tick"></i>').join('');
  return t.map(st => `<i class="sx-tick is-${st}"></i>`).join('')
    + Array.from({ length: Math.max(0, 18 - t.length) },
        () => '<i class="sx-tick"></i>').join('');
}

function activityFeed() {
  const f = el('div', 'sx-feed');
  S.spans.forEach(sp => {
    const st = stateOf(sp.node);
    const row = el('div', `sx-ev is-${st}`);
    const ico = st === 'ok' ? ICON.check : st === 'degraded'
      ? 'M12 4l9 16H3z' : st === 'error' ? ICON.cross : ICON.check;
    row.innerHTML = `
      <span class="sx-ev-seq">${String(sp.seq).padStart(2, '0')}</span>
      <span class="sx-ev-ico">${svg(ico)}</span>
      <span class="sx-ev-txt"><b>${esc(LABEL[sp.node] || sp.node)}</b>
        <span>${sp.fallback_used
          ? esc(String(sp.fallback_used).split('(')[0].trim())
          : `${(sp.tool_calls || []).length} tool call${(sp.tool_calls || []).length === 1 ? '' : 's'}`}</span></span>
      <span class="sx-ev-t">${dur(wallOf(sp))}</span>`;
    row.addEventListener('click', () => { openProduction(S.jobId, sp.node); });
    f.append(row);
  });
  return f;
}

/* ══════════════════════ view: production workspace ═════════════════ */

/* The five phases a person recognises, mapped onto the real graph nodes.
   This is a VIEW of the trace, never a substitute for it: a phase is only
   "done" when every node behind it has actually recorded a span. */
const UX_PHASES = [
  ['Analyzing your footage', ['clip_analysis', 'vision', 'research']],
  ['Understanding your request', ['content_strategy']],
  ['Building the story', ['story', 'voiceover', 'music', 'animation',
                          'transition', 'caption']],
  ['Editing your video', ['editing_director', 'render']],
  ['Final quality check', ['critic']],
];

function phaseState(nodes) {
  const ran = nodes.filter(n => spanOf(n)).length;
  if (!ran) return 'pending';
  if (ran < nodes.length) return 'active';
  return nodes.some(n => S.status?.current_node === n) ? 'active' : 'done';
}

function workingScreen() {
  const wrap = el('section', 'wx-wrap');
  const t = S.status?.totals || {};
  wrap.innerHTML = `
    <p class="sx-kicker">In production</p>
    <h2 class="wx-h">Creating your video…</h2>
    <p class="wx-sub">This takes a few minutes. You can leave this page open.</p>`;
  // A run can die before it writes anything - ffmpeg missing from PATH is
  // the usual cause. The server now reports that instead of 404ing, and it
  // must be shown: the phase list below would otherwise sit at "queued"
  // forever and read as "still working".
  // Left half-built when the server stopped. Different from a failure:
  // nothing went wrong, the thread simply went away with the process,
  // and the checkpoint on disk can carry it the rest of the way. Shown
  // BEFORE the phase list, which would otherwise sit at "queued" and
  // read as a build still in progress - which is exactly what happened
  // three times in one afternoon.
  const stopped = S.status?.interrupted;
  if (stopped && !S.status?.failed) {
    const box = el('div', 'wx-failed');
    box.innerHTML = `<h3>This production was interrupted</h3>
      <p class="sx-dim">${esc(stopped.why || '')}</p>
      <p class="wx-count">${esc(stopped.nodes_done)} step(s) finished${
        stopped.stopped_at ? `, stopped at ${esc(stopped.stopped_at)}` : ''}</p>`;
    const go = el('button', 'wx-send');
    go.innerHTML = `${svg(ICON.send)}<span>Continue from where it stopped</span>`;
    go.addEventListener('click', async () => {
      go.disabled = true;
      go.querySelector('span').textContent = 'Continuing…';
      try {
        const r = await fetch(
          `${V2}/jobs/${encodeURIComponent(S.jobId)}/resume`,
          { method: 'POST' });
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          throw new Error(d.detail || `could not resume (${r.status})`);
        }
        await openProduction(S.jobId);
      } catch (e) {
        go.disabled = false;
        go.querySelector('span').textContent = 'Continue from where it stopped';
        box.append(el('p', 'cx-note is-warn', esc(String(e.message || e))));
      }
    });
    box.append(go);
    wrap.append(box);
    return wrap;
  }

  const failed = S.status?.failed;
  if (failed) {
    const box = el('div', 'wx-failed');
    box.innerHTML = `<h3>This production stopped</h3>
      <p class="sx-dim">The agents could not finish. Nothing was rendered.</p>
      <pre class="wx-failed-why">${esc(String(failed.error || 'no reason reported'))}</pre>`;
    if (failed.traceback) {
      const d = el('details', 'wx-failed-more');
      d.innerHTML = `<summary>Technical detail</summary>
        <pre>${esc(String(failed.traceback))}</pre>`;
      box.append(d);
    }
    wrap.append(box);
    return wrap;
  }
  if (S.status && S.status.starting && !S.spans.length) {
    wrap.append(el('p', 'wx-count', 'Starting the production…'));
  }
  if (finishing()) {
    wrap.append(el('p', 'wx-count',
      'Your video passed review — writing the title and publishing it…'));
  }
  const list = el('ol', 'wx-phases');
  UX_PHASES.forEach(([label, nodes]) => {
    const st = phaseState(nodes);
    list.append(el('li', `wx-phase is-${st}`,
      `<span class="wx-dot"></span><span class="wx-label">${esc(label)}</span>
       <span class="wx-tick">${st === 'done' ? '✓' : ''}</span>`));
  });
  wrap.append(list);
  if (t.nodes) wrap.append(el('p', 'wx-count',
    `${t.nodes} of 18 production steps complete`));
  return wrap;
}

function refineBox() {
  const box = el('section', 'wx-refine');
  box.innerHTML = `<h3>Refine with AI</h3>
    <p class="sx-dim">Describe a change. Alpha Titans produces a new version
      from the same clips, carrying your original brief plus this.</p>`;
  const row = el('div', 'wx-refine-row');
  const ta = el('textarea', 'wx-refine-ta');
  ta.rows = 2;
  ta.placeholder = 'e.g. make the opening faster and use calmer music';
  const send = el('button', 'wx-send');
  send.innerHTML = `${svg(ICON.send)}<span>Send</span>`;
  send.addEventListener('click', async () => {
    const text = ta.value.trim();
    if (!text) return;
    send.disabled = true; send.querySelector('span').textContent = 'Starting…';
    try {
      // prefer what this browser sent; otherwise the settings the backend
      // recorded for the run, so refining a job from a reloaded page keeps
      // the original brief, mood, length and quality instead of resetting
      const prev = S.produced[S.jobId] || S.status?.settings || {};
      const body = Object.assign({}, prev, {
        instruction: [prev.instruction, text].filter(Boolean).join('\n'),
      });
      const r = await fetch(`${V2}/jobs/${encodeURIComponent(S.jobId)}/produce`, {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body) });
      if (r.status === 409 && /already being produced/.test(
            (await r.clone().json().catch(() => ({}))).detail || '')) {
        // Not an error worth showing: the person pressed Send twice
        // because the first press looked as though it had done nothing.
        // Show them the run that is already going.
        await openProduction(S.jobId);
        return;
      }
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(d.detail || `could not start (${r.status})`);
      }
      S.produced[S.jobId] = body;
      // the old verdict is still on disk until the new run overwrites it;
      // show production now rather than the previous result
      S.status = { job_id: S.jobId, verdict: null, totals: {}, starting: true };
      S.spans = []; S.selected = null;
      render();
      await openProduction(S.jobId);
    } catch (e) {
      send.disabled = false; send.querySelector('span').textContent = 'Send';
      box.append(el('p', 'cx-note is-warn', esc(String(e.message || e))));
    }
  });
  row.append(ta, send);
  box.append(row);
  return box;
}

/* The one link between an edit and its reception.
   Nothing in a build can know its own YouTube id - the upload happens
   outside this system - so the operator records it once, here, after
   uploading. Without it the Learning agent has nothing to learn FROM:
   a view count that cannot be attributed to a decision teaches nothing.
   The numbers themselves come from a CSV exported from the warehouse and
   dropped in data/analytics; this panel only ever reports what the
   server matched, and says plainly when it matched nothing. */
function publishedBox() {
  const box = el('section', 'wx-refine');
  box.innerHTML = `<h3>Published to YouTube?</h3>
    <p class="sx-dim">Paste the link after you upload. That is how
      performance gets attached to this edit - and how the Learning agent
      eventually gets something to learn from.</p>`;

  const pub = S.published;
  if (pub && pub.video_id) {
    const line = el('p', 'cx-note');
    line.innerHTML = `Recorded: <span class="sx-mono">${esc(pub.video_id)}</span>`;
    box.append(line);

    if (pub.matched && pub.metrics) {
      const m = pub.metrics;
      const g = el('div', 'wx-metrics');
      [['Views', m.views], ['Watch minutes', m.watch_minutes],
       ['Likes', m.likes], ['Subscribers', m.subscribers_gained]]
        .forEach(([label, v]) => g.append(el('div', 'wx-metric',
          `<b>${esc(Math.round(Number(v) || 0).toLocaleString())}</b>
           <span>${esc(label)}</span>`)));
      box.append(g);
      box.append(el('p', 'sx-dim',
        `from ${esc(m.source_file)}, covering ${esc(m.window_days)} day(s)`));
    } else {
      // never zeroes: absent data must not read as a measured nothing
      box.append(el('p', 'cx-note is-warn', esc(pub.why || 'no analytics yet')));
    }
    (pub.export_skipped || []).forEach(line =>
      box.append(el('p', 'cx-note is-warn', esc(line))));

    const undo = el('button', 'wx-ghost', 'Remove this link');
    undo.addEventListener('click', async () => {
      undo.disabled = true;
      await fetch(`${V2}/jobs/${encodeURIComponent(S.jobId)}/published`,
                  { method: 'DELETE' });
      S.published = null; await refreshPublished(S.jobId); render();
    });
    box.append(undo);
    return box;
  }

  const row = el('div', 'wx-refine-row');
  const input = el('input', 'wx-refine-ta');
  input.type = 'text';
  input.placeholder = 'https://youtu.be/… or the 11-character id';
  const save = el('button', 'wx-send');
  save.innerHTML = `${svg(ICON.send)}<span>Save</span>`;
  const send = async () => {
    const url = input.value.trim();
    if (!url) return;
    save.disabled = true; save.querySelector('span').textContent = 'Saving…';
    try {
      const r = await fetch(
        `${V2}/jobs/${encodeURIComponent(S.jobId)}/published`,
        { method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ url }) });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(d.detail || `could not save (${r.status})`);
      }
      await refreshPublished(S.jobId); render();
    } catch (e) {
      save.disabled = false; save.querySelector('span').textContent = 'Save';
      box.append(el('p', 'cx-note is-warn', esc(String(e.message || e))));
    }
  };
  save.addEventListener('click', send);
  input.addEventListener('keydown', ev => { if (ev.key === 'Enter') send(); });
  row.append(input, save);
  box.append(row);
  return box;
}

async function refreshPublished(id) {
  S.published = await tryGet(`${V2}/jobs/${encodeURIComponent(id)}/published`);
}

function viewProduction() {
  const f = document.createDocumentFragment();
  const picker = el('select', 'wx-picker');
  picker.innerHTML = `<option value="">Select a production…</option>` +
    S.jobs.map(j => `<option value="${esc(j.id)}"${j.id === S.jobId ? ' selected' : ''}>${esc(j.id)}</option>`).join('');
  picker.addEventListener('change', () => picker.value && openProduction(picker.value));

  // `finishing` keeps the working screen up for the last few seconds after
  // the verdict, while the file is published. Showing the result early is
  // what put a broken player in front of the person.
  // `producing` is the server's own answer about the thread; the verdict
  // is the PREVIOUS run's until the new one overwrites it, so a refine
  // that had just started read as finished and bounced the person back
  // to the old video with its Send button, as if nothing had happened.
  const running = !!(S.jobId && S.status &&
    (S.status.producing || S.status.starting ||
     S.status.interrupted || !S.status.verdict || finishing()));
  f.append(topBar(running ? 'In production' : 'Your video',
    S.jobId ? `<span class="sx-mono">${esc(S.jobId)}</span>` : 'Nothing open',
    picker));

  if (!S.jobId) {
    f.append(el('div', 'sx-empty',
      `<h4>Nothing open</h4><p>Start one from Create, or pick a past production above.</p>`));
    return f;
  }

  if (running) f.append(workingScreen());
  else { f.append(videoStage()); f.append(refineBox());
         f.append(publishedBox()); }

  // the engine, folded away
  const t = S.status?.totals || {};
  // The server's own total counts a RUNNING span as zero seconds, so a
  // build four minutes into its first agent reported "0ms". Sum the
  // spans here instead, where a running one is measured to now.
  const liveWall = S.spans.reduce((n, sp) => n + wallOf(sp), 0)
                 || Number(t.wall_s) || 0;
  const det = el('details', 'wx-activity');
  if (S.activityOpen) det.open = true;
  det.addEventListener('toggle', () => { S.activityOpen = det.open; });
  det.innerHTML = `<summary>Show AI activity
    <span class="wx-act-meta">${t.nodes ?? 0} agents · ${dur(liveWall)}</span>
  </summary>`;
  const inner = el('div', 'wx-act-body');
  const work = el('div', 'sx-work' + (S.selected ? ' has-insp' : ''));
  const cw = el('div', 'sx-canvas-wrap');
  cw.append(el('div', 'sx-canvas-bar', `
    <b>${t.nodes ?? 0}</b> agents ran
    <span>·</span> <b>${dur(liveWall)}</b>
    <span>·</span> <b>${t.tool_calls ?? 0}</b> tool calls
    <span class="sx-legend">
      <span class="sx-leg ok"><i></i>completed</span>
      <span class="sx-leg deg"><i></i>degraded</span>
      <span class="sx-leg run"><i></i>running</span>
      <span class="sx-leg queued"><i></i>queued</span>
    </span>`));
  cw.append(canvas());
  work.append(cw);
  if (S.selected) work.append(inspector());
  inner.append(work);
  det.append(inner);
  f.append(det);

  if (!running) {
    f.append(timeline());
    f.append(criticSection());
    f.append(packagingSection());
  }
  return f;
}

function nodeEl(n) {
  if (n === '__pass' || n === '__revise') {
    const st = stateOf(n);
    const b = el('div', `sx-branch ${n === '__pass' ? 'pass' : 'rev'}`);
    b.dataset.node = n;
    b.style.opacity = st === 'ok' ? '1' : '.4';
    b.textContent = n === '__pass' ? 'Pass' : 'Revision';
    return b;
  }
  const m = metaFor(n), sp = spanOf(n), st = stateOf(n);
  const b = el('button', `sx-node s-${st} t-${m?.agency || 'deterministic'}`
    + (S.selected === n ? ' is-sel' : ''));
  b.dataset.node = n;
  b.type = 'button';
  b.setAttribute('aria-pressed', S.selected === n ? 'true' : 'false');
  b.innerHTML = `
    <span class="sx-node-hd">
      <span class="sx-glyph">${glyphFor(n)}</span>
      <span class="sx-node-name">${esc(LABEL[n] || n)}</span>
    </span>
    <p class="sx-node-role">${esc(SHORT[n] || '')}</p>
    <span class="sx-node-ft">
      <span class="sx-node-st"><i></i>${STATE_WORD[st]}</span>
      <span class="sx-node-ms">${sp ? dur(wallOf(sp)) : ''}</span>
    </span>
    ${st === 'degraded'
      ? '<span class="sx-warnmark" title="ran its deterministic fallback">△</span>'
      : ''}`;
  b.addEventListener('click', () => {
    S.selected = S.selected === n ? null : n;
    render();
  });
  return b;
}

function canvas() {
  const c = el('div', 'sx-canvas');
  const edges = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  edges.setAttribute('class', 'sx-edges');
  c.append(edges);
  const rows = el('div', 'sx-rows');
  ROWS.forEach(r => {
    const row = el('div', 'sx-row' + (r.chain ? ' is-chain' : ''));
    r.nodes.forEach(n => row.append(nodeEl(n)));
    rows.append(row);
  });
  c.append(rows);
  // NOT drawn here: this element is still detached, and a later render can
  // replace it before a queued frame fires. render() draws once the tree is
  // actually in the document.
  return c;
}

function paintEdges() {
  const c = root?.querySelector('.sx-canvas');
  const s = c?.querySelector('.sx-edges');
  if (c && s) drawEdges(c, s);
}

function drawEdges(wrap, svgEl) {
  const box = wrap.getBoundingClientRect();
  svgEl.setAttribute('viewBox', `0 0 ${box.width} ${box.height}`);
  svgEl.style.width = box.width + 'px';
  svgEl.style.height = box.height + 'px';
  svgEl.textContent = '';
  // Arrowheads on same-row edges. Those rows are a CHAIN (music briefs
  // animation briefs transition briefs caption); without a direction mark a
  // row of four reads as parallel, which is the one thing it is not.
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  defs.innerHTML = `<marker id="sxArrow" viewBox="0 0 8 8" refX="7" refY="4"
      markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0.5,1 L7,4 L0.5,7" fill="none" stroke="currentColor"
        stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>`;
  svgEl.append(defs);
  const at = n => {
    const e = wrap.querySelector(`[data-node="${n}"]`);
    if (!e) return null;
    const r = e.getBoundingClientRect();
    return { l: r.left - box.left, r: r.right - box.left,
             t: r.top - box.top, b: r.bottom - box.top,
             cx: r.left + r.width / 2 - box.left,
             cy: r.top + r.height / 2 - box.top };
  };
  EDGES.forEach(([a, b, kind]) => {
    const A = at(a), B = at(b);
    if (!A || !B) return;
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    let d;
    if (kind === 'cycle') {
      const x = Math.max(A.r, B.r) + 44;
      d = `M${A.r},${A.cy} C${x},${A.cy} ${x},${B.cy} ${B.r},${B.cy}`;
      p.setAttribute('class', 'is-cycle');
    } else if (Math.abs(A.cy - B.cy) < 12) {
      d = `M${A.r + 3},${A.cy} L${B.l - 9},${B.cy}`;      // same row: a step
      p.setAttribute('marker-end', 'url(#sxArrow)');
      p.classList.add('is-step');
    } else {
      const m = (A.b + B.t) / 2;
      d = `M${A.cx},${A.b} C${A.cx},${m} ${B.cx},${m} ${B.cx},${B.t}`;
    }
    p.setAttribute('d', d);
    const sa = stateOf(a), sb = stateOf(b);
    if (sa === 'running' || sb === 'running') p.classList.add('is-active');
    else if (sa !== 'queued' && sb !== 'queued') p.classList.add('is-done');
    svgEl.append(p);
    if (sb === 'running' && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('r', '2.5'); dot.setAttribute('class', 'sx-flow');
      const anim = document.createElementNS('http://www.w3.org/2000/svg', 'animateMotion');
      anim.setAttribute('dur', '1.4s'); anim.setAttribute('repeatCount', 'indefinite');
      anim.setAttribute('path', d);
      dot.append(anim); svgEl.append(dot);
    }
  });
}

/* ───────────────────────────── inspector ──────────────────────────── */

function inspector() {
  const n = S.selected;
  const m = metaFor(n), sp = spanOf(n), st = stateOf(n);
  const a = el('aside', 'sx-insp');
  const top = el('div', 'sx-insp-top');
  top.innerHTML = `
    <div class="sx-insp-hd">
      <span class="sx-glyph">${glyphFor(n)}</span>
      <h3>${esc(LABEL[n] || n)}</h3>
    </div>
    <p class="sx-insp-role">${esc(SHORT[n] || '')}</p>
    <button class="sx-x" aria-label="Close">&times;</button>`;
  top.querySelector('.sx-x').addEventListener('click', () => { S.selected = null; render(); });
  a.append(top);

  const body = el('div', 'sx-insp-body');
  const ds = decsOf(n);
  const conf = ds.length
    ? (ds.reduce((x, d) => x + (Number(d.confidence) || 0), 0) / ds.length).toFixed(2)
    : null;

  body.append(el('div', 'sx-blk', `
    <dl class="sx-vitals">
      <div class="sx-vital s-${st}"><dt>Status</dt><dd>${STATE_WORD[st]}</dd></div>
      <div class="sx-vital"><dt>Execution</dt>
        <dd>${sp ? dur(wallOf(sp)) : '—'}</dd></div>
      <div class="sx-vital"><dt>Confidence</dt>
        <dd>${conf != null ? conf
          : '<span class="sx-na" style="font-size:13px">Unavailable</span>'}</dd></div>
      <div class="sx-vital"><dt>Type</dt>
        <dd style="font-size:14px;font-weight:450">${esc(m
          ? (TYPE_LABEL[m.agency] || m.agency) : '—')}</dd></div>
    </dl>
    <p class="sx-dim" style="margin:18px 0 0">${esc(ROLE[n] || '')}</p>`));

  const dec = el('div', 'sx-blk');
  dec.innerHTML = '<h4>Decision</h4>';
  if (ds.length) {
    ds.forEach(d => dec.append(el('div', 'sx-dec', `
      <div class="sx-dec-h"><strong>${esc(d.what)}</strong>
        <span class="sx-chip">${esc(d.decided_by || '')}</span></div>
      <p class="sx-dec-v">${esc(String(d.chose))}</p>
      ${d.why ? `<p class="sx-dec-w">${esc(d.why)}</p>` : ''}
      ${d.confidence != null ? `<div class="sx-meter"><i style="width:${
        Math.round(Number(d.confidence) * 100)}%"></i></div>` : ''}`)));
  } else {
    dec.append(el('p', 'sx-dim',
      `${NA('')}. Structured decisions are returned only by
       <code class="sx-mono">/api/jobs/{id}/plan</code>, which covers the four
       agents that write into the edit plan.`));
  }
  body.append(dec);

  const tools = (sp?.tool_calls) || [];
  const tb = el('div', 'sx-blk');
  tb.innerHTML = `<h4>Tools${tools.length ? ` · ${tools.length}` : ''}</h4>`;
  if (tools.length) {
    tools.forEach(t => {
      const { tool, at, ...rest } = t;
      const ps = Object.entries(rest)
        .filter(([, v]) => v !== null && v !== undefined && v !== '')
        .map(([k, v]) => `<span class="sx-param"><b>${esc(k)}</b>${esc(short(v))}</span>`)
        .join('');
      tb.append(el('div', 'sx-tool', `
        <div class="sx-tool-h"><code>${esc(tool)}</code>
          <span class="sx-tool-t">+${dur(at)}</span></div>
        ${ps ? `<div class="sx-params">${ps}</div>` : ''}`));
    });
  } else tb.append(el('p', 'sx-dim', 'No tool calls recorded.'));
  body.append(tb);

  const arts = (sp?.output_artifacts) || [];
  const ab = el('div', 'sx-blk');
  ab.innerHTML = '<h4>Artifacts</h4>';
  if (arts.length) {
    arts.forEach(id => ab.append(el('dl', 'sx-dl', `
      <dt>ID</dt><dd><code>${esc(id)}</code></dd>
      <dt>Kind</dt><dd>${m ? esc(m.produces) : '—'}</dd>
      <dt>Revision</dt><dd>${S.status?.revisions?.[m?.produces] ?? NA('not returned')}</dd>
      <dt>Produced by</dt><dd>${esc(sp.agent || n)}</dd>
      <dt>Degraded</dt><dd>${sp.fallback_used ? 'Yes' : 'No'}</dd>
      <dt>Fallback</dt><dd>${sp.fallback_used
        ? esc(String(sp.fallback_used).split('(')[0].trim()) : '—'}</dd>`)));
  } else ab.append(el('p', 'sx-dim', 'No artifact recorded on this span.'));
  body.append(ab);

  body.append(el('div', 'sx-blk', `<h4>Execution</h4>
    <dl class="sx-dl">
      <dt>Start</dt><dd>${sp?.started_at
        ? new Date(sp.started_at * 1000).toLocaleTimeString() : NA('')}</dd>
      <dt>End</dt><dd>${sp?.ended_at
        ? new Date(sp.ended_at * 1000).toLocaleTimeString() : NA('')}</dd>
      <dt>Duration</dt><dd>${sp ? dur(wallOf(sp)) : NA('')}</dd>
      <dt>Retries</dt><dd>${sp ? (sp.retry_count || 0) : NA('')}</dd>
      <dt>Attempt</dt><dd>${sp ? sp.attempt : NA('')}</dd>
    </dl>`));

  const wb = el('div', 'sx-blk');
  wb.innerHTML = '<h4>Warnings</h4>';
  if (sp?.error) wb.append(el('div', 'sx-warn-box err',
    `<strong>Error</strong><p>${esc(sp.error)}</p>`));
  if (sp?.fallback_used) wb.append(el('div', 'sx-warn-box deg',
    `<strong>Deterministic fallback</strong><p>${esc(sp.fallback_used)}</p>`));
  if (!sp?.error && !sp?.fallback_used) {
    wb.append(el('p', 'sx-clean', sp ? 'Nothing flagged.' : 'Did not run.'));
  }
  (sp?.notes || []).forEach(nt => wb.append(el('p', 'sx-dim',
    `<span style="font-size:12px">${esc(nt)}</span>`)));
  body.append(wb);

  a.append(body);
  return a;
}

/* ──────────────────────────── video stage ─────────────────────────── */

function videoStage() {
  const sec = el('section', 'sx-screen');
  const n = S.notes;
  const v = String(S.status?.verdict || '');
  const inner = el('div', 'sx-screen-in');

  const player = el('div', 'sx-player');
  if (n?.file) {
    player.innerHTML = `<video controls preload="metadata"
      poster="/api/jobs/${encodeURIComponent(S.jobId)}/file/${encodeURIComponent(n.thumbnail || '')}"
      src="/api/jobs/${encodeURIComponent(S.jobId)}/download"></video>`;
  } else {
    player.innerHTML = `<div class="sx-player-empty">No delivered file for this run</div>`;
  }
  inner.append(player);

  const strip = el('dl', 'sx-filmstrip');
  const cell = (k, val, cls = '', small = false) =>
    `<div class="sx-fs ${cls}"><dt>${esc(k)}</dt>
       <dd class="${small ? 'sx-small' : ''}">${val}</dd></div>`;
  strip.innerHTML =
    cell('Title', n?.file ? esc(n.file) : NA(''), '', true)
    + cell('Duration', n?.duration != null
        ? `${Number(n.duration).toFixed(1)}s` : NA(''))
    + cell('Resolution', n?.width ? `${n.height}p` : NA(''))
    + cell('Audio', n?.audio_codec ? `✓ ${esc(n.audio_codec)}` : NA('none'),
        n?.audio_codec ? 'is-ok' : '')
    + cell('Critic', v ? `${v === 'PASS' ? '✓ ' : ''}${esc(v)}` : NA(''),
        v === 'PASS' ? 'is-ok' : v ? 'is-warn' : '');
  inner.append(strip);
  sec.append(inner);

  // thumbnail + packaging sit after the screening, not beside the player
  const after = el('div', 'sx-after');
  const tw = el('div', 'sx-thumbwrap');
  tw.innerHTML = `<h4>Thumbnail</h4>${n?.thumbnail
    ? `<img alt="Chosen thumbnail"
        src="/api/jobs/${encodeURIComponent(S.jobId)}/file/${encodeURIComponent(n.thumbnail)}">`
    : `<p class="sx-dim">No thumbnail recorded.</p>`}`;
  after.append(tw);

  const pack = el('div', 'sx-pack');
  const frames = n?.frames != null
    ? `${n.frames} frames${n.frames_ok ? ', verified' : ''}` : 'Unavailable';
  pack.innerHTML = `<h4>Specification</h4>
    <dl class="sx-dl">
      <dt>Container</dt><dd>${n?.file ? esc(n.file.split('.').pop()) : NA('')}</dd>
      <dt>Video</dt><dd>${n?.video_codec ? esc(n.video_codec) : NA('')}</dd>
      <dt>Frames</dt><dd>${esc(frames)}</dd>
      <dt>Canvas</dt><dd>${S.job?.analysis?.canvas
        ? S.job.analysis.canvas.join('×') : NA('')}</dd>
    </dl>`;
  after.append(pack);
  sec.append(after);
  return sec;
}

/* ───────────────────────────── timeline ───────────────────────────── */

function timeline() {
  const sec = el('section', 'sx-sec');
  const an = S.job?.analysis;
  const segs = Array.isArray(an?.segments) ? an.segments : [];
  const total = Number(an?.total_duration) || 0;
  sec.append(sectionHead('Timeline',
    total ? `Assembled by the agents · ${total.toFixed(2)}s`
          : 'No assembled timeline for this run'));
  if (!segs.length || !total) {
    sec.append(el('div', 'sx-empty', '<p>No segment data returned for this run.</p>'));
    return sec;
  }

  const tl = el('div', 'sx-tl');
  tl.append(el('div', 'sx-tl-bar',
    `<span><b style="color:var(--ink-2)">${segs.length}</b> segments</span>
     <span>·</span><span>${(an.transitions || []).length} transition mark(s)</span>
     <span>·</span><span>${an.canvas ? an.canvas.join('×') : ''}</span>`));

  const ruler = el('div', 'sx-tl-ruler');
  const ticks = el('div', 'sx-tl-ticks');
  const step = total > 30 ? 10 : 5;
  for (let s = 0; s <= total; s += step) {
    const sp = el('span', null, `${s}s`);
    sp.style.left = `${(s / total) * 100}%`;
    ticks.append(sp);
  }
  ruler.append(el('div', null, ''), ticks);
  tl.append(ruler);

  const rows = el('div', 'sx-tl-rows');
  const track = (label, build) => {
    const r = el('div', 'sx-tl-row');
    r.append(el('div', 'sx-tl-label', label));
    const t = el('div', 'sx-tl-track');
    build(t);
    r.append(t);
    rows.append(r);
  };

  // VIDEO — real in/out points from analysis.segments
  let acc = 0;
  const placed = segs.map(s => {
    const d = Number(s.duration) || 0;
    const o = { start: acc, dur: d, clip: s.clip, i: s.index };
    acc += d; return o;
  });
  const span = Math.max(acc, total);
  track('Video', t => placed.forEach(p => {
    const c = el('div', 'sx-tl-clip v',
      `${p.i}. ${esc(String(p.clip).replace(/\.[^.]+$/, '').slice(0, 26))}`);
    c.style.left = `${(p.start / span) * 100}%`;
    c.style.width = `calc(${(p.dur / span) * 100}% - 2px)`;
    c.title = `${p.clip} · ${p.dur.toFixed(2)}s`;
    t.append(c);
  }));

  // TRANSITIONS — real marks, real names
  const names = Array.isArray(S.job?.edit_plan?.transitions)
    ? S.job.edit_plan.transitions : [];
  track('Transitions', t => {
    const marks = an.transitions || [];
    if (!marks.length) { t.append(el('div', 'sx-tl-empty', 'No cuts')); return; }
    marks.forEach((m, i) => {
      const k = el('div', 'sx-tl-mark');
      k.style.left = `${(Number(m) / span) * 100}%`;
      k.dataset.l = names[i % Math.max(names.length, 1)] || 'cut';
      t.append(k);
    });
  });

  // MUSIC — mood comes from the render span's music_stage tool call
  const rsp = spanOf('render');
  const mus = (rsp?.tool_calls || []).find(x => x.tool === 'extras.music_stage');
  track('Music', t => {
    if (!mus || mus.music_source === 'none') {
      t.append(el('div', 'sx-tl-empty', 'No score for this run')); return;
    }
    const c = el('div', 'sx-tl-clip m', `${esc(mus.mood || 'score')} · full length`);
    c.style.cssText = 'left:0;width:calc(100% - 2px)';
    t.append(c);
  });

  // STORY / VOICEOVER / CAPTIONS — honest about what the agents produced
  const emptyWhy = node => {
    const sp = spanOf(node);
    if (!sp) return 'Did not run';
    if (sp.fallback_used) return String(sp.fallback_used).split('(')[0].trim();
    return 'Nothing recorded';
  };
  // Story and Voiceover used to print `emptyWhy` unconditionally - no code
  // ever drew what they produced. A run whose SIX beats shaped the whole
  // cut showed "Nothing recorded" under Story, which reads as "this agent
  // did nothing" when it had in fact decided the order of the video.
  const beats = S.artifacts.Story?.payload?.beats || [];
  track('Story', t => {
    if (!beats.length) {
      t.append(el('div', 'sx-tl-empty', esc(emptyWhy('story')))); return;
    }
    // Beats are an ORDER, not timings - the story is written before the
    // cut exists. They are laid out evenly and labelled by role, which is
    // what the artifact actually says; inventing a start for each one
    // would put a time on screen that nothing measured.
    beats.forEach((b, i) => {
      const c = el('div', 'sx-tl-clip s',
        `${esc(b.role || 'beat')}${b.narrative ? ' · ' + esc(String(b.narrative).slice(0, 40)) : ''}`);
      c.style.cssText =
        `left:${(i / beats.length) * 100}%;width:calc(${100 / beats.length}% - 2px)`;
      c.title = `Beat ${b.index}: ${b.role || ''} — ${b.narrative || ''}`;
      t.append(c);
    });
  });
  const vlines = (S.artifacts.Voiceover?.payload?.spoken_segments || [])
    .filter(x => x && x.spoken);
  track('Voiceover', t => {
    if (!vlines.length) {
      // "Nothing recorded" was true and useless. An agent that wrote no
      // narration because none was ASKED for has not failed at anything,
      // and saying so is the difference between a gap and a mistake.
      const asked = S.status?.settings?.voiceover;
      t.append(el('div', 'sx-tl-empty', asked === 'none'
        ? 'No narration asked for'
        : esc(emptyWhy('voiceover'))));
      return;
    }
    t.append(el('div', 'sx-tl-empty',
      `${vlines.length} line(s) written, over beat(s) `
      + vlines.map(x => x.beat_index).join(', ')));
  });
  track('Captions', t => {
    const caps = (S.job?.captions || []).length;
    t.append(el('div', 'sx-tl-empty',
      caps ? `${caps} caption line(s)` : esc(emptyWhy('caption'))));
  });

  tl.append(rows);
  sec.append(tl);
  return sec;
}

/* ────────────────────────────── critic ────────────────────────────── */

function criticSection() {
  const sec = el('section', 'sx-sec');
  sec.append(sectionHead('Quality control',
    'Measured on the rendered file, not on the plan'));
  const sp = spanOf('critic');
  const calls = (sp && sp.tool_calls) || [];
  const find = t => calls.filter(c => c.tool === t).pop();
  const n = S.notes;
  const wrap = el('div', 'sx-qa');

  // ── the score, as a dial ──
  const score = Number(S.status?.score);
  const v = String(S.status?.verdict || '');
  const pct = isFinite(score) ? Math.max(0, Math.min(100, score)) : 0;
  const C = 2 * Math.PI * 76;
  const side = el('div', 'sx-score');
  side.innerHTML = `
    <div class="sx-dial ${v.toLowerCase()}">
      <svg viewBox="0 0 168 168" aria-hidden="true">
        <circle class="trk" cx="84" cy="84" r="76"></circle>
        <circle class="val" cx="84" cy="84" r="76"
          stroke-dasharray="${C.toFixed(1)}"
          stroke-dashoffset="${(C * (1 - pct / 100)).toFixed(1)}"></circle>
      </svg>
      <div class="sx-dial-mid">
        <div>
          <div class="sx-dial-n">${isFinite(score) ? score : '—'}</div>
          <div class="sx-dial-v">${esc(v || 'no verdict')}</div>
        </div>
      </div>
    </div>
    <div class="sx-loop">${(S.status?.attempts > 1)
      ? 'Render <b>→</b> Critic <b>→</b> Revision <b>→</b> Critic'
      : 'Passed on the first render'}</div>`;
  wrap.append(side);

  // ── the checks, as figures on rules ──
  const checks = el('div', 'sx-checks');
  // `ok` may be true, false, or null. null means the critic does not gate on
  // this - it is reported, not judged. Showing a red cross for something the
  // critic never failed would contradict the verdict sitting next to it.
  const add = (name, ok, val) => {
    const cls = ok === null ? 'info' : ok ? 'pass' : 'fail';
    const mark = ok === null ? '·' : ok ? '✓' : '✕';
    checks.append(el('div', `sx-check ${cls}`,
      `<span class="sx-check-ico">${mark}</span>
       <span><span class="sx-check-n">${esc(name)}</span>
         <span class="sx-check-v">${esc(val)}</span></span>`));
  };

  if (!calls.length) {
    checks.append(el('p', 'sx-dim', 'No critic trace for this run.'));
  } else {
    if (find('verify.decode_check')) add('Decode', true, 'no errors');
    if (n?.frames != null) add('Frame integrity', !!n.frames_ok,
      `${n.frames} frames`);
    if (n?.width) add('Resolution', true, `${n.width}×${n.height}`);
    if (find('verify.black_frames')) add('Black frames', true, 'none detected');
    const loud = find('verify.loudness');
    if (loud && loud.lufs !== undefined) {
      add('Audio', !loud.silent, `${loud.lufs} LUFS`);
    } else if (n?.audio_codec) add('Audio', true, esc(n.audio_codec));
    const hook = find('verify.score_hook');
    if (hook) add('Opening hook', hook.score >= 65, `${hook.score} / 100`);
    // captions and transitions are reported from what the run actually made
    const caps = (S.job?.captions || []).length;
    add('Captions', caps > 0 ? true : null,
      caps ? `${caps} line(s)` : 'none written');
    const tr = (S.job?.analysis?.transitions || []).length;
    add('Transitions', tr > 0 ? true : null,
      tr ? `${tr} join(s)` : 'single segment');
  }
  wrap.append(checks);
  sec.append(wrap);
  return sec;
}

/* ──────────────────── packaging / performance / learning ──────────── */

function packagingSection() {
  const sec = el('section', 'sx-sec');
  sec.append(sectionHead('Packaging & outcome',
    'Everything after the cut was approved'));
  const cols = el('div', 'sx-cols');

  // publishing
  const pub = spanOf('publishing');
  const pubCall = (pub?.tool_calls || []).find(c => c.tool === 'publishing.scan_platform');
  const perfRead = (spanOf('performance_analysis')?.tool_calls || [])
    .find(c => c.tool === 'performance.read_publication');
  const pstate = perfRead?.status || (pub ? 'needs_review' : null);
  const cls = pstate === 'published' ? 'ready' : pstate === 'needs_review'
    ? 'review' : pstate ? 'blocked' : 'none';
  cols.append(el('div', 'sx-panel', `<h4>Publishing</h4>
    <span class="sx-state ${cls}">${esc(pstate ? pstate.replace('_', ' ') : 'unavailable')}</span>
    <dl class="sx-dl" style="margin-top:14px">
      <dt>Title</dt><dd>${spanOf('title')?.fallback_used
        ? NA('no model configured') : (S.job?.output?.file ? esc(S.job.output.file) : NA(''))}</dd>
      <dt>Thumbnail</dt><dd>${S.notes?.thumbnail ? esc(S.notes.thumbnail) : NA('')}</dd>
      <dt>Provider</dt><dd>${pubCall
        ? `${pubCall.upload ? 'configured' : 'none — the operator publishes'}` : NA('')}</dd>
    </dl>`));

  // performance
  const pa = spanOf('performance_analysis');
  cols.append(el('div', 'sx-panel', `<h4>Performance</h4>
    ${pa?.fallback_used
      ? `<div class="sx-unavail"><strong>Analytics unavailable</strong>
         <p>${esc(pa.fallback_used)}</p></div>`
      : '<p class="sx-dim">No performance span for this run.</p>'}`));

  // learning
  const lg = (spanOf('learning')?.tool_calls || [])
    .find(c => c.tool === 'learning.evaluate_gates');
  const hist = (spanOf('learning')?.tool_calls || [])
    .find(c => c.tool === 'learning.read_history');
  const lp = el('div', 'sx-panel');
  lp.innerHTML = '<h4>Learning</h4>';
  if (lg) {
    // `not_evaluated` means the gate never ran - NOT that it passed. Marking
    // those with a tick would claim a result the backend did not produce.
    const failed = lg.failed || [];
    const skipped = lg.not_evaluated || [];
    const passed = (lg.passed || []).filter(
      g => !failed.includes(g) && !skipped.includes(g));
    const gates = el('div', 'sx-gates');
    const add = (g, cls, mark) => gates.append(el('div', `sx-gate ${cls}`,
      `<span>${esc(String(g).replace(/_/g, ' '))}</span>
       <span class="sx-gate-m">${mark}</span>`));
    failed.forEach(g => add(g, 'no', '✕'));
    passed.forEach(g => add(g, 'yes', '✓'));
    skipped.forEach(g => add(g, 'skip', '—'));
    lp.append(el('p', 'sx-dim', hist
      ? `${hist.usable} usable sample(s) of ${hist.jobs} job(s).` : ''));
    lp.append(gates);
    if (skipped.length) lp.append(el('p', 'sx-dim',
      `<span style="font-size:11.5px">— not evaluated: the earlier gates
       failed first, so these were never reached.</span>`));
  } else lp.append(el('p', 'sx-dim', 'No learning span for this run.'));
  cols.append(lp);

  sec.append(cols);
  return sec;
}

/* ═══════════════════════════ view: agents ══════════════════════════ */

function viewAgents() {
  const f = document.createDocumentFragment();
  const h = S.health;
  f.append(topBar('Agents', h ? `${h.agent_count} on the team` : ''));

  const hero = el('section', 'sx-hero');
  hero.style.paddingTop = 'clamp(28px, 4vw, 56px)';
  hero.innerHTML = `
    <div class="sx-hero-in">
      <p class="sx-kicker">The roster</p>
      <h2>Your AI production team</h2>
      <p class="sx-hero-lede">${h
        ? `${h.agent_count} specialists. ${h.reasoning_agents} reason over the
           footage; ${h.agent_count - h.reasoning_agents} measure it. Each owns
           exactly one decision and one artifact.`
        : ''}</p>
    </div>`;
  f.append(hero);

  if (!h) { f.append(el('div', 'sx-skel')); return f; }

  CRAFT.forEach(([title, note, nodes]) => {
    const g = el('section', 'sx-dir-group');
    g.append(sectionHead(title, note));
    const grid = el('div', 'sx-dir-grid');
    nodes.forEach(node => {
      const a = metaFor(node);
      if (!a) return;
      const sp = spanOf(node);
      const card = el('button', 'sx-dir-card');
      card.innerHTML = `
        <div class="sx-dir-top">
          <span class="sx-glyph" style="color:${a.agency === 'reasoning'
            ? 'var(--violet-2)' : a.agency === 'hybrid' ? 'var(--cyan)'
            : 'var(--ink-3)'}">${GLYPH[a.agency]}</span>
          <strong>${esc(LABEL[node] || node)}</strong>
        </div>
        <p>${esc(ROLE[node] || '')}</p>
        <div class="sx-dir-foot">
          <code>${esc(a.produces)}</code>
          <span style="margin-left:auto">${sp
            ? `${STATE_WORD[stateOf(node)]} · ${dur(wallOf(sp))}`
            : 'no run selected'}</span>
        </div>`;
      card.addEventListener('click', () => openProduction(S.jobId, node));
      grid.append(card);
    });
    g.append(grid);
    f.append(g);
  });
  return f;
}

/* ════════════════════════════ view: runs ═══════════════════════════ */

function viewRuns() {
  const f = document.createDocumentFragment();
  f.append(topBar('Runs', `${S.jobs.length} recorded`));
  if (!S.jobs.length) {
    f.append(el('div', 'sx-empty', '<h4>No runs</h4>'));
    return f;
  }
  const list = el('div', 'sx-prods');
  S.jobs.forEach(j => list.append(prodRow(j)));
  f.append(list);
  return f;
}

/* ═══════════════════════ view: system health ═══════════════════════ */

function viewHealth() {
  const f = document.createDocumentFragment();
  f.append(topBar('System Health', 'Reported by the running backend'));
  const h = S.health;
  if (!h) { f.append(el('div', 'sx-skel')); return f; }
  const g = el('div', 'sx-health');
  const row = (name, state, detail) => g.append(el('div', 'sx-hrow',
    `<span class="sx-hname">${esc(name)}</span>
     <span class="sx-hstate ${state[0]}"><i></i>${esc(state[1])}</span>
     <span class="sx-hdetail">${detail}</span>`));

  row('API', ['ok', 'Online'], 'core routes responding');
  row('Agent registry', ['ok', 'Online'],
    `${h.agent_count} agents discovered by package walk`);
  row('LLM', h.reasoning_enabled ? ['ok', 'Configured'] : ['off', 'Not configured'],
    h.reasoning_enabled ? `${esc(h.provider || '')} · ${esc(h.model || '')}`
      : esc(h.why_no_model || ''));
  row('Research provider', ['off', 'Not configured'],
    'AT_SEARCH_API_KEY is unset — the Research agent degrades its report');
  row('Artifact store', ['ok', 'Online'],
    `${h.jobs_in_store} job(s) · <code class="sx-mono">${esc(h.shared_store || '')}</code>`);
  row('Jobs on disk', ['ok', 'Online'],
    (h.agent_jobs_on_disk || []).join(', ') || 'none');
  row('Schema', ['ok', `v${h.schema_version}`], 'artifact envelope version');
  g.append(el('div', 'sx-hrow',
    `<span class="sx-hname">FFmpeg · Scheduler · Analytics</span>
     <span class="sx-hstate off"><i></i>Not reported</span>
     <span class="sx-hdetail">no health route exposes these; nothing is assumed</span>`));
  f.append(g);
  return f;
}

/* ═════════════════════════════ plumbing ════════════════════════════ */

const VIEWS = { create: viewCreate, overview: viewOverview,
  production: viewProduction, agents: viewAgents, runs: viewRuns,
  health: viewHealth };

function render() {
  if (!root) return;
  const y = window.scrollY;
  root.textContent = '';
  if (S.error) {
    root.append(el('div', 'sx-err',
      `<h4>Cannot reach the backend</h4><p class="sx-dim">${esc(S.error)}</p>`));
    return;
  }
  if (!S.loaded) {
    root.append(el('div', 'sx-skel'), el('div', 'sx-skel'), el('div', 'sx-skel'));
    return;
  }
  root.append((VIEWS[S.view] || viewOverview)());
  buildSidebar();
  window.scrollTo(0, y);
  // the canvas is measured from the live tree, so edges are drawn after the
  // append and again on the next frame once fonts/images have settled
  paintEdges();
  requestAnimationFrame(paintEdges);
}

/* The create wizard and the studio views take turns owning the screen.
   Any studio navigation has to reclaim it, otherwise a sidebar click while
   the wizard is open re-renders a view nobody can see. */
function enterStudio() {
  const app = document.querySelector('.app');
  const view = document.querySelector('#sxView');
  if (app) app.hidden = true;
  if (view) view.classList.add('on');
}

/* ──────────────────────── where you are, in the URL ────────────────────
 * There was no routing at all: every view lived only in `S`, so a refresh -
 * or the back button, or reopening the tab - dropped a running production
 * and landed back on Create with no way to find the job again. A build
 * takes minutes; losing it to an accidental F5 is not acceptable.
 *
 * The hash is written as views change and read back on boot, so a refresh
 * returns to the same place, back/forward work, and the address bar is a
 * link you can send someone.
 */
let _settingHash = false;

// `push` adds a history entry, so the back button walks the views the
// person actually visited. The boot sync REPLACES instead, because landing
// on a page should not leave an entry that goes nowhere.
function writeHash(push = true) {
  const want = (S.view === 'production' && S.jobId)
    ? `#/production/${encodeURIComponent(S.jobId)}`
    : `#/${S.view || 'create'}`;
  if (location.hash === want) return;
  _settingHash = true;                       // our own write, not a click
  try {
    history[push ? 'pushState' : 'replaceState'](null, '', want);
  } catch {
    location.hash = want;
  }
  _settingHash = false;
}

function readHash() {
  const m = /^#\/([a-z_]+)(?:\/([A-Za-z0-9_-]+))?$/.exec(location.hash || '');
  return m ? { view: m[1], id: m[2] || null } : null;
}

async function applyHash() {
  const r = readHash();
  if (!r) return false;
  if (r.view === 'production' && r.id) { await openProduction(r.id); return true; }
  if (r.view && r.view !== 'production') { go(r.view); return true; }
  return false;
}

function go(view) {
  S.view = view;
  S.armedDelete = null;   // never stay armed across views
  enterStudio();
  writeHash();
  render();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Read this job's current state. ALWAYS fetches.
//
// This used to live inside `openProduction` behind `if (id !== S.jobId)`,
// and `poll()` re-entered `openProduction(S.jobId)` - the SAME id - so the
// guard was false every time and the fetch never ran again. The production
// screen froze on whatever it loaded when first opened: for a new job that
// is the 404 "starting" state, so it sat on "Starting the production…" with
// "0 agents ran · 0ms · 0 tool calls" for the whole build, while the graph
// ran all eighteen agents and finished the video behind it.
async function refreshProduction(id) {
  try {
    // tryGet, not get: for the first few seconds of a new production the
    // status route 404s, because the job has not written an artifact yet
    // and so does not look like an agent job. That is a normal starting
    // state, not a failure, and it must not replace the page.
    const [st, tr] = await Promise.all([
      tryGet(`${V2}/jobs/${encodeURIComponent(id)}/status`),
      tryGet(`${V2}/jobs/${encodeURIComponent(id)}/trace`),
    ]);
    if (id !== S.jobId) return;        // the person opened another job mid-flight
    S.status = st || { job_id: id, verdict: null, totals: {}, starting: true };
    S.spans = ((tr && tr.trace) || []).slice()
      .sort((a, b) => (a.seq || 0) - (b.seq || 0));
    const [plan, notes, job] = await Promise.all([
      tryGet(`/api/jobs/${encodeURIComponent(id)}/plan`),
      tryGet(`/api/jobs/${encodeURIComponent(id)}/notes`),
      tryGet(`/api/jobs/${encodeURIComponent(id)}`),
    ]);
    if (id !== S.jobId) return;
    S.decisions = Array.isArray(plan?.decisions) ? plan.decisions : [];
    S.notes = (notes && !notes.detail) ? notes : null;
    S.job = job && !job.detail ? job : null;

    // What the narrative agents actually wrote. `/status` reports only
    // artifact IDs, so without this the timeline could say an agent ran
    // and not one word of what it decided.
    const want = ['Story', 'Voiceover'];
    const got = await Promise.all(want.map(k =>
      tryGet(`${V2}/jobs/${encodeURIComponent(id)}/artifacts/${k}`)));
    if (id !== S.jobId) return;
    S.artifacts = Object.fromEntries(
      want.map((k, i) => [k, got[i] && !got[i].detail ? got[i] : null]));

    // the recorded upload and whatever the analytics export says of it
    await refreshPublished(id);
  } catch (e) {
    // a transient read must not take the workspace down
    S.loadNote = String(e.message || e);
  }
}

async function openProduction(id, select) {
  S.view = 'production';
  enterStudio();
  if (id && id !== S.jobId) {
    S.jobId = id; S.spans = []; S.status = null; S.decisions = [];
    S.settled = 0;              // the settle counter belongs to one job
    S.notes = null; S.job = null; S.selected = null; S.artifacts = {};
    S.published = null;
    render();                   // show the shell while the first read runs
  }
  writeHash();                 // survive a refresh from here on
  if (id) await refreshProduction(id);
  if (select) S.selected = select;
  render();
  poll();
}

// A verdict is NOT the end of the run. The critic writes it, then thumbnail,
// title, publishing, performance and learning still run, and only when the
// graph returns does the finished file get published into the legacy record
// that /download reads. Stopping at the verdict left the result screen
// showing a <video> whose source answered 409, with no further poll to
// repair it: the person watched a dead player until they reloaded. Observed
// end to end - verdict at 150s, file servable ~30s later.
const SETTLE_POLLS = 40;        // ~100s, then show what we have regardless

function finishing() {
  if (!S.status?.verdict || S.status?.failed || S.job?.output) return false;
  // A REVISE verdict is mid-run, not the end: the director re-plans and the
  // renderer runs again, which can take minutes. While a node is still
  // working the wait is not "settling" and must not burn the budget below.
  if (S.status.current_node) return true;
  return (S.settled || 0) < SETTLE_POLLS;
}

function poll() {
  clearTimeout(S.timer);
  if (!S.status || S.status.failed) return;       // terminal: stopped
  // A refine re-runs the graph over a job that already has a verdict, so
  // the verdict alone is not an ending. Polling stopped on the stale one
  // the instant a refine began, and the screen then never moved again.
  if (!S.status.producing && S.status.verdict && !finishing()) {
    return;                                       // finished AND servable
  }
  if (!S.status.producing && S.status.verdict && !S.status.current_node) {
    S.settled = (S.settled || 0) + 1;             // only genuine idle counts
  }
  const id = S.jobId;
  S.timer = setTimeout(async () => {
    if (S.jobId !== id || S.view !== 'production') return;  // moved on
    await refreshProduction(id);
    if (S.jobId !== id) return;
    render();
    poll();
  }, 2500);
}

async function load() {
  try {
    const [h, j] = await Promise.all([get(`${V2}/health`), get(`${V2}/jobs`)]);
    S.health = h; S.jobs = j.jobs || [];
  } catch (e) { S.error = String(e.message || e); }
  S.loaded = true;
  render();
  loadCreateOptions();
  loadTracks();
}

/* Per-job traces for the dashboard ticks. Capped, parallel, and failures
   are silent: a job whose trace cannot be read simply keeps grey ticks. */
async function loadTracks() {
  const todo = S.jobs.slice(0, 8).filter(j => !S.tracks[j.id]);
  if (!todo.length) return;
  await Promise.all(todo.map(async j => {
    const tr = await tryGet(`${V2}/jobs/${encodeURIComponent(j.id)}/trace`);
    const rows = (tr?.trace || []).slice().sort((a, b) => (a.seq || 0) - (b.seq || 0));
    if (!rows.length) return;
    S.tracks[j.id] = rows.map(sp => sp.status === 'error' ? 'error'
      : sp.fallback_used ? 'degraded' : 'ok');
    const grab = rows.find(sp => sp.node === 'thumbnail')?.tool_calls
      ?.find(t => t.tool === 'thumbnail.frame_grab');
    if (grab?.image) S.thumbs[j.id] = grab.image;
  }));
  if (S.view === 'overview' || S.view === 'runs') render();
}

let ro = null;
function watchResize() {
  if (ro || !window.ResizeObserver) return;
  let t = null;
  ro = new ResizeObserver(() => {
    clearTimeout(t);
    t = setTimeout(() => {
      const c = root?.querySelector('.sx-canvas');
      const s = c?.querySelector('.sx-edges');
      if (c && s) drawEdges(c, s);
    }, 120);
  });
  ro.observe(document.body);
}

/* ─────────────────────────────── mount ────────────────────────────── */

export function initStudio({ viewSel = '#sxView', sideSel = '#sxSide',
                             onCreateClick } = {}) {
  root = document.querySelector(viewSel);
  sideEl = document.querySelector(sideSel);
  if (!root || !sideEl) return null;
  onCreate = onCreateClick;
  buildSidebar();
  watchResize();
  // back / forward, and a hash someone pasted in
  // pushState does not raise `hashchange`, so the back button arrives as
  // `popstate`. Both are listened for: one covers our own navigation, the
  // other a hash someone typed or pasted.
  const onNav = () => { if (!_settingHash) applyHash(); };
  window.addEventListener('popstate', onNav);
  window.addEventListener('hashchange', onNav);

  let started = false;
  return {
    async show(view = 'create') {
      S.view = view;
      if (!started) {
        started = true;
        render();
        await load();
        // A refresh used to land on Create and lose a running build. If the
        // address bar still names one, go back to it.
        if (!(await applyHash())) writeHash(false);
      } else render();
    },
    hide() { clearTimeout(S.timer); },
    go,
  };
}

export { ROWS, EDGES, LABEL };
