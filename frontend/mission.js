// Mission Control - which agent is working now, and what it is doing.
//
// Reads /api/v2/jobs/<id>/mission (a snapshot) and follows
// /api/v2/jobs/<id>/mission/stream (server-sent events). It draws what the
// trace says and nothing else: a planned worker is shown as planned, never as
// working, and with no run selected it shows the structure and says so.
//
// Every string from the server goes in as a text node, never as markup.

const API = "/api/v2";
const $ = (sel, root = document) => root.querySelector(sel);

const LABEL = {
  working: "Working", done: "Done", failed: "Failed", waiting: "Waiting",
  planned: "Planned", interrupted: "Interrupted", incomplete: "Incomplete",
  not_run: "Did not run", skipped: "Skipped", pending: "Pending",
  pass: "Passed", warn: "Warnings", fail: "Failed", running: "Running",
  stopped: "Stopped", cancelled: "Cancelled", queued: "Queued",
};
const MARK = { pass: "✓", warn: "!", fail: "✗", pending: "·" };

const state = {
  jobId: null, snap: null, selected: null, follow: true, scope: "all",
  es: null, poll: null, receivedAt: 0,
  data: null, dataBusy: false, dataError: null, agent: null,
};

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

const badge = (st) => h("span", { class: `mc-badge s-${st}` }, LABEL[st] || st);
const dot = (st) => h("span", { class: `mc-dot s-${st}`, title: LABEL[st] || st });

function dur(s) {
  s = Math.max(0, Math.round(s || 0));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}
const clock = (t) => new Date(t * 1000).toLocaleTimeString([], { hour12: false });
const live = (w) => (w.duration_s || 0) + (Date.now() / 1000 - state.receivedAt);

// ── a snapshot built from the static structure, when no run is selected ──
function fromHierarchy(hier) {
  return {
    job_id: null, status: "queued", chief: { ...hier.chief, state: "queued" },
    now: null, feed: [], generated_at: Date.now() / 1000,
    totals: { built_workers: hier.counts.built_workers, done_workers: 0,
              planned_workers: hier.counts.planned_workers, tool_calls: 0,
              tokens: 0, usd: 0, rounds: 0, wall_s: 0 },
    teams: hier.teams.map((t) => ({
      id: t.id, label: t.label, phase: t.phase, purpose: t.purpose,
      state: t.workers.some((w) => w.built) ? "waiting" : "planned",
      degraded: false,
      lead: { label: t.lead.label, state: "waiting", summary: "no run selected" },
      workers: t.workers.map((w) => ({ ...w, state: w.built ? "waiting" : "planned" })),
      checker: { label: t.checker.label, state: "pending", checks: [] },
    })),
  };
}

// ── render ──
function renderTop(s) {
  $("#mcJobState").replaceChildren(badge(s.status));
  const t = s.totals;
  $("#mcTotals").replaceChildren(
    h("span", {}, "agents ", h("b", {}, `${t.done_workers}/${t.built_workers}`)),
    h("span", {}, "planned ", h("b", {}, t.planned_workers)),
    h("span", {}, "tools ", h("b", {}, t.tool_calls)),
    h("span", {}, "tokens ", h("b", {}, t.tokens.toLocaleString())),
    h("span", {}, "rounds ", h("b", {}, t.rounds)),
    h("span", {}, "time ", h("b", { id: "mcWall" }, dur(t.wall_s))),
  );
}

function renderChief(s) {
  const now = s.now;
  $("#mcChief").replaceChildren(
    dot(s.chief.state === "running" ? "working" : (s.status === "done" ? "done" : "waiting")),
    h("div", {},
      h("h2", {}, "Chief"),
      h("p", {}, s.chief.does)),
    h("div", { class: "mc-now", "aria-live": "polite" },
      now ? [h("b", {}, `${now.team_label} › ${now.worker}`), " — ", now.text]
          : h("span", { class: "mc-idle" },
              { done: "Run finished.", failed: "Run failed.", stopped:
                "Run stopped before it finished.", cancelled: "Run cancelled.",
                queued: "No run selected, or it has not started." }[s.status]
              || "Between agents…")),
  );
}

// ── the map: every team as a card, the way the work and the lessons move ──
//
// Drawn like the operations map the system was designed on: orange arrows
// carry the work forward, purple dashed arrows carry lessons back (the
// Quality team's re-cut, Learning's findings), the blue dotted arrow is data
// arriving from the warehouse, and a padlock is where a person approves.
// Cards light up from the live trace; nothing on the map is decoration.

const SVGNS = "http://www.w3.org/2000/svg";
function sv(tag, attrs, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
  for (const kid of kids.flat()) {
    if (kid == null) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

const CW = 168, CH = 128, GX = 34, X0 = 10, ROWS = [34, 236];
const PLACE = {                           // [column, row]
  insight: [0, 0], footage: [1, 0], creative: [2, 0], sound: [3, 0], edit: [4, 0],
  data: [0, 1], learning: [1, 1], publish: [2, 1], quality: [3, 1], render: [4, 1],
};
const GATE = {
  creative: "A person approves the story before cutting (planned)",
  quality: "A person approves the final cut (planned)",
  publish: "The manager approves going public",
};
const LETTER = { insight: "A", footage: "B", creative: "C", sound: "D", edit: "E",
                 render: "F", quality: "G", publish: "H", learning: "I", data: "J" };
const xy = (id) => { const [c, r] = PLACE[id]; return [X0 + c * (CW + GX), ROWS[r]]; };

function dataCard() {
  const d = state.data;
  const steps = d && d.steps || [];
  const agents = (d && d.agents) || ["Source Loader", "Cleaner", "Enricher", "Analytics Builder"];
  const workers = agents.map((a) => {
    const st = steps.find((x) => x.agent === a);
    return { label: a, state: !st ? "waiting" : st.ok ? "done" : "failed" };
  });
  const rep = d && d.report && d.report.summary;
  const failed = steps.some((x) => !x.ok);
  return {
    id: "data", label: "Data Layers", phase: "Warehouse → 4 layers",
    state: state.dataBusy ? "working" : !steps.length ? "waiting" : failed ? "failed" : "done",
    workers,
    line: state.dataBusy ? "reading the warehouse…"
      : rep ? `${rep.videos} videos · last ${clock(steps[steps.length - 1].ended_at)}`
      : "not run yet",
  };
}

function cardLine(t) {
  if (t.line) return t.line;
  const w = t.workers.find((x) => x.state === "working");
  if (w) return `now: ${w.label}`;
  const built = t.workers.filter((x) => x.built).length;
  if (!built) return "planned";
  return `${LABEL[t.state] || t.state}${t.degraded ? " · warnings" : ""}`;
}

function card(t) {
  const [x, y] = xy(t.id);
  const sel = t.id === state.selected;
  const built = t.workers.filter((w) => w.built !== false).length;
  const pick = () => {
    state.selected = t.id; state.agent = null; state.follow = false; syncFollow(); draw();
    const d = document.getElementById("mcDrill");
    if (d) d.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
      ? "auto" : "smooth", block: "nearest" });
  };
  const g = sv("g", {
    class: `mc-card s-${t.state}${sel ? " is-sel" : ""}`, tabindex: "0", role: "button",
    "aria-pressed": sel ? "true" : "false",
    "aria-label": `${t.label}: ${LABEL[t.state] || t.state}. ${t.phase || ""}`,
  });
  g.addEventListener("click", pick);
  g.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); }
  });
  g.append(
    sv("rect", { class: "mc-card-bg", x, y, width: CW, height: CH, rx: 9 }),
    sv("rect", { class: "mc-card-stripe", x: x + 1, y: y + 1, width: CW - 2, height: 4, rx: 2 }),
    sv("rect", { class: "mc-card-letter-bg", x: x + 10, y: y + 14, width: 22, height: 20, rx: 5 }),
    sv("text", { class: "mc-card-letter", x: x + 21, y: y + 28, "text-anchor": "middle" }, LETTER[t.id] || ""),
    sv("text", { class: "mc-card-title", x: x + 40, y: y + 29 }, t.label),
    sv("text", { class: "mc-card-ph", x: x + 10, y: y + 52 }, (t.phase || "").slice(0, 30)),
    sv("text", { class: "mc-card-chip", x: x + 10, y: y + 74 },
      `${built}/${t.workers.length} agents${t.id === "quality" ? " · revise ≤ 2" : ""}`),
  );
  t.workers.forEach((w, i) => g.append(sv("circle", {
    class: `mc-card-dot s-${w.state}`, cx: x + 15 + i * 15, cy: y + 92, r: 4.5,
  }, sv("title", {}, `${w.label}: ${LABEL[w.state] || w.state}`))));
  g.append(
    sv("line", { class: "mc-card-rule", x1: x + 10, x2: x + CW - 10, y1: y + 104, y2: y + 104 }),
    sv("text", { class: "mc-card-st", x: x + 10, y: y + 119 }, cardLine(t).slice(0, 30)),
  );
  if (GATE[t.id]) {
    g.append(sv("g", { class: "mc-lock" },
      sv("title", {}, GATE[t.id]),
      sv("rect", { x: x + CW - 22, y: y + 21, width: 11, height: 8.5, rx: 1.6 }),
      sv("path", { d: `M${x + CW - 19.2},${y + 21} v-2.6 a2.7,2.7 0 0 1 5.4,0 v2.6` })));
  }
  return g;
}

function arrow(cls, d, label, lx, ly, anchor = "middle") {
  const out = [sv("path", { class: cls, d, "marker-end": `url(#mc-head-${cls.split("-")[1]})` })];
  if (label) out.push(sv("text", { class: "mc-map-lbl", x: lx, y: ly, "text-anchor": anchor }, label));
  return out;
}

function renderFlow(s) {
  const teams = [...s.teams, dataCard()];
  const R1 = ROWS[0] + CH / 2, R2 = ROWS[1] + CH / 2;
  const colX = (c) => X0 + c * (CW + GX);
  const fwd = [];
  for (let c = 0; c < 4; c++) {
    fwd.push(...arrow("mc-flow", `M${colX(c) + CW + 2},${R1} H${colX(c + 1) - 4}`));
  }
  const edX = colX(4) + CW / 2;
  fwd.push(...arrow("mc-flow", `M${edX},${ROWS[0] + CH + 2} V${ROWS[1] - 4}`, "cut", edX + 8, ROWS[0] + CH + 40, "start"));
  for (let c = 4; c > 1; c--) {
    fwd.push(...arrow("mc-flow", `M${colX(c) - 2},${R2} H${colX(c - 1) + CW + 4}`));
  }
  const back = [
    ...arrow("mc-fb", `M${colX(3) + CW - 30},${ROWS[1] - 2} L${colX(4) + 30},${ROWS[0] + CH + 4}`,
      "revise ≤ 2", colX(3) + CW + 10, ROWS[1] - 30, "middle"),
    ...arrow("mc-fb", `M${colX(1) + CW - 30},${ROWS[1] - 2} L${colX(2) + 30},${ROWS[0] + CH + 4}`,
      "lessons", colX(1) + CW + 10, ROWS[1] - 30, "middle"),
    ...arrow("mc-data", `M${colX(0) + CW + 2},${R2} H${colX(1) - 4}`, "day-7", colX(0) + CW + 17, R2 - 8),
  ];
  const svg = sv("svg", {
    class: "mc-map", viewBox: "0 0 996 376", role: "group",
    "aria-label": "The teams, in the order the work moves. Choose one to open its room.",
  },
    sv("defs", {},
      ...["flow", "fb", "data"].map((k) => sv("marker", {
        id: `mc-head-${k}`, viewBox: "0 0 10 10", refX: 9, refY: 5,
        markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse",
      }, sv("path", { d: "M0,0 L10,5 L0,10 z", class: `mc-head-${k}` })))),
    sv("text", { class: "mc-map-row", x: X0, y: 20 }, "MAKE THE VIDEO"),
    sv("text", { class: "mc-map-row", x: X0, y: ROWS[1] - 14 }, "CHECK, SHIP AND LEARN"),
    ...fwd, ...back, ...teams.map(card));
  $("#mcFlow").replaceChildren(
    h("div", { class: "mc-map-wrap" }, svg),
    h("div", { class: "mc-legend" },
      h("span", { class: "k-flow" }, "work moves on"),
      h("span", { class: "k-fb" }, "sent back / lessons"),
      h("span", { class: "k-data" }, "data from the warehouse"),
      h("span", { class: "k-lock" }, "a person approves")),
    renderDrill(s));
}

// ── the agents inside the chosen team, right under the map ──
//
// Click a team: the agents inside it appear in a row, built ones with their
// state, planned ones dashed.

const DATA_DOES = {
  "Source Loader": "Copies the channel's daily numbers from BigQuery, untouched.",
  "Cleaner": "One row per video per day; every video judged on its first seven days.",
  "Enricher": "A model tags each title's hook and topic.",
  "Analytics Builder": "One row per video, the groups, and the file Learning reads.",
  "Feature Analyst": "Measures each published video: length, opening energy, number of shots.",
  "Opening Reviewer": "Looks at each video's first seconds: finished build shown? hands in the way?",
  "Pattern Miner": "Finds lessons that still hold without the best video; a person approves each.",
  "Why Writer": "Says in a few sentences why some videos do better, citing only those lessons.",
};

function teamWorkers(s, id) {
  if (id === "data") {
    const d = state.data || {};
    return (d.agents || Object.keys(DATA_DOES)).map((a) => {
      const st = (d.steps || []).find((x) => x.agent === a);
      return { key: `data:${a}`, label: a, does: DATA_DOES[a] || "", built: true,
               state: state.dataBusy ? "working" : !st ? "waiting" : st.ok ? "done" : "failed",
               checks: st ? st.checks : [], duration_s: st ? st.ended_at - st.started_at : 0 };
    });
  }
  const t = s.teams.find((x) => x.id === id);
  return t ? t.workers : [];
}

function renderDrill(s) {
  // Only the agents of the chosen team: who is in it, and whether each has
  // run. No decisions or logs here - the team room further down has those.
  const id = state.selected;
  const team = id === "data" ? dataCard() : s.teams.find((t) => t.id === id);
  if (!team) return h("div");
  const workers = teamWorkers(s, id);
  const built = workers.filter((w) => w.built).length;
  return h("section", { class: "mc-drill", id: "mcDrill", "aria-label": `${team.label} agents` },
    h("div", { class: "mc-drill-h" },
      h("span", { class: "mc-drill-letter" }, LETTER[id] || ""),
      h("h2", {}, `${team.label} team`),
      h("span", { class: "mc-drill-count" },
        `${built} agent${built === 1 ? "" : "s"} built${workers.length > built
          ? ` · ${workers.length - built} planned` : ""}`)),
    h("div", { class: "mc-agent-row" }, workers.map((w, i) => [
      i ? h("span", { class: "mc-agent-sep", "aria-hidden": "true" }, "→") : null,
      h("button", {
        class: `mc-agent-chip s-${w.built ? w.state : "planned"}${w.key === state.agent ? " is-sel" : ""}`,
        type: "button", "aria-pressed": w.key === state.agent ? "true" : "false",
        onclick: () => { state.agent = w.key === state.agent ? null : w.key; draw(); },
      }, dot(w.built ? w.state : "planned"), w.label,
        w.built ? null : h("em", {}, "planned")),
    ]).flat()),
    aboutAgent(workers.find((w) => w.key === state.agent)));
}

// One short line about the chosen agent: what it does, and how it went.
function aboutAgent(w) {
  if (!w) return h("p", { class: "mc-about-hint" }, "Click an agent to see what it does.");
  const how = !w.built ? "Planned: not built yet."
    : [LABEL[w.state] || w.state, w.duration_s ? dur(w.duration_s) : null,
       w.fallback ? "used its fallback" : null].filter(Boolean).join(" · ");
  return h("div", { class: "mc-about" },
    h("b", {}, w.label), h("span", {}, w.does || ""), h("em", {}, how));
}

function workerRow(w) {
  const meta = [];
  if (w.built && w.state !== "waiting" && w.state !== "not_run") {
    meta.push(h("span", { "data-live": w.state === "working" ? "1" : null,
                          "data-base": w.duration_s }, dur(w.duration_s)));
    if (w.tools) meta.push(h("span", {}, `${w.tools} tool calls`));
    if (w.tokens) meta.push(h("span", {}, `${w.tokens.toLocaleString()} tokens`));
    if (w.rounds > 1) meta.push(h("span", {}, `${w.rounds} rounds`));
    if (w.tries > w.rounds) meta.push(h("span", { class: "warn" }, `${w.tries - w.rounds} retries`));
    if (w.fallback) meta.push(h("span", { class: "warn" }, `fallback: ${w.fallback}`));
    if (w.state === "working" && w.quiet_s > 60)
      meta.push(h("span", { class: "warn" }, `quiet for ${dur(w.quiet_s)}`));
    if (w.error) meta.push(h("span", { class: "bad" }, w.error));
  }
  return h("li", { class: `mc-worker s-${w.state}` },
    h("span", { class: "mc-wdot" }, dot(w.state)),
    h("div", {},
      h("h4", {}, w.label, w.essential ? " " : "",
        w.essential ? h("span", { class: "mc-badge", title:
          "The run cannot continue without this agent" }, "essential") : null),
      h("p", { class: "does" }, w.does),
      w.state === "working" && w.doing ? h("p", { class: "doing" }, w.doing) : null,
      w.state === "planned" ? h("p", { class: "meta" }, "Planned: not built yet.") : null,
      meta.length ? h("div", { class: "meta" }, meta) : null),
    h("div", { class: "right" }, badge(w.state)));
}

function renderDataRoom() {
  const d = state.data || {};
  const steps = d.steps || [];
  const rep = d.report;
  const sum = rep && rep.summary;
  const run = h("button", {
    class: "mc-btn", type: "button", disabled: state.dataBusy ? "" : null,
    onclick: runDataLayers,
  }, state.dataBusy ? "Reading the warehouse…" : "Refresh from BigQuery");
  const workers = (d.agents || []).map((a) => {
    const st = steps.find((x) => x.agent === a);
    const s = state.dataBusy ? "working" : !st ? "waiting" : st.ok ? "done" : "failed";
    return h("li", { class: `mc-worker s-${s}` },
      h("span", { class: "mc-wdot" }, dot(s)),
      h("div", {},
        h("h4", {}, a),
        h("p", { class: "does" }, {
          "Source Loader": "Copies the channel's daily numbers from the warehouse, untouched.",
          "Cleaner": "One row per video per day; every video judged on its first seven days.",
          "Enricher": "Tags each title's hook and topic, and signs who tagged it.",
          "Analytics Builder": "One row per video, the groups, and the file Learning reads.",
          "Feature Analyst": "Measures each published video: length, opening energy, number of shots.",
          "Opening Reviewer": "Looks at each video's first seconds: finished build shown? hands in the way?",
          "Pattern Miner": "Finds lessons that still hold without the best video; a person approves each.",
          "Why Writer": "Says in a few sentences why some videos do better, citing only those lessons.",
        }[a] || ""),
        st ? h("ul", { class: "mc-checks" }, st.checks.map((k) =>
          h("li", { class: "mc-check" },
            h("span", { class: `ic s-${k.ok ? "pass" : "fail"}` }, k.ok ? "✓" : "✗"),
            h("span", { class: "nm" }, k.name.replaceAll("_", " ")),
            h("span", {}, k.detail)))) : null),
      h("div", { class: "right" }, badge(s)));
  });
  const vids = (rep && rep.videos) || [];
  const groups = (rep && rep.groups) || [];
  $("#mcRoom").replaceChildren(
    h("div", { class: "mc-room-h" },
      h("h2", {}, "Data Layers"), badge(state.dataBusy ? "working" : steps.length ? (steps.every((x) => x.ok) ? "done" : "failed") : "waiting"),
      run,
      h("p", {}, `Reads ${d.source || "the warehouse"} and builds source → staging → AI → analytics. ` +
        "A checker after every layer: if it fails, the layers after it do not run.")),
    state.dataError ? h("p", { class: "mc-note bad" }, state.dataError) : null,
    h("div", { class: "mc-sub" }, "Workers"),
    h("ul", { class: "mc-workers" }, workers),
    sum ? h("div", { class: "mc-sub" }, "What the analytics layer says") : null,
    sum ? h("p", { class: "mc-stat" },
      h("b", {}, sum.videos), " videos · ",
      h("b", {}, sum.matured), " with a full first week · ",
      h("b", {}, Math.round(sum.day7_views_total).toLocaleString()), " first-week views · average view ",
      h("b", {}, `${sum.avg_view_s}s`)) : null,
    groups.length ? h("div", { class: "mc-groups" }, groups.map((g) =>
      h("div", { class: "mc-group" },
        h("span", { class: "nm" }, `${g.dimension === "format" ? "format" : "hook"}: ${g.value || "-"}`),
        h("b", {}, `${g.avg_day7_view_s}s`), h("span", {}, " avg view · "),
        h("b", {}, Math.round(g.avg_day7_views)), h("span", {}, ` views · ${g.videos} videos`)))) : null,
    vids.length ? h("div", { class: "mc-table-wrap" }, h("table", { class: "mc-table" },
      h("thead", {}, h("tr", {}, ["#", "First-week views", "Avg view", "Hook", "Title"].map((c) => h("th", {}, c)))),
      h("tbody", {}, vids.map((v) => h("tr", {},
        h("td", {}, v.rank_views),
        h("td", { class: "num" }, Math.round(v.day7_views).toLocaleString()),
        h("td", { class: "num" }, `${(v.day7_avg_view_s || 0).toFixed(1)}s`),
        h("td", {}, v.hook_type || ""),
        h("td", { class: "ttl" }, v.title || "")))))) : null,
    d.why ? h("div", { class: "mc-sub" }, "Why some videos do better") : null,
    d.why ? h("div", { class: "mc-why" },
      d.why.text.split(/\n/).map((t) => h("p", {}, t.replace(/\s*\[[0-9a-f]{10}\]/g, ""))),
      h("p", { class: "mc-note" }, `Written by ${d.why.written_by} from lessons that held up; ` +
        "each sentence rests on one of the lessons below.")) : null,
    h("div", { class: "mc-sub" }, "Lessons from the channel's videos"),
    (d.lessons || []).length
      ? h("ul", { class: "mc-lessons" }, d.lessons.map((l) =>
          h("li", { class: `mc-lesson st-${l.status} c-${l.confidence.replace(" ", "-")}` },
            h("div", { class: "mc-lesson-h" },
              h("span", { class: "mc-badge" }, l.confidence),
              h("span", { class: "mc-badge" }, l.status),
              h("span", { class: "dim" }, `${l.dimension.replace("_", " ")} · ${l.metric === "day7_views" ? "views" : "seconds watched"}`)),
            h("p", {}, l.text),
            l.stable ? h("div", { class: "mc-lesson-act" },
              l.status !== "approved" ? h("button", { class: "mc-btn", type: "button",
                onclick: () => decideLesson(l.id, "approve") }, "Approve for next videos") : null,
              l.status !== "rejected" ? h("button", { class: "mc-btn", type: "button",
                onclick: () => decideLesson(l.id, "reject") }, l.status === "approved" ? "Stop using" : "Reject") : null)
              : h("p", { class: "mc-note" }, "Fragile: one video made it. It cannot be approved."))))
      : h("p", { class: "mc-note" }, "Run the layers to look for lessons."),
    sum ? h("p", { class: "mc-note" },
      "Approved lessons go to the Content Strategy and Editing Director agents when they plan " +
      "the next video, with their size label, as evidence rather than orders.") : null,
    sum ? h("p", { class: "mc-note" },
      `The stricter Learning agent compares our own editing decisions; it needs ${sum.learning_needs} videos with a finished first week, each linked to the ` +
      `Alpha Titans run that made it, before it trusts a pattern. Linked so far: ${sum.linked_to_jobs}. ` +
      "Until then these are descriptions, not causes.") : null);
}

async function loadData() {
  try {
    const r = await fetch(`${API}/data/layers`);
    if (r.ok) state.data = await r.json();
  } catch { /* the map still draws */ }
}

async function decideLesson(id, decision) {
  try {
    const r = await fetch(`${API}/data/lessons/${encodeURIComponent(id)}/${decision}`, { method: "POST" });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
    state.data = body; state.dataError = null;
  } catch (e) {
    state.dataError = `Could not save that decision: ${e.message}`;
  }
  draw();
}

async function runDataLayers() {
  state.dataBusy = true; state.dataError = null; draw();
  try {
    const r = await fetch(`${API}/data/layers/run`, { method: "POST" });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
    state.data = body;
    if (body.exit_code) state.dataError = "A checker failed; the layers after it did not run.";
  } catch (e) {
    state.dataError = `Could not run the data layers: ${e.message}`;
  }
  state.dataBusy = false; draw();
}

function draw() {
  const s = state.snap;
  if (!s) return;
  const pinnedData = state.selected === "data" && !state.follow;
  if (!pinnedData && (state.follow || !s.teams.some((t) => t.id === state.selected))) {
    const active = s.teams.find((t) => t.state === "working")
      || [...s.teams].reverse().find((t) => t.state === "done" || t.state === "failed")
      || s.teams.find((t) => t.state === "waiting") || s.teams[0];
    state.selected = active.id;
  }
  renderTop(s); renderChief(s); renderFlow(s);
  // only the Data Layers keep a room: its refresh button and lesson approvals
  const room = $("#mcRoom");
  room.hidden = state.selected !== "data";
  if (!room.hidden) renderDataRoom();
}

function syncFollow() {
  $("#mcFollow").setAttribute("aria-pressed", state.follow ? "true" : "false");
}

// ── data ──
function setSnapshot(snap) {
  // Redraw only when something a viewer can see has changed. Rebuilding the
  // page on every poll would wipe a click, a hover or a text selection.
  const key = JSON.stringify(snap, (k, v) => (k === "generated_at" ? undefined : v));
  if (key === state.key && state.jobId === state.keyJob) return;
  state.key = key; state.keyJob = state.jobId;
  state.snap = snap;
  state.receivedAt = Date.now() / 1000;
  draw();
}

function stop() {
  if (state.es) { state.es.close(); state.es = null; }
  if (state.poll) { clearTimeout(state.poll); state.poll = null; }
}

async function fetchSnapshot(id) {
  const r = await fetch(`${API}/jobs/${encodeURIComponent(id)}/mission`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function pollLater(id, ms) {
  state.poll = setTimeout(async () => {
    if (state.jobId !== id) return;
    try {
      const s = await fetchSnapshot(id);
      setSnapshot(s);
      if (s.status === "running") return follow(id);   // it started again
    } catch { /* keep trying */ }
    pollLater(id, 5000);
  }, ms);
}

function follow(id) {
  stop();
  if (!("EventSource" in window)) return pollLater(id, 1000);
  const es = new EventSource(`${API}/jobs/${encodeURIComponent(id)}/mission/stream`);
  state.es = es;
  es.addEventListener("snapshot", (ev) => {
    if (state.jobId === id) setSnapshot(JSON.parse(ev.data));
  });
  es.addEventListener("end", () => { es.close(); state.es = null; pollLater(id, 5000); });
  es.onerror = () => { es.close(); state.es = null; pollLater(id, 2000); };
}

async function openJob(id) {
  stop();
  state.jobId = id;
  state.follow = true; syncFollow();
  const url = new URL(location.href);
  if (id) url.searchParams.set("job", id); else url.searchParams.delete("job");
  history.replaceState(null, "", url);
  if (!id) {
    const r = await fetch(`${API}/mission/teams`);
    return setSnapshot(fromHierarchy(await r.json()));
  }
  try { setSnapshot(await fetchSnapshot(id)); }
  catch (e) {
    $("#mcFlow").replaceChildren(h("p", { class: "mc-empty" },
      `Could not read run ${id}: ${e.message}`));
    return;
  }
  follow(id);
}

async function init() {
  $("#mcFollow").addEventListener("click", () => {
    state.follow = !state.follow; syncFollow(); draw();
  });
  const pick = $("#mcJob");
  pick.addEventListener("change", () => openJob(pick.value || null));
  setInterval(() => {                                // tick running timers
    if (!state.snap) return;
    document.querySelectorAll("[data-live]").forEach((el) => {
      el.textContent = dur(Number(el.dataset.base) + (Date.now() / 1000 - state.receivedAt));
    });
  }, 1000);

  await loadData();
  setInterval(async () => { if (!state.dataBusy) { await loadData(); draw(); } }, 60000);

  let jobs = [];
  try {
    const r = await fetch(`${API}/mission/jobs`);
    jobs = (await r.json()).jobs || [];
  } catch { /* the page still shows the structure */ }
  pick.replaceChildren(h("option", { value: "" }, "— no run —"),
    ...jobs.map((j) => h("option", { value: j.id },
      `${j.id} · ${LABEL[j.status] || j.status} · ${clock(j.started_at)}`)));
  const want = new URL(location.href).searchParams.get("job");
  const first = want || (jobs.find((j) => j.status === "running") || jobs[0] || {}).id || null;
  pick.value = first || "";
  if (first && !jobs.some((j) => j.id === first))
    pick.append(h("option", { value: first }, first));
  pick.value = first || "";
  openJob(first);
}

init();
