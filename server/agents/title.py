"""
Title Agent - the title the delivered video is published under.

WHERE IT SITS
-------------
    ... render -> critic --PASS-----> finish   --+
                    |                             +--> thumbnail -> title -> END
                    +--ESCALATE--> escalate -----+

PACKAGING, not editing. It runs ONCE, after the render/critic cycle has
settled on the video that will be delivered, so it can never cost a render or
trigger a revision - and so it knows which clips actually made the cut.

WHAT PUBLISHING ACTUALLY EXISTS
-------------------------------
None, as an integration: nothing in this project uploads a video or writes
container metadata. A title reaches exactly two places:

  * master_stage names the delivered FILE from `JobSettings.title`. That
    happens inside the render, before packaging, and is never changed after
    the fact - renaming a verified file would orphan every path that points
    at it.
  * the app shows `settings.title` on the job card and above the player.
    Untitled agent builds show the JobSettings placeholder there.

So the title is published by filling that one display slot - and only when
the operator did not name the video. Operator settings win: an operator title
is recorded verbatim and nothing else is considered.

NO WORD IS UNSUPPORTED
----------------------
Every content word of a title must appear in the validated material about
THIS video: the operator's domain, the ContentStrategy, and the Story beats on
clips the final cut kept. Research is read as context - how the subject is
talked about - but a word found ONLY in research describes the wider world,
not this footage, and is rejected with that reason. Superlatives and hype are
rejected outright, a number must be one the material contains, and a link or
path is not a title.

FALLBACK
--------
With no model, a title is SELECTED, never written: the strategy's central
topic, then the story's logline, then the operator's own domain - each used
verbatim apart from title case, each passed through the same check. With none
of those, there is no title, and the placeholder stands.
"""

from __future__ import annotations

import math
import re
import time

from .artifacts import (TITLE_MAX_CHARS, Artifact, ContentStrategy, Cost,
                        Decision, EditPlan, JobSettings, ResearchReport,
                        Story, Title)
from .base import Agent, AgentContext
from .trace import TraceSpan

PROMPT_VERSION = "title-prompt-v1"
MAX_TITLES = 3
MAX_WORDS = 14
MAX_LIST = 8

# The value JobSettings.title takes when the operator named nothing - read
# from the schema so the two cannot drift.
PLACEHOLDER = JobSettings.model_fields["title"].default

# Words that carry no claim about the footage and so need no grounding. Number
# words are deliberately NOT here: "Three Gears" is a claim.
FUNCTION_WORDS = frozenset("""
a an the and or but nor of to in on at for with without from by into onto
out up down over under through about as than then so vs versus via per
toward towards across between among against during after before upon
within beyond near off again while because although until since
amid along around behind beside inside outside past throughout
this that these those it its it's is are was were be been being has have had
do does did can could will would should may might
i me my we us our you your he him his she her they them their
how what why when where which who whose whom
step steps part
how-to
watch see look inside meet make making made build building built work works
working
""".split())

# Claims no footage supports on its own. Rejected even when some upstream text
# happens to contain the word: in a title they describe the VIDEO.
HYPE = frozenset("""
best ultimate greatest worst insane crazy shocking unbelievable incredible
amazing epic secret secrets hack hacks guaranteed viral exclusive official
breaking perfect genius impossible mindblowing mind-blowing legendary
revolutionary world's #1 must-see must-watch jawdropping jaw-dropping
""".split())

# What a reply might ask for that this pipeline cannot publish.
EXTRA_KEYS = ("description", "tags", "hashtags", "keywords", "emoji",
              "subtitle", "thumbnail_text", "chapters", "category")

SMALL = frozenset("a an the and or nor of to in on at for with by from vs "
                  "via as".split())

_TOKEN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*")
_LINKISH = re.compile(r"https?://|www\.|[\\/]|\.(?:com|net|org|io)\b", re.I)

PROMPT = """You are choosing the TITLE for a finished short video.

The video is already cut and rendered. Everything below describes what it
actually contains - and nothing else may appear in the title.

{about}{research}
WHAT A TITLE HERE MAY BE
  - one line, at most {max_chars} characters and {max_words} words
  - every content word must appear in the material ABOVE THE RESEARCH. Words
    found only in the research describe the wider subject, not this video,
    and a title using them is REJECTED.
  - no superlatives or hype ({hype}); no number the material does not
    contain; no links, no file names
  - only a title: there is no description, tag, hashtag or thumbnail text
    to write. If you want one, put it in `unsupported_requests`.

Every title is checked word by word after you reply. One that fails is
dropped, and a weaker supported title is used instead.

Reply with ONLY a JSON object, no prose around it:

{{
  "titles": ["<best>", "<second>", "<third>"],
  "rationale": "<why the first one fits this video and its audience>",
  "unsupported_requests": ["<anything you wanted to publish besides a title>"],
  "risks": ["<what could make this title mislead or underperform>"],
  "limitations": ["<what you could not judge from what you were given>"],
  "confidence": <0.0-1.0 - be low when you were given little to work with>
}}"""


# ------------------------------------------------------------- sanitising

def _num(raw, lo: float, hi: float, fallback: float = 0.0) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return fallback
    return fallback if not math.isfinite(v) else max(lo, min(hi, v))


def _text(raw, limit: int = 200) -> str:
    if not isinstance(raw, (str, int, float)) or isinstance(raw, bool):
        return ""
    return " ".join(str(raw).split())[:limit]


def _texts(raw, limit: int = MAX_LIST, each: int = 160) -> list[str]:
    if not isinstance(raw, list):
        return []
    out = []
    for x in raw[:limit]:
        t = _text(x, each)
        if t:
            out.append(t)
    return out


def _line(raw: str) -> str:
    """One trimmed printable line, without the full stop a sentence ends on."""
    s = "".join(ch if ch.isprintable() else " " for ch in raw)
    return " ".join(s.split()).rstrip(" .;:,")


def _title_case(s: str) -> str:
    """Case only - never a word added, dropped or reordered. A word that is
    not plain lowercase (LEGO, iPhone, McLaren) is left exactly as written."""
    out = []
    for i, w in enumerate(s.split()):
        if not w.islower():
            out.append(w)
        elif i and w in SMALL:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


# ---------------------------------------------------------------- words

def _forms(word: str) -> set[str]:
    """A word and its plain inflections, so 'gears' finds 'gear' and
    'spinning' finds 'spins' - without any notion of synonyms."""
    w = word.lower().replace("’", "'")
    if w.endswith("'s"):
        w = w[:-2]
    w = w.replace("'", "")
    out = {w}
    for suf, add in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""),
                     ("ing", "e"), ("ed", ""), ("ed", "e"), ("d", ""),
                     ("ers", ""), ("er", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[:-len(suf)] + add
            out.add(base)
            # spinning -> spinn -> spin
            if not add and len(base) >= 4 and base[-1] == base[-2]:
                out.add(base[:-1])
    return out


def _vocab(texts: list[tuple[str, str]]) -> dict[str, str]:
    """form -> where it was found, first source wins. A hyphenated word also
    contributes its parts, so 'quad-blade' grounds 'blade'."""
    v: dict[str, str] = {}
    for label, text in texts:
        for w in _TOKEN.findall(text or ""):
            for part in [w] + (w.split("-") if "-" in w else []):
                for f in _forms(part):
                    v.setdefault(f, label)
    return v


def _where(word: str, vocab: dict[str, str]) -> str | None:
    """Where a title word was found - whole, or every part of a hyphenated
    word ('gear-driven' needs both 'gear' and 'driven')."""
    hit = next((vocab[f] for f in _forms(word) if f in vocab), None)
    if hit is not None or "-" not in word:
        return hit
    parts = [p for p in word.split("-") if p and p.lower() not in
             FUNCTION_WORDS]
    found = [next((vocab[f] for f in _forms(p) if f in vocab), None)
             for p in parts]
    return found[0] if parts and all(found) else None


def _check(text: str, vocab: dict[str, str], research: dict[str, str],
           max_chars: int) -> tuple[str | None, list[str]]:
    """(why it fails, or None; the grounded content words)."""
    if not text:
        return "empty", []
    if len(text) > max_chars:
        return f"longer than {max_chars} characters", []
    if _LINKISH.search(text):
        return "contains a link or a path", []
    words = _TOKEN.findall(text)
    if not words:
        return "contains no words", []
    if len(words) > MAX_WORDS:
        return f"more than {MAX_WORDS} words", []
    hype = [w for w in words if w.lower() in HYPE]
    if "#" in text:
        hype.append("#")
    if hype:
        return "an unsupported claim: " + ", ".join(hype), []

    terms, missing, research_only = [], [], []
    for w in words:
        if w.lower() in FUNCTION_WORDS:
            continue
        where = _where(w, vocab)
        if where is not None:
            terms.append(f"{w.lower()}<-{where}")
        elif _where(w, research) is not None:
            research_only.append(w)
        else:
            missing.append(w)
    if research_only:
        return ("found only in research about the wider subject, not in "
                "anything about this video: " + ", ".join(research_only[:5])), []
    if missing:
        return ("not in anything about this video: "
                + ", ".join(missing[:5])), []
    if not terms:
        return "no content word - it says nothing about the video", []
    return None, terms


class TitleAgent(Agent):
    id = "title"
    produces = "Title"
    uses_llm = True
    agency = "reasoning"

    def execute(self, ctx: AgentContext, span: TraceSpan,
                plan: EditPlan | None = None,
                strategy: ContentStrategy | None = None,
                story: Story | None = None,
                research: ResearchReport | None = None,
                revision: int = 0) -> Artifact[Title]:
        started = time.time()
        domain = (ctx.job.settings.domain or "").strip()
        ldesc = ctx.llm.describe()

        raw_op = (ctx.job.settings.title or "").strip()
        operator = raw_op if raw_op and raw_op != PLACEHOLDER else None

        cs_ok = bool(strategy is not None and strategy.central_topic)
        in_cut = ({s.clip_id for s in plan.segments} if plan is not None
                  else None)
        beats = ([b for b in story.beats
                  if in_cut is None or b.clip_id in in_cut]
                 if story is not None else [])
        # the story counts only if it lends words: a beat with no narrative,
        # or an assembled logline, says nothing a title could rest on
        st_ok = bool(any(b.narrative.strip() for b in beats)
                     or (story is not None and story.live and story.logline))
        r_ok = bool(research is not None and research.live
                    and not research.insufficient_signal)

        vocab = _vocab(self._about(domain, strategy, story, beats))
        rvocab = _vocab(self._research_texts(research)) if r_ok else {}

        inputs_hash = self.hash_inputs(
            domain, PROMPT_VERSION, ldesc.get("model"), ldesc.get("provider"),
            operator, self._strategy_digest(strategy),
            self._story_digest(story, beats),
            self._research_digest(research) if r_ok else None,
            sorted(in_cut) if in_cut is not None else None, TITLE_MAX_CHARS)
        cached = ctx.store.get_by_inputs(ctx.job.job_id, "Title",
                                         inputs_hash, Title)
        if cached is not None:
            span.cache(1)
            span.note("reusing the title: the operator's setting and every "
                      "source it was drawn from are unchanged")
            span.skipped("cache hit - upstream inputs identical")
            return cached

        b = dict(domain=domain, operator=operator, raw_op=raw_op,
                 cs_ok=cs_ok, st_ok=st_ok, r_ok=r_ok, vocab=vocab,
                 rvocab=rvocab, dropped_beats=(len(story.beats) - len(beats)
                                               if story is not None else 0))

        # ---- the operator named it: nothing else is considered --------
        if operator is not None:
            return self._operator(ctx, span, b, revision, started,
                                  inputs_hash)

        if not ctx.llm.available():
            return self._deterministic(
                ctx, span, b, strategy, story, revision, started,
                inputs_hash,
                why=(ldesc.get("why_disabled") or "no model configured"))

        prompt = self._build_prompt(domain, strategy, story, beats,
                                    research if r_ok else None)
        span.tool("llm.complete_json", purpose="title",
                  prompt_chars=len(prompt))
        # 700 was enough for a model that answers straight away, and far too
        # little for a reasoning model, which spends most of the output
        # budget thinking before it writes anything. Measured against
        # gpt-oss-120b this agent hit exactly 700/700 on every run and
        # returned an empty `content`, so it fell back to a deterministic
        # title every single time. 1400 matches Caption and Transition, the
        # two agents with comparably sized replies.
        raw, reply = ctx.llm.complete_json(prompt, max_tokens=1400)
        cost = Cost(tokens_in=reply.tokens_in, tokens_out=reply.tokens_out,
                    usd=reply.usd, wall_s=round(time.time() - started, 2))
        span.llm(reply.tokens_in, reply.tokens_out, reply.usd)

        if not reply.ok or not isinstance(raw, dict):
            span.note(f"title reply unusable: "
                      f"{reply.error or 'not a JSON object'}")
            return self._deterministic(
                ctx, span, b, strategy, story, revision, started,
                inputs_hash, why="the model reply was unusable", cost=cost)

        passed, rejected = self._screen(raw.get("titles"), b)
        span.tool("title.ground", proposed=len(passed) + len(rejected),
                  passed=len(passed))
        for r in rejected:
            span.note("title rejected - " + r)
        if not passed:
            return self._deterministic(
                ctx, span, b, strategy, story, revision, started,
                inputs_hash, why="no proposed title could be supported",
                cost=cost, prior_rejected=rejected)

        text, terms = passed[0]
        unsupported = _texts(raw.get("unsupported_requests"), each=160)
        for k in EXTRA_KEYS:
            v = raw.get(k)
            if v not in (None, "", [], {}):
                unsupported.append(f"{k} - this pipeline publishes a title "
                                   f"and nothing else")
        limitations = _texts(raw.get("limitations"), each=200)
        limitations += self._standing_limitations(b)
        if unsupported:
            limitations.append("%d request(s) fall outside what this "
                               "pipeline can publish" % len(unsupported))
        confidence = _num(raw.get("confidence"), 0.0, 1.0)
        if not (b["cs_ok"] or b["st_ok"]):
            confidence = min(confidence, 0.4)

        payload = Title(
            domain=domain, title_type="reasoned", text=text, source="model",
            grounded_terms=terms,
            alternatives=[t for t, _ in passed[1:]],
            rejected=rejected[:MAX_LIST],
            operator_title=None, published=True,
            unsupported_requests=list(dict.fromkeys(unsupported))[:MAX_LIST],
            strategy_available=cs_ok, story_available=st_ok,
            research_available=r_ok,
            risks=_texts(raw.get("risks")),
            limitations=limitations[:MAX_LIST],
            insufficient_signal=False, confidence=confidence,
            live=True, model=ldesc.get("model"))
        ctx.emit("title: %r (%d grounded word(s), %d rejected)"
                 % (text, len(terms), len(rejected)))
        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span,
            decisions=self._decisions(payload, b,
                                      rationale=_text(raw.get("rationale"),
                                                      300)),
            confidence=confidence, degraded=False, fallback_used=None,
            cost=cost)

    # ------------------------------------------------------------ sources

    @staticmethod
    def _about(domain: str, strategy: ContentStrategy | None,
               story: Story | None, beats) -> list[tuple[str, str]]:
        """Everything validated about THIS video, labelled by where it is.

        A reasoned strategy contributes its brief; an assembled one only its
        topic, because the rest of it is machinery ("clips in order of...").
        Story beats count only for clips the final cut kept.
        """
        texts = [("domain", domain)]
        if strategy is not None:
            texts.append(("strategy", strategy.central_topic))
            if strategy.live:
                texts += [("strategy", strategy.core_angle),
                          ("strategy", strategy.objective),
                          ("strategy", strategy.audience),
                          ("strategy", strategy.hook.description)]
                texts += [("strategy", k.point)
                          for k in strategy.grounded_points]
        if story is not None:
            if story.live:
                texts.append(("story", story.logline))
                texts += [("story", t) for t in story.themes]
            texts += [("story", b.narrative) for b in beats]
        return [(label, t) for label, t in texts if t]

    @staticmethod
    def _research_texts(research: ResearchReport | None):
        if research is None:
            return []
        texts = [("research", t) for t in
                 research.keywords + research.topics +
                 research.audience_signals]
        texts += [("research", t.label) for t in research.trends]
        texts += [("research", e.title) for e in research.evidence]
        return texts

    # ------------------------------------------------------------- screen

    def _screen(self, raw, b: dict) -> tuple[list, list[str]]:
        """Every proposed title, checked word by word. Returns the ones that
        passed as (text, grounded terms), and why each other one failed."""
        if isinstance(raw, str):
            raw = [raw]
        passed, rejected = [], []
        seen = set()
        for t in (raw if isinstance(raw, list) else [])[:MAX_TITLES]:
            text = _line(t) if isinstance(t, str) else ""
            if not text or text.lower() in seen:
                if not text:
                    rejected.append("an empty or non-text title")
                continue
            seen.add(text.lower())
            why, terms = _check(text, b["vocab"], b["rvocab"],
                                TITLE_MAX_CHARS)
            if why is None:
                passed.append((text, terms))
            else:
                rejected.append(f"{text[:60]!r}: {why}")
        return passed, rejected

    @staticmethod
    def _standing_limitations(b: dict) -> list[str]:
        out = []
        if not (b["cs_ok"] or b["st_ok"]):
            out.append("neither a strategy topic nor any story beat in the "
                       "cut was available, so the title rests on the "
                       "operator's domain alone")
        if b["dropped_beats"]:
            out.append(f"{b['dropped_beats']} story beat(s) are on clips the "
                       f"final cut dropped and lent the title nothing")
        if not b["r_ok"]:
            out.append("no live research was available, so the wording is "
                       "not informed by how the subject is searched for")
        out.append("the delivered file keeps the name it was rendered under; "
                   "the title is what the app shows for it")
        return out

    # ------------------------------------------------------------ operator

    def _operator(self, ctx: AgentContext, span: TraceSpan, b: dict,
                  revision: int, started: float,
                  inputs_hash: str) -> Artifact[Title]:
        """The operator's title, verbatim. It is not checked, scored or
        improved on - operator settings win over anything an agent decides."""
        op = b["operator"]
        fits = (len(op) <= TITLE_MAX_CHARS
                and all(ch.isprintable() for ch in op))
        span.note("the operator named this video; no title was generated")
        payload = Title(
            domain=b["domain"], title_type="operator",
            text=op if fits else "", source="operator" if fits else None,
            operator_title=op, published=False,
            strategy_available=b["cs_ok"], story_available=b["st_ok"],
            research_available=b["r_ok"],
            limitations=([] if fits else [
                "the operator's title is kept exactly as set; it is longer "
                f"than {TITLE_MAX_CHARS} characters or not one printable "
                "line, so it is recorded here but not repeated as a title"]),
            insufficient_signal=False, confidence=1.0,
            live=False, model=None)
        ctx.emit(f"title: the operator's own - {op!r}")
        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=self._decisions(payload, b),
            confidence=1.0, degraded=False, fallback_used=None,
            cost=Cost(wall_s=round(time.time() - started, 2)))

    # ------------------------------------------------------ deterministic

    def _deterministic(self, ctx: AgentContext, span: TraceSpan, b: dict,
                       strategy: ContentStrategy | None,
                       story: Story | None, revision: int, started: float,
                       inputs_hash: str, why: str, cost: Cost | None = None,
                       prior_rejected: list[str] | None = None
                       ) -> Artifact[Title]:
        """Select, never write: the strategy's topic, then the story's
        logline, then the operator's domain - verbatim but for case, and each
        through the same check a model's title faces."""
        span.fallback(f"deterministic title ({why})")
        options = []
        if strategy is not None and strategy.central_topic:
            options.append(("strategy", strategy.central_topic))
        if story is not None and story.live and story.logline:
            options.append(("story", story.logline))
        if b["domain"]:
            options.append(("domain", b["domain"]))

        rejected = list(prior_rejected or [])
        text, source, terms = "", None, []
        for src, raw_text in options:
            cand = _title_case(_line(raw_text))
            reason, got = _check(cand, b["vocab"], b["rvocab"],
                                 TITLE_MAX_CHARS)
            if reason is None:
                text, source, terms = cand, src, got
                break
            rejected.append(f"{src} {cand[:60]!r}: {reason}")

        limitations = [f"{why}; " + (
            f"the title is the {source} text, selected verbatim - nothing "
            f"was written" if text else
            "no supportable title could be selected, so the placeholder "
            "stands - none was invented")]
        limitations += self._standing_limitations(b)

        payload = Title(
            domain=b["domain"],
            title_type="degraded" if text else "minimal",
            text=text, source=source, grounded_terms=terms,
            rejected=rejected[:MAX_LIST], operator_title=None,
            published=bool(text),
            strategy_available=b["cs_ok"], story_available=b["st_ok"],
            research_available=b["r_ok"],
            limitations=limitations[:MAX_LIST],
            insufficient_signal=True, confidence=0.0,
            live=False, model=None)
        ctx.emit("title: %s (%s)" % (repr(text) if text else "none", why))
        return self.wrap(
            ctx, payload, revision=revision, inputs_hash=inputs_hash,
            span=span, decisions=self._decisions(payload, b, why=why),
            confidence=0.0, degraded=True,
            fallback_used=f"deterministic title ({why})",
            cost=cost or Cost(wall_s=round(time.time() - started, 2)))

    # ------------------------------------------------------------ prompt

    def _build_prompt(self, domain: str, strategy, story, beats,
                      research) -> str:
        rows = ["WHAT THIS VIDEO IS"]
        if domain:
            rows.append(f"  footage   {domain} (the operator's own words)")
        if strategy is not None and strategy.central_topic:
            rows.append(f"  topic     {strategy.central_topic}")
            if strategy.live:
                for label, val in (("angle", strategy.core_angle),
                                   ("audience", strategy.audience),
                                   ("objective", strategy.objective)):
                    if val:
                        rows.append(f"  {label:<9} {val}")
                for k in strategy.grounded_points[:5]:
                    rows.append(f"  point     {k.point}")
        if story is not None and story.live and story.logline:
            rows.append(f"  logline   {story.logline}")
        if beats:
            rows.append("  in the cut, in order:")
            rows += [f"    {b.role:<8} {b.narrative}" for b in beats
                     if b.narrative]
        about = "\n".join(rows) + "\n"

        rtext = ""
        if research is not None:
            r = ["\nHOW THE SUBJECT IS TALKED ABOUT (research - CONTEXT ONLY,"
                 " not a source of words)"]
            if research.keywords:
                r.append("  keywords  " + ", ".join(research.keywords[:10]))
            for t in research.trends[:4]:
                r.append(f"  trend     {t.label}")
            rtext = "\n".join(r) + "\n"
        return PROMPT.format(about=about, research=rtext,
                             max_chars=TITLE_MAX_CHARS, max_words=MAX_WORDS,
                             hype=", ".join(sorted(HYPE)[:8]) + ", ...")

    # ---------------------------------------------------------- digests

    @staticmethod
    def _strategy_digest(s: ContentStrategy | None):
        if s is None:
            return None
        return (s.central_topic, s.core_angle, s.audience, s.objective,
                s.hook.description, s.live,
                [k.point for k in s.grounded_points])

    @staticmethod
    def _story_digest(story: Story | None, beats):
        if story is None:
            return None
        return (story.logline, story.live, list(story.themes),
                [(b.index, b.clip_id, b.narrative) for b in beats])

    @staticmethod
    def _research_digest(r: ResearchReport):
        return (r.keywords, r.topics, [t.label for t in r.trends],
                [e.title for e in r.evidence])

    # -------------------------------------------------------- decisions

    def _decisions(self, p: Title, b: dict, why: str = "",
                   rationale: str = "") -> list[Decision]:
        sources = ([s for s, ok in (("ContentStrategy", b["cs_ok"]),
                                    ("Story", b["st_ok"]),
                                    ("Research (context only)", b["r_ok"]))
                    if ok] or ["the operator's domain alone"])
        out = [Decision(
            what="Evidence the title was chosen from",
            chose=" + ".join(sources),
            why=("recorded explicitly so a title chosen without a reading of "
                 "the video cannot later be read as though it had one; "
                 "research may shape wording but never supplies a word"),
            decided_by="measurement", confidence=1.0)]

        out.append(Decision(
            what="Where a title goes",
            chose=("the app's title slot, in place of the placeholder"
                   if p.published else
                   "nowhere new - the operator's title already shows"
                   if p.source == "operator" else
                   "nowhere - the placeholder stands"),
            why=("no publishing integration exists; master_stage named the "
                 "delivered file during the render and it keeps that name, so "
                 "the title is what the app shows for the video"),
            decided_by="measurement", confidence=1.0))

        if p.title_type == "operator":
            out.append(Decision(
                what="Title", chose=b["operator"][:200],
                why="the operator named this video; operator settings win "
                    "over anything an agent decides, so nothing was generated",
                decided_by="human", confidence=1.0))
            return out

        out.append(Decision(
            what="Title",
            chose=(p.text or "none")[:200],
            why=((rationale or "no reason given")[:300] if p.live else
                 f"{why}; " + (f"selected verbatim from the {p.source}"
                               if p.text else
                               "nothing supportable to select")),
            decided_by="llm" if p.live else "fallback",
            confidence=round(p.confidence, 2) if p.live else 1.0))

        if p.grounded_terms:
            out.append(Decision(
                what="Every content word found upstream",
                chose=f"{len(p.grounded_terms)} word(s)",
                why=", ".join(p.grounded_terms[:8])[:300],
                decided_by="measurement", confidence=1.0))
        if p.rejected:
            out.append(Decision(
                what="Titles rejected",
                chose=f"{len(p.rejected)} title(s)",
                why="; ".join(p.rejected[:2])[:300],
                decided_by="policy", confidence=1.0))
        if p.unsupported_requests:
            out.append(Decision(
                what="Asked for but not possible",
                chose=f"{len(p.unsupported_requests)} request(s)",
                why=("; ".join(p.unsupported_requests[:2])[:250]
                     + " - recorded as a gap rather than promised"),
                decided_by="policy", confidence=1.0))
        return out


__all__ = ["TitleAgent", "PLACEHOLDER"]
