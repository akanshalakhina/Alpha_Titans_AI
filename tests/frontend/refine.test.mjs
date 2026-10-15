// A refine must not read as "already finished".
//
// THE BUG THIS PINS
// -----------------
// Refining re-runs the whole graph over a job that already has a verdict,
// and the previous run's CriticResult stays on disk until the new one
// replaces it. So in the seconds after Send:
//
//   * `running` was computed from the verdict, saw PASS, and rendered the
//     OLD video with its Send button again - the working screen never
//     appeared and it looked as though the click had done nothing;
//   * `poll()` saw the same verdict and stopped, so the screen could not
//     correct itself either.
//
// The person pressed Send again and got "96f5d33b5168 is already being
// produced", which was true: their first press had started a run that was
// at that moment four nodes in.
//
// The fix is to stop inferring. `/status` now carries `producing`, asked
// of the thread that would be doing the work. `current_node` cannot stand
// in for it: a run killed mid-node leaves its row saying "running" for
// ever, which would pin the UI to a working screen that never ends.
import { test } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const SRC = fs.readFileSync(path.join(here, '../../frontend/studio.js'), 'utf8');

function grab(re) {
  const m = SRC.match(re);
  if (!m) throw new Error('not found in studio.js: ' + re);
  return m[0];
}

const RUNNING_RULE = grab(/ {2}const running = !!\(S\.jobId[\s\S]*?finishing\(\)\)\);/);
const POLL = grab(/function poll\(\)[\s\S]*?\n}/);
const FINISHING = grab(/function finishing\(\)[\s\S]*?\n}/);
const SETTLE = grab(/const SETTLE_POLLS = \d+;/);

/** Evaluate just the `running` expression against a state. */
function isRunning(S) {
  const ctx = vm.createContext({ S, console });
  return vm.runInContext(
    SETTLE + '\n' + FINISHING + '\n'
    + RUNNING_RULE.replace(/^ {2}const running = /, '') .replace(/;$/, ''),
    ctx);
}

/** Run poll() once and report whether it scheduled another read. */
function pollsAgain(S) {
  let scheduled = false;
  const ctx = vm.createContext({
    S, console,
    clearTimeout() {},
    setTimeout() { scheduled = true; return 1; },
  });
  vm.runInContext(SETTLE + '\n' + FINISHING + '\n' + POLL + '\n;poll()', ctx);
  return scheduled;
}

const JOB = 'j1';
const PUBLISHED = { output: '/x.mp4' };

// -------------------------------------------------- the moment after Send

const JUST_REFINED = {
  jobId: JOB, job: PUBLISHED, settled: 0,
  status: { producing: true, verdict: 'PASS', current_node: null },
};

test('a refine that has just started shows the working screen', () => {
  assert.equal(isRunning(JUST_REFINED), true,
    'the verdict belongs to the PREVIOUS run');
});

test('a refine that has just started keeps polling', () => {
  assert.equal(pollsAgain(JUST_REFINED), true,
    'stopping here is what froze the screen for the whole run');
});

test('the settle budget is not spent while a run is in flight', () => {
  const S = { ...JUST_REFINED, settled: 0 };
  pollsAgain(S);
  assert.equal(S.settled, 0, 'settling means idle, and this is not idle');
});

// ------------------------------------------------------- still an ending

test('a genuinely finished job is still finished', () => {
  const done = {
    jobId: JOB, job: PUBLISHED, settled: 0,
    status: { producing: false, verdict: 'PASS', current_node: null },
  };
  assert.equal(isRunning(done), false);
  assert.equal(pollsAgain(done), false, 'nothing left to wait for');
});

test('a stale running row does not pin the UI to a working screen', () => {
  // what a killed run leaves behind: a node that never closed its row
  const stale = {
    jobId: JOB, job: PUBLISHED, settled: 0,
    status: { producing: false, verdict: 'PASS', current_node: 'music' },
  };
  assert.equal(isRunning(stale), false,
    'the thread is gone; only `producing` knows that');
});

test('a first run with no verdict yet is running, as before', () => {
  assert.equal(isRunning({
    jobId: JOB, job: null, settled: 0,
    status: { producing: true, verdict: null, current_node: 'measure' },
  }), true);
});

test('the optimistic flag set before the first poll counts too', () => {
  assert.equal(isRunning({
    jobId: JOB, job: PUBLISHED, settled: 0,
    status: { starting: true, verdict: 'PASS' },
  }), true, 'between POST /produce and the first /status read');
});

test('a failed run is terminal whatever else it says', () => {
  assert.equal(pollsAgain({
    jobId: JOB, job: null, settled: 0,
    status: { producing: true, verdict: null, failed: { error: 'x' } },
  }), false);
});
