// The production-polling rule, tested as pure logic.
//
// THE BUG THIS PINS
// -----------------
// `poll()` stopped the moment a verdict existed. But the critic writes the
// verdict and the graph keeps going - thumbnail, title, publishing, learning
// - and only when it returns is the finished file published into the record
// that /download serves. The result screen's <video> streams from that same
// route, so it rendered against a 409 and, because polling had stopped,
// nothing ever repaired it: the person sat looking at a dead player until
// they reloaded. Measured on a real run: verdict at 86s, file servable at
// 143s - a 57 second window.
//
// A REVISE verdict is worse still: it is mid-run, so the old rule showed the
// result screen while the director was re-planning.
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

// SETTLE_POLLS and finishing() together are the whole rule
const code = grab(/const SETTLE_POLLS = \d+;/)
  + '\n' + grab(/function finishing\(\)[\s\S]*?\n}/)
  + '\n;({ finishing, SETTLE_POLLS })';

function withState(S) {
  const ctx = vm.createContext({ S, console });
  return vm.runInContext(code, ctx);
}

// read the constant once, so tests can reference it when building state
const { SETTLE_POLLS } = withState({ status: null, job: null, settled: 0 });

const RUNNING = { verdict: null, current_node: 'render' };

test('still rendering: not finishing, because there is no verdict yet', () => {
  const { finishing } = withState({ status: RUNNING, job: null, settled: 0 });
  assert.equal(finishing(), false);
});

test('verdict in, file not published yet: keep waiting', () => {
  const { finishing } = withState({
    status: { verdict: 'PASS', current_node: null }, job: {}, settled: 0 });
  assert.equal(finishing(), true, 'the 57s window the player died in');
});

test('verdict in and file published: done', () => {
  const { finishing } = withState({
    status: { verdict: 'PASS', current_node: null },
    job: { output: { file: 'v.mp4' } }, settled: 0 });
  assert.equal(finishing(), false);
});

test('REVISE is mid-run, not the end', () => {
  const { finishing } = withState({
    status: { verdict: 'REVISE', current_node: 'editing_director' },
    job: {}, settled: 0 });
  assert.equal(finishing(), true);
});

test('a long revision does not burn the settle budget', () => {
  // a node is working, so the wait is not "settling" however long it takes
  const { finishing } = withState({
    status: { verdict: 'REVISE', current_node: 'render' },
    job: {}, settled: SETTLE_POLLS * 10 });
  assert.equal(finishing(), true);
});

test('the wait gives up eventually so the result is never withheld', () => {
  const { finishing } = withState({
    status: { verdict: 'PASS', current_node: null },
    job: {}, settled: SETTLE_POLLS });
  assert.equal(finishing(), false, 'show what we have rather than hang');
});

test('a failed run is terminal, not finishing', () => {
  const { finishing } = withState({
    status: { verdict: 'PASS', failed: { error: 'boom' }, current_node: null },
    job: {}, settled: 0 });
  assert.equal(finishing(), false);
});

test('an old job opened from the list is immediately done', () => {
  const { finishing } = withState({
    status: { verdict: 'PASS', current_node: null },
    job: { output: { file: 'old.mp4' } }, settled: 0 });
  assert.equal(finishing(), false, 'must not re-poll a finished production');
});

// ───────────────────────── routing: a refresh must not lose the build ──
//
// There was no routing at all: every view lived only in `S`, so a refresh -
// or the back button, or reopening the tab - dropped a running production
// and landed back on Create with no way to find the job again. A build takes
// minutes; losing it to an accidental F5 is not acceptable.
const routeCode = grab(/function readHash\(\)[\s\S]*?\n}/) + '\n;({ readHash })';

function readHashWith(hash) {
  const ctx = vm.createContext({ location: { hash }, console });
  return vm.runInContext(routeCode, ctx).readHash(hash);
}

test('a production hash names the job to reopen', () => {
  assert.deepEqual(readHashWith('#/production/e426ddebc4fd'),
                   { view: 'production', id: 'e426ddebc4fd' });
});

test('a plain view hash carries no job', () => {
  assert.deepEqual(readHashWith('#/agents'), { view: 'agents', id: null });
  assert.deepEqual(readHashWith('#/create'), { view: 'create', id: null });
});

test('no hash at all is not a route', () => {
  assert.equal(readHashWith(''), null);
  assert.equal(readHashWith('#'), null);
});

test('a job id with underscores and dashes still parses', () => {
  assert.deepEqual(readHashWith('#/production/ab_12-CD'),
                   { view: 'production', id: 'ab_12-CD' });
});

test('junk in the hash is ignored rather than guessed at', () => {
  assert.equal(readHashWith('#/production/../../etc/passwd'), null);
  assert.equal(readHashWith('#/Production/ABC'), null);
  assert.equal(readHashWith('#not-a-route'), null);
});
