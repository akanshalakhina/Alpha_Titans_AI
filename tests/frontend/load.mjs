// Pull named top-level declarations out of frontend/app.js and evaluate them
// in a sandbox, so the pure logic can be unit-tested without a browser.
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const SRC = fs.readFileSync(path.join(here, '../../frontend/app.js'), 'utf8');
const lines = SRC.split('\n');
const START = /^(?:async\s+)?(?:function\s+([A-Za-z_$][\w$]*)|const\s+([A-Za-z_$][\w$]*)|let\s+([A-Za-z_$][\w$]*))/;

function block(name) {
  const i = lines.findIndex(l => { const m = l.match(START); return m && (m[1] || m[2] || m[3]) === name; });
  if (i < 0) throw new Error('not found in app.js: ' + name);
  let j = i + 1;
  while (j < lines.length && !(START.test(lines[j]) || /^\/\*|^boot\(\);/.test(lines[j]))) j++;
  return lines.slice(i, j).join('\n');
}

export function load(names, globals = {}) {
  const ctx = vm.createContext({ console, Math, Date, JSON, Number, String, performance, ...globals });
  const code = names.map(block).join('\n') + '\n;({' + names.join(',') + '})';
  return vm.runInContext(code, ctx);
}
