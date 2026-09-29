#!/usr/bin/env bash
# Runs every Alpha Titans test layer. Usage: ./run_tests.sh [--fast]
set -u
cd "$(dirname "$0")"
FAST=${1:-}
rc=0
echo "== 1/3 Frontend unit tests (node) =="
node --test tests/frontend/app.test.mjs || rc=1
echo "== 2/3 Backend unit + integration tests (pytest) =="
if [ "$FAST" = "--fast" ]; then python3 -m pytest tests/backend -m "not slow" -q -p no:cacheprovider || rc=1
else python3 -m pytest tests/backend -q -p no:cacheprovider || rc=1; fi
echo "== 3/3 End-to-end + regression tests (Playwright) =="
python3 -m pytest tests/e2e -v -p no:cacheprovider || rc=1
exit $rc
