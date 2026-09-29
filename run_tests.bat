@echo off
REM Alpha Titans - run every test layer on Windows.
REM   run_tests.bat            full run (includes real renders, ~10 min)
REM   run_tests.bat --fast     skip the slow render tests
REM   set AT_TEST_BROWSER=msedge ^& run_tests.bat    use real Edge (plays H.264)
REM One-time setup:
REM   pip install -r requirements.txt pytest httpx playwright
REM   python -m playwright install chromium
setlocal
cd /d "%~dp0"
set RC=0
echo == 1/3 Frontend unit tests (node) ==
where node >nul 2>nul && (node --test tests/frontend/app.test.mjs || set RC=1) || echo    node not installed - skipped
echo == 2/3 Backend unit + integration tests (pytest) ==
if "%1"=="--fast" (python -m pytest tests/backend -m "not slow" -q -p no:cacheprovider || set RC=1) else (python -m pytest tests/backend -q -p no:cacheprovider || set RC=1)
echo == 3/3 End-to-end + regression tests (Playwright) ==
python -m pytest tests/e2e -v -p no:cacheprovider || set RC=1
if %RC%==0 (echo. & echo ALL TESTS PASSED) else (echo. & echo SOME TESTS FAILED - see above)
exit /b %RC%
