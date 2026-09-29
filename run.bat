@echo off
REM Alpha Titans - Content Engine (Windows)
REM   run.bat            -> http://127.0.0.1:8000
REM   set AT_PORT=9000 ^& run.bat
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [X] python is not on PATH.
  exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [X] ffmpeg is not on PATH - the pipeline cannot run without it.
  echo     winget install Gyan.FFmpeg
  exit /b 1
)

where ffprobe >nul 2>nul
if errorlevel 1 (
  echo [X] ffprobe is not on PATH - it ships with ffmpeg.
  exit /b 1
)

python -c "import fastapi, uvicorn, multipart" 2>nul
if errorlevel 1 (
  echo [X] Python dependencies missing. Run: pip install -r requirements.txt
  exit /b 1
)

if not exist uploads    mkdir uploads
if not exist outputs    mkdir outputs
if not exist data\music mkdir data\music

if "%AT_HOST%"=="" set AT_HOST=127.0.0.1
if "%AT_PORT%"=="" set AT_PORT=8000

echo.
echo   Alpha Titans - Content Engine
echo   open http://%AT_HOST%:%AT_PORT%
echo.

python -m uvicorn main:app --app-dir "%~dp0server" --host %AT_HOST% --port %AT_PORT%
endlocal
