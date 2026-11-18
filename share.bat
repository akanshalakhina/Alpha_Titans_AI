@echo off
REM Alpha Titans - share a public link (Cloudflare quick tunnel)
REM Double-click this file. Keep BOTH windows open while your manager uses the link.
setlocal
cd /d "%~dp0"

set CF=cloudflared.exe
if not exist %CF% (
  where cloudflared >nul 2>nul && (set CF=cloudflared) || (
    echo [X] cloudflared was not found. Install it once with:
    echo     winget install --id Cloudflare.cloudflared
    echo or put cloudflared.exe in this folder.
    pause
    exit /b 1
  )
)

if "%AT_PORT%"=="" set AT_PORT=8000

echo Starting the Alpha Titans server in a new window...
start "Alpha Titans server" cmd /k run.bat

echo Waiting for the server to start...
timeout /t 6 /nobreak >nul

echo.
echo ================================================================
echo   Your share link appears below in a box, ending in
echo   .trycloudflare.com  -  copy it and send it to your manager.
echo   Closing this window turns the link off.
echo ================================================================
echo.
%CF% tunnel --url http://127.0.0.1:%AT_PORT%
pause
