@echo off
REM ============================================================
REM  Alpha Titans - ek click mein start
REM
REM  Ye file ffmpeg ka folder khud PATH mein daal deti hai, phir
REM  server chalu karke browser khol deti hai. Windows ki system
REM  settings mein kuch nahi badalta - ye sirf is window ke liye
REM  hai.
REM
REM  Chalane ke liye: is file pe double-click karo.
REM  Band karne ke liye: is window mein Ctrl+C dabao.
REM ============================================================
setlocal
cd /d "%~dp0"

REM --- ffmpeg ka folder (winget ne yahan install kiya tha) -----
set "FFBIN=%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
if exist "%FFBIN%\ffmpeg.exe" (
  set "PATH=%FFBIN%;%PATH%"
)

REM --- check: sab kuch mil raha hai? --------------------------
where python >nul 2>nul
if errorlevel 1 (
  echo [X] python nahi mila. Python install karo aur PATH mein daalo.
  pause
  exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [X] ffmpeg nahi mila.
  echo     Install karo:  winget install Gyan.FFmpeg
  echo     Ya is file mein FFBIN ka path theek karo.
  pause
  exit /b 1
)

if not exist ".env" (
  echo [!] .env file nahi hai - AI agents fallback pe chalenge.
  echo     Video phir bhi banegi, par reasoning nahi hogi.
  echo.
)

if not exist uploads mkdir uploads
if not exist outputs mkdir outputs

echo.
echo   ============================================
echo     Alpha Titans chal raha hai
echo     Browser kholo:  http://127.0.0.1:8000
echo     Band karne ke liye: Ctrl+C
echo   ============================================
echo.

REM --- browser 3 second baad khud khul jayega ------------------
REM  ping ka istemaal "sleep" ke liye - `timeout` har shell pe nahi chalta
start "" /b cmd /c "ping -n 4 127.0.0.1 >nul & start http://127.0.0.1:8000"

python -m uvicorn main:app --app-dir "%~dp0server" --host 127.0.0.1 --port 8000
endlocal
