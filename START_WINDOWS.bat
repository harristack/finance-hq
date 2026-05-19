@echo off
echo.
echo  Finance HQ -- Starting...
echo.

REM Install dependencies (first time only)
pip install -r requirements.txt --quiet

REM Kill anything on port 5051
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5051 ^| findstr LISTENING') do taskkill /f /pid %%a 2>nul

REM Start the backend
start "Curve Engine" python curve_engine.py

timeout /t 2 /nobreak >nul

REM Open the dashboard
start chrome "command-hq.html"

echo  Done. Dashboard opened in Chrome.
echo  Close this window to keep running, or Ctrl+C to stop.
pause
