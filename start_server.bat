@echo off
rem VideoGen server watchdog: restarts uvicorn whenever it exits.
rem Run this in a terminal (or double-click). Stop with Ctrl+C / closing the window.
cd /d "%~dp0"
:loop
echo [%date% %time%] starting VideoGen server on http://localhost:8080 >> server.log
python -m uvicorn app:app --host 127.0.0.1 --port 8080 >> server.log 2>&1
echo [%date% %time%] server exited (code %errorlevel%), restarting in 3s... >> server.log
timeout /t 3 /nobreak > nul
goto loop
