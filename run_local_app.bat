@echo off
cd /d %~dp0src
echo Starting IntelliBridge... a browser tab will open. Keep this window open. Press Ctrl+C to stop.
python intellibridge_app.py
pause
