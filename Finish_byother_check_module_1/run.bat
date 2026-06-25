@echo off
cd /d "%~dp0"
set OPENROUTER_API_KEY=YOUR_API_KEY
set PYTHONIOENCODING=utf-8
python "main.py"
pause
