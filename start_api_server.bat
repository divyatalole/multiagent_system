@echo off
cd /d D:\multiagent_system
call .venv\Scripts\activate.bat
echo Starting API Server...
python server_simple.py
pause

