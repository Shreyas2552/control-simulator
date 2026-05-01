@echo off
title Control Systems Simulator
cd /d "%~dp0"

echo ===================================================
echo   Interactive Control Systems Simulator
echo ===================================================
echo.

echo [1/2] Installing / verifying dependencies...
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo WARNING: pip install reported an error - attempting to start anyway.
    echo.
)

echo [2/2] Launching simulator...
echo.
echo  Keep this window open while using the app.
echo  Browser will open automatically at http://localhost:8501
echo  Press Ctrl+C in this window to stop the server.
echo.

streamlit run app.py

echo.
echo Simulator stopped.
pause
