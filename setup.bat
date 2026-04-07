@echo off
REM setup.bat — Unified System first-time setup for Windows
REM Run from the unified_system folder:  setup.bat

echo.
echo =====================================================
echo   Unified AI Face + Cinematic Hand FX -- Setup
echo =====================================================
echo.

pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Make sure Python is in PATH.
    pause
    exit /b 1
)

echo.
echo [Setup] Building face embeddings...
python scripts\build_dataset.py
if errorlevel 1 (
    echo [WARNING] build_dataset.py failed - check data\faces\ has images.
)

echo.
echo [Setup] Done!
echo.
echo Run with:
echo   python main.py
echo   python main.py --camera 1 --threshold 0.68 --debug
echo.
pause
