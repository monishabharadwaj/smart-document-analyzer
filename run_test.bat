@echo off
title Smart Document Analyzer - System Test
echo ============================================
echo   Smart Document Analyzer - System Test
echo ============================================
echo.
echo Running comprehensive system test...
echo.
cd /d "%~dp0"
python final_test.py
echo.
echo Test completed!
pause
