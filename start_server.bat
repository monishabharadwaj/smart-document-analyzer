@echo off
title Smart Document Analyzer - API Server
echo ============================================
echo   Smart Document Analyzer - API Server
echo ============================================
echo.
echo Starting API server...
echo Server will be available at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.
cd /d "%~dp0"
python api_server.py
pause
