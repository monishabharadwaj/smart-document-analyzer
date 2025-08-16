@echo off
echo ========================================
echo  Smart Document Analyzer - Quick Start
echo ========================================
echo.

echo Starting Backend Server...
echo.
start "Backend Server" cmd /k "cd /d C:\Users\Hp\smart-document-analyzer && python api_server.py"

echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend Server...
echo.
start "Frontend Server" cmd /k "cd /d \"C:\Users\Hp\OneDrive\Desktop\frontend dev\project\" && npm run dev"

echo.
echo ========================================
echo  Application Starting!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo API Docs: http://localhost:8000/docs
echo.
echo Both servers are starting in separate windows.
echo Close this window when done testing.
echo.
pause
