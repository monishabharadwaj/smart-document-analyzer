@echo off
echo Starting Smart Document Analyzer...
echo.

echo [1/2] Starting Backend Server...
start "Backend" cmd /k "python api_server.py"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo [2/2] Starting Frontend Server...
start "Frontend" cmd /k "cd frontend && npm run dev"

echo Waiting for frontend to start...
timeout /t 8 /nobreak > nul

echo [3/3] Opening website in browser...
start http://localhost:5173

echo.
echo ✅ Application started successfully!
echo.
echo 🌐 Frontend: http://localhost:5173
echo 📊 Backend:  http://localhost:8000
echo 📖 API Docs: http://localhost:8000/docs
echo.
echo The website should open automatically.
echo If not, click the frontend link above.
echo.
pause
