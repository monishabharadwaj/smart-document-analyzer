@echo off
echo =========================================
echo  Smart Document Analyzer - Full Stack
echo =========================================
echo.

echo Current directory: %CD%
echo.

echo [1/3] Starting Backend Server...
echo.
start "Smart Document Analyzer - Backend" cmd /k "cd /d %CD% && echo Starting backend... && python api_server.py"

echo Waiting 5 seconds for backend to initialize...
timeout /t 5 /nobreak > nul

echo [2/3] Installing Frontend Dependencies (if needed)...
cd frontend
if not exist node_modules (
    echo Installing npm packages...
    npm install
) else (
    echo Frontend dependencies already installed.
)

echo.
echo [3/3] Starting Frontend Development Server...
echo.
start "Smart Document Analyzer - Frontend" cmd /k "cd /d %CD% && echo Starting frontend... && npm run dev"

echo.
echo =========================================
echo  🚀 Application Started Successfully!
echo =========================================
echo.
echo 📊 Backend API:     http://localhost:8000
echo 🌐 Frontend App:    http://localhost:5173  
echo 📖 API Docs:       http://localhost:8000/docs
echo.
echo 💡 Tips:
echo - Upload .txt files from frontend/sample-documents/
echo - Test all features: Upload → Classify → Entities → Summarize → Search
echo - Check API documentation for technical details
echo.
echo Both servers are running in separate windows.
echo Press any key to close this launcher...
pause > nul
