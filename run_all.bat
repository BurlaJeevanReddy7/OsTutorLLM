@echo off
echo ===================================================
echo [1/3] Starting Ollama Background Daemon...
echo ===================================================
start "Ollama Service" /min cmd /c "ollama serve"

timeout /t 3 /nobreak >nul

echo ===================================================
echo [2/3] Pre-loading and Warming Up Qwen 1.5B...
echo ===================================================
ollama run qwen2.5:1.5b ""

timeout /t 2 /nobreak >nul

echo ===================================================
echo [3/3] Launching OSTutorLLM FastAPI Server...
echo ===================================================
start "OSTutorLLM Backend" cmd /k "python main.py"

echo.
echo ===================================================
echo All systems running! Opening web interface...
echo ===================================================
timeout /t 2 /nobreak >nul
start http://127.0.0.1:8000