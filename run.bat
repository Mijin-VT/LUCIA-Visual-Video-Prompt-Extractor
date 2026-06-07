@echo off
echo ==========================================
echo  Starting LUCIA - Visual Video Prompt Extractor
echo  (Local GGUF + CUDA Mode)
echo ==========================================
echo.
echo The application will use llama.cpp binaries (CUDA 13.3)
echo and .gguf models from the 'models' and 'uncensored' folders.
echo.
echo ==========================================
streamlit run app.py --server.headless true
pause
