@echo off
title LUCIA - Project Dependencies Installer

echo ==========================================
echo  LUCIA - Project Dependencies Installer
echo ==========================================
echo.

:: Check if Python is already installed in PATH
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo [INFO] Python is already installed.
    python --version
    goto :install_deps
)

:: Check if Python Launcher (py) is installed
py --version >nul 2>&1
if %errorlevel% == 0 (
    echo [INFO] Python Launcher is installed.
    py --version
    goto :install_deps_py
)

echo [INFO] Python not found. Downloading Python 3.12.4...
echo [INFO] Please ensure you have an active internet connection.
echo.

:: Download Python 3.12.4 64-bit installer
set PYTHON_INSTALLER=python-3.12.4-amd64.exe
set PYTHON_URL=https://www.python.org/ftp/python/3.12.4/python-3.12.4-amd64.exe

echo [INFO] Downloading from python.org...
curl -L -o "%PYTHON_INSTALLER%" "%PYTHON_URL%"

if not exist "%PYTHON_INSTALLER%" (
    echo [ERROR] Download failed. Please check your internet connection.
    pause
    exit /b 1
)

echo [INFO] Installing Python silently...
echo [INFO] This may take a few minutes. Please wait...
"%PYTHON_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_test=0

:: Wait for installation to finish and register in PATH
timeout /t 15 /nobreak >nul
del "%PYTHON_INSTALLER%"

echo [INFO] Python installed successfully.
echo.

:: Update PATH for the current session
set "PATH=C:\Program Files\Python312;C:\Program Files\Python312\Scripts;%PATH%"

:install_deps
echo [INFO] Installing project dependencies via pip...
pip install streamlit opencv-python numpy
goto :end

:install_deps_py
echo [INFO] Installing project dependencies via Python Launcher...
py -m pip install streamlit opencv-python numpy
goto :end

:end
echo.
echo ==========================================
echo  Installation completed successfully!
echo ==========================================
pause