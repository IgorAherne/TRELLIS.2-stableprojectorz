@echo off
rem ---------------------------------------------------------------------------
rem Installs the Python packages and the models when this folder is not installed yet,
rem and again after update.bat. Called by the launchers, after environment.bat.
rem ---------------------------------------------------------------------------

set "CODE_DIR=%~dp0..\code"
set "INIT_FLAG=%CODE_DIR%\trellis2_init_done.txt"

if exist "%INIT_FLAG%" goto :installed

echo Performing first-time installation for Trellis.2; This may take several minutes. Please wait...
%PYTHON% -m pip install pip==%PIP_VERSION%
%PYTHON% "%CODE_DIR%\install.py"

rem Not "if errorlevel 1": a Python that crashed or was interrupted exits with a negative code
if %ERRORLEVEL% neq 0 (
    echo Installation failed. Please check the logs above for errors.
    exit /b 1
)
echo Installation successful. Creating initialization flag file.
echo Trellis.2 Init Complete > "%INIT_FLAG%"

:installed
rem Every model is on disk now, so the app does not ask HuggingFace at each start:
rem it starts without internet, and a change of the model repository cannot trigger a new download.
set HF_HUB_OFFLINE=1
exit /b 0
