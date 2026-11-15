@echo off
call "%~dp0environment.bat"
if %ERRORLEVEL% neq 0 goto :not_ready
call "%~dp0ensure-installed.bat"
if %ERRORLEVEL% neq 0 goto :not_ready

echo --------------------------------------------------
echo Virtual Env: %VIRTUAL_ENV%
echo --------------------------------------------------

rem Change directory to the 'code' folder
cd /d "%~dp0..\code"

echo Starting the StableProjectorz API server, please wait...

rem Forward all arguments (%*) to the application script
%PYTHON% .\api_spz\main_api.py %*

if %ERRORLEVEL% neq 0 (
    echo Something went wrong during execution.
    echo If errors persist, delete the file 'code\trellis2_init_done.txt' and launch again: the installer then re-checks every package.
    echo If that does not help, also delete the 'code\venv' folder to install from scratch.
)

pause
exit /b

:not_ready
echo The app was not started. Please check the messages above.
pause
exit /b 1
