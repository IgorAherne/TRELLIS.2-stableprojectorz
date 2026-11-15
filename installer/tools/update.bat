@echo off
setlocal
rem ---------------------------------------------------------------------------
rem Updates the "code" folder to the newest version from GitHub, with the portable Git of tools\git.
rem The models and the Python packages stay: the next launch re-checks the packages and installs
rem only what changed.
rem
rem Keep this file plain ASCII with Windows line endings, and never put brackets into an echo
rem that sits inside an if-block: cmd stops the whole script with "... was unexpected at this time".
rem ---------------------------------------------------------------------------

set "PORTABLE_GIT_BASE=%~dp0git"
set "CODE_DIR=%~dp0..\code"
set "INIT_FLAG_FILE=%CODE_DIR%\trellis2_init_done.txt"
set "REPO_URL=https://github.com/IgorAherne/TRELLIS.2-stableprojectorz.git"

set "PATH=%PORTABLE_GIT_BASE%\mingw64\bin;%PORTABLE_GIT_BASE%\cmd;%PORTABLE_GIT_BASE%\usr\bin;%PORTABLE_GIT_BASE%\mingw64\libexec\git-core;%PATH%"
set "GIT_SSL_CAINFO=%PORTABLE_GIT_BASE%\mingw64\etc\ssl\certs\ca-bundle.crt"
set "SSL_CERT_FILE=%PORTABLE_GIT_BASE%\mingw64\etc\ssl\certs\ca-bundle.crt"

rem Settings for this run only, nothing is written to the user's own Git configuration:
rem  - safe.directory: Git refuses folders on exFAT or external drives, whose owner it cannot read
rem  - core.longpaths: paths over 260 characters
rem  - core.autocrlf:  keep the files byte for byte as they are in the repository
set GIT_CONFIG_COUNT=3
set GIT_CONFIG_KEY_0=safe.directory
set GIT_CONFIG_VALUE_0=*
set GIT_CONFIG_KEY_1=core.longpaths
set GIT_CONFIG_VALUE_1=true
set GIT_CONFIG_KEY_2=core.autocrlf
set GIT_CONFIG_VALUE_2=false

echo Starting the update. Close any running Trellis window first.

if not exist "%PORTABLE_GIT_BASE%\cmd\git.exe" (
    echo ERROR: the portable Git is missing from the tools\git folder.
    goto :fail
)
if not exist "%CODE_DIR%\" (
    echo ERROR: the code folder was not found next to the tools folder.
    goto :fail
)

cd /d "%CODE_DIR%"

if not exist ".git" (
    echo Connecting the code folder to the repository...
    git init -q
    if errorlevel 1 goto :fail
)

git remote remove origin >nul 2>&1
git remote add origin "%REPO_URL%"
if errorlevel 1 goto :fail

echo Downloading the newest code...
git fetch --depth=1 origin main
if errorlevel 1 (
    echo.
    echo ERROR: the download failed. Check the internet connection and try again.
    echo Nothing was changed.
    goto :fail
)

rem Files of the repository are replaced by their newest version. Everything else in the folder is
rem left alone: the models, the venv, the flash_attn wheel, your generated files.
git checkout -q -f -B main FETCH_HEAD
if errorlevel 1 (
    echo.
    echo ERROR: the new files could not be written. Close the Trellis windows and try again.
    goto :fail
)

rem Without this file the next launch runs the installer, which brings the Python packages
rem to the versions the new code was tested with. Packages already at the right version are kept.
if exist "%INIT_FLAG_FILE%" del "%INIT_FLAG_FILE%"

rem The launchers of this tools folder come with the code. This is one block on purpose, and the last
rem thing that runs: cmd has read all of it before update.bat itself is replaced.
(
    if exist "%CODE_DIR%\installer\tools\*.bat" copy /y "%CODE_DIR%\installer\tools\*.bat" "%~dp0" >nul
    echo.
    echo -----------------------------------------------------------------------------
    echo Update completed.
    echo The next launch re-checks the Python packages and installs what changed.
    echo -----------------------------------------------------------------------------
    if not defined SKIP_PAUSE pause
    exit /b 0
)

:fail
echo.
echo The update did not complete.
if not defined SKIP_PAUSE pause
exit /b 1
