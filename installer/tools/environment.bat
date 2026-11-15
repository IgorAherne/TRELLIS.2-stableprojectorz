@echo off

set DIR=%~dp0.
set PORTABLE_PYTHON=%DIR%\python\python.exe
set BASE_PATH=%DIR%\git\mingw64\bin;%DIR%\python;%DIR%\python\Scripts
set PATH=%BASE_PATH%;%PATH%
set VENV_PATH=%~dp0.\..\code\venv
:: Define site-packages path early for cleanup logic
set SITE_PACKAGES=%DIR%\python\Lib\site-packages

:: Prevent setuptools distutils hack from conflicting with portable Python
set SETUPTOOLS_USE_DISTUTILS=stdlib

:: Force typical Windows appdata variables if not defined.
:: important to avoid errors with  winreg.QueryValueEx(key, shell_folder_name)
if not defined APPDATA       set APPDATA=%USERPROFILE%\AppData\Roaming
if not defined LOCALAPPDATA  set LOCALAPPDATA=%USERPROFILE%\AppData\Local
if not defined PROGRAMDATA   set PROGRAMDATA=C:\ProgramData

rem Check critical DLLs (Assuming Python 3.11 dependencies)
if not exist "%DIR%\python\vcruntime140.dll" (
    echo Error: Missing vcruntime140.dll
    exit /b 1
)
if not exist "%DIR%\python\vcruntime140_1.dll" (
    echo Error: Missing vcruntime140_1.dll
    exit /b 1
)
if not exist "%DIR%\python\_ctypes.pyd" (
    echo Error: Missing _ctypes.pyd
    exit /b 1
)
if not exist "%DIR%\python\libffi-8.dll" (
    echo Error: Missing libffi-8.dll
    exit /b 1
)
if not exist "%DIR%\python\libcrypto-3.dll" (
    echo Error: Missing libcrypto-3.dll
    exit /b 1
)
if not exist "%DIR%\python\libssl-3.dll" (
    echo Error: Missing libssl-3.dll
    exit /b 1
)
if not exist "%DIR%\python\_ssl.pyd" (
    echo Error: Missing _ssl.pyd
    exit /b 1
)


:: Prevent Python from using user site packages (APPDATA)
set PYTHONNOUSERSITE=1

rem Clear any existing Python/Conda related env vars
set CONDA_PREFIX=
set CONDA_PYTHON_EXE=
set CONDA_DEFAULT_ENV=
set CONDA_SHLVL=
set CONDAPATH=
set PYTHONPATH=
set PYTHONHOME=

:: First set basic environment variables
set PIP_INSTALLER_LOCATION=%DIR%\python\get-pip.py
set PIP_DISABLE_PIP_VERSION_CHECK=1
::so that pip never tries to read from system config that triggers the registry call:
set PIP_CONFIG_FILE=NUL


rem Exact versions of the tools that build the virtual environment. Unpinned, a new release of any
rem of them can break every fresh install overnight.
set PIP_VERSION=26.3
set VIRTUALENV_PACKAGES=virtualenv==21.15.6 distlib==0.4.4 filelock==3.25.2 platformdirs==4.14.5 python-discovery==1.9.5 packaging==26.3

rem Windows without Developer Mode cannot make symlinks; the HuggingFace hub then prints a long warning
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

rem The models are kept inside this folder, whichever bat starts the app
set "HF_HOME=%~dp0..\code\models"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"

echo Setting up environment...
echo Portable Python: "%PORTABLE_PYTHON%"

:: A venv records the full path of the Python it was made from, so it cannot start any more after this
:: folder was moved or renamed. It is then made again below, over the old one: its packages stay.
set VENV_READY=
if exist "%VENV_PATH%\Scripts\activate.bat" "%PORTABLE_PYTHON%" -c "import sys; sys.exit(('home = ' + sys.prefix + '\n').lower() not in open(sys.argv[1], encoding='utf-8').read().lower())" "%VENV_PATH%\pyvenv.cfg" >nul 2>&1 && set VENV_READY=1

:: create venv if it doesn't exist, or belongs to another folder
if not defined VENV_READY (

    :: Manually remove potentially broken pip and virtualenv from portable python's site-packages
    echo Cleaning packages from portable Python's site-packages...
    rmdir /s /q "%SITE_PACKAGES%\pip" >nul 2>&1
    rmdir /s /q "%SITE_PACKAGES%\virtualenv" >nul 2>&1

    :: Use FOR /D to reliably find and remove directories matching a wildcard
    for /d %%d in ("%SITE_PACKAGES%\pip-*.dist-info") do rmdir /s /q "%%d"
    for /d %%d in ("%SITE_PACKAGES%\virtualenv-*.dist-info") do rmdir /s /q "%%d"

    echo Cleanup finished.

    :: Attempt to install pip using get-pip.py (ensurepip is often missing in embeddable zip)
    echo Bootstrapping pip using get-pip.py...
    "%PORTABLE_PYTHON%" "%DIR%\python\get-pip.py" "pip==%PIP_VERSION%"
    if errorlevel 1 (
        echo ERROR: get-pip.py failed to install pip in portable Python even after cleanup.
        echo Please check the integrity of the tools\python directory.
        pause
        exit /b 1
    )


    :: Install virtualenv package if missing, using the portable python
    "%PORTABLE_PYTHON%" -m pip show virtualenv >nul 2>&1
    if errorlevel 1 (
        echo Installing virtualenv using portable Python...
        "%PORTABLE_PYTHON%" -m pip install %VIRTUALENV_PACKAGES%
        if errorlevel 1 (
            echo Failed to install virtualenv using portable Python.
            pause
            exit /b 1
        )
    )

    :: Verify portable python is available before trying to create venv
    "%PORTABLE_PYTHON%" --version >nul 2>&1
    if errorlevel 1 (
        echo Portable Python "%PORTABLE_PYTHON%" not found or failed.
        pause
        exit /b 1
    )
    echo Creating virtual environment...
    rem embed + no periodic update: the pip and setuptools this virtualenv version carries,
    rem never a newer one it found on the internet during an earlier run
    "%PORTABLE_PYTHON%" -m virtualenv --no-periodic-update --pip embed --setuptools embed "%VENV_PATH%"
    if errorlevel 1 (
        echo Failed to create virtual environment using portable Python.
        pause
        exit /b 1
    )
)


:: Always ensure our pip.ini is in place if it exists in the source
if exist "%DIR%\python\pip.ini" (
    copy /Y "%DIR%\python\pip.ini" "%VENV_PATH%\pip.ini" >nul 2>&1
)

:: activate venv
call "%VENV_PATH%\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)

:: Set the PYTHON variable to the venv python executable
set PYTHON="%VENV_PATH%\Scripts\python.exe"

echo Virtual environment activated. Python: %PYTHON%