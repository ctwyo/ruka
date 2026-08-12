@echo off
setlocal enabledelayedexpansion
set VENV=last

rem ── health check: does the venv's python actually run? ─────────────────
"%VENV%\Scripts\python.exe" -c "import sys" >nul 2>&1
if not errorlevel 1 goto :run

echo [venv] broken (base Python moved, reinstalled, or project copied) - repairing...

rem ── locate a Python 3.14 base interpreter, trying several sources ──────
rem prints "<dir>|<major.minor.patch>" only if the interpreter is 3.14.x
set "FINDPY=import sys,os;v=sys.version_info;print(os.path.dirname(sys.executable)+'|'+'.'.join(map(str,v[:3]))) if v[:2]==(3,14) else None"

set "PYHOME="
set "PYVER="
rem 1) py launcher, explicit 3.14
call :probe py -3.14
rem 2) plain python on PATH (verifies it is 3.14)
if not defined PYHOME call :probe python
rem 3) py launcher default
if not defined PYHOME call :probe py

if not defined PYHOME (
    echo [venv] Python 3.14 not found on this machine.
    echo        Install Python 3.14 from python.org, then run this again.
    echo        Tip: check with  python --version   and   py -0p
    pause
    exit /b 1
)

echo [venv] found Python %PYVER% at: !PYHOME!
(
    echo home = !PYHOME!
    echo include-system-site-packages = false
    echo version = !PYVER!
    echo executable = !PYHOME!\python.exe
    echo command = !PYHOME!\python.exe -m venv %CD%\%VENV%
) > "%VENV%\pyvenv.cfg"

rem verify the repair worked
"%VENV%\Scripts\python.exe" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo [venv] repair failed - recreate the venv manually:
    echo        py -3.14 -m venv %VENV%  ^&^&  %VENV%\Scripts\activate  ^&^&  pip install -r requirements.txt
    pause
    exit /b 1
)
echo [venv] repaired OK.

:run
call %VENV%\Scripts\activate.bat
python servertasks.py
goto :eof

rem ── :probe <launcher> [args] -> sets PYHOME/PYVER if it is Python 3.14 ─
:probe
for /f "tokens=1,2 delims=|" %%A in ('%* -c "%FINDPY%" 2^>nul') do (
    set "PYHOME=%%A"
    set "PYVER=%%B"
)
goto :eof
