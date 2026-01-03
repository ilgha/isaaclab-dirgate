@echo off
setlocal

set REPO_ROOT=%~dp0
set PYTHONPATH=%REPO_ROOT%src;%PYTHONPATH%

REM Adjust this if needed
set ISAACLAB_ROOT=C:\IsaacLab

REM %* forwards ALL arguments correctly
"%ISAACLAB_ROOT%\isaaclab.bat" -p %*

endlocal