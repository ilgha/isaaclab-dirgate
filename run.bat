@echo off
setlocal
set REPO_ROOT=%~dp0
set PYTHONPATH=%REPO_ROOT%src;%PYTHONPATH%

set ISAACLAB_ROOT=C:\IsaacLab

"%ISAACLAB_ROOT%\isaaclab.bat" -p "%REPO_ROOT%%*"
endlocal