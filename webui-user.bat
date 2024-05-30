@echo off
set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=

set "current_dir=%~dp0"
cd /d "%current_dir%"

call webui.bat