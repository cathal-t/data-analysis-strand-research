@echo off
echo === Hair-mech GUI bootstrap ===
python -m pip install --upgrade pip
python -m pip install --upgrade hairmech[ui]
echo.
echo Launching GUI...
hairmech-gui
pause
