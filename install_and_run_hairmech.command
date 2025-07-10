#!/usr/bin/env bash
echo "=== Hair-mech GUI bootstrap ==="
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade hairmech[ui]
echo
echo "Launching GUI..."
hairmech-gui
