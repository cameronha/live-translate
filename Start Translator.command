#!/bin/bash
# Double-click this file to start the live translation overlay.
cd "$(dirname "$0")" || exit 1
exec ./venv/bin/python translate.py
