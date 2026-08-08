#!/bin/sh
set -e
cd "$(dirname "$0")"
exec python3 serve_local.py
