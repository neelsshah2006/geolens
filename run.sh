#!/usr/bin/env bash
# Start GeoLens at http://localhost:8000
set -e
cd "$(dirname "$0")/backend"
exec python3 -m uvicorn geolens.main:app --host 127.0.0.1 --port "${PORT:-8000}"
