#!/usr/bin/env bash

cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
exec uvicorn app.server:app --reload --host 0.0.0.0 --port 8000