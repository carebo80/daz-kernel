#!/bin/bash

cd ~/dev/daz_kernel
source .venv/bin/activate

# Tailwind im Hintergrund
npx @tailwindcss/cli -i ./assets/css/app.css -o ./static/css/app.css --watch &

# Server starten
uvicorn app.server:app --reload --host 0.0.0.0 --port 8000