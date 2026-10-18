#!/bin/sh

set -eu

APP_URL=${APP_URL:-http://127.0.0.1:7860}

if ! command -v cloudflared >/dev/null 2>&1; then
    echo "cloudflared is not installed."
    echo "Run: brew install cloudflared"
    exit 1
fi

if ! curl --fail --silent "$APP_URL/health" >/dev/null; then
    echo "The application is not ready at $APP_URL."
    echo "Start it first with: ./scripts/run_demo_server.sh"
    exit 1
fi

if [ -n "${TUNNEL_ALLOWED_EMAIL:-}" ]; then
    exec cloudflared tunnel \
        --url "$APP_URL" \
        --allowed-mail "$TUNNEL_ALLOWED_EMAIL"
fi

exec cloudflared tunnel --url "$APP_URL"
