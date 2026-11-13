#!/bin/sh
# Entry point for Dockerfile.api. Lambda's disk is read-only except /tmp, and the app writes reply
# caches into .cache/ and SQLite files into the Chroma index, so those are copied to /tmp/app and
# everything read-only is linked. A warm instance reuses /tmp/app; a new one starts fresh.
set -e
if [ ! -d /tmp/app ]; then
  mkdir -p /tmp/app/data
  cp -r /app/data/index /app/data/redacted /tmp/app/data/
  cp -r /app/.cache /tmp/app/.cache
  ln -s /app/models /tmp/app/models
  ln -s /app/results /tmp/app/results
  cp -r /opt/hf /tmp/hf
  chmod -R u+w /tmp/app /tmp/hf  # copies keep the image's read-only modes
fi
cd /tmp/app
# --proxy-headers: the rate limits key on the visitor's address from X-Forwarded-For.
exec python -m uvicorn callsearch.api.main:app --host 127.0.0.1 --port 8080 \
  --proxy-headers --forwarded-allow-ips '*'
