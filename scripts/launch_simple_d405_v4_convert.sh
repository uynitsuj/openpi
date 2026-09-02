#!/usr/bin/env bash
# Retry the siemens_simple_d405_v4 conversion launch until SSO allows it.
set -u
cd /home/karim/openpi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY socks_proxy SOCKS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost"

log() { echo "[v4-launch $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

tries=0
until uv run sky jobs launch sky/convert_siemens_simple_d405_v4.yaml -n siemens-simple-d405-v4-convert -d --yes; do
    tries=$((tries + 1))
    [ "$tries" -ge 300 ] && { log "ERROR: giving up after 300 tries"; exit 1; }
    log "AUTH_WAIT: launch failed (run 'aws sso login'); retry $tries in 120s"
    sleep 120
done
log "LAUNCHED siemens-simple-d405-v4-convert"
