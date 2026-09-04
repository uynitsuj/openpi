# Source this to point AWS_CONFIG_FILE at a clone of ~/.aws/config with
# aggressive S3 transfer settings injected into the active profile.
# On the SZ box <-> us-west-2 link, 64 concurrent 8MB range-fetches measured
# ~6.3 MiB/s vs ~0.3 MiB/s with CLI defaults (2026-09-03 staging of
# siemens_simple_d405_v4). No-op (with a warning) if AWS_PROFILE is unset or
# the profile section is missing, so callers degrade to default settings.
aws_turbo_env() {
    # Always clone the REAL config, never a (possibly stale) AWS_CONFIG_FILE — a
    # prior turbo clone in /tmp may have been cleared, and chaining off it yields
    # an empty clone that breaks every aws call ("config profile could not be
    # found"). Drop any broken pre-set AWS_CONFIG_FILE first.
    local real="$HOME/.aws/config"
    local prof="${AWS_PROFILE:-}"
    local dst="${TMPDIR:-/tmp}/aws_turbo_config.$$"
    if [ -n "${AWS_CONFIG_FILE:-}" ] && [ ! -s "${AWS_CONFIG_FILE}" ]; then
        unset AWS_CONFIG_FILE
    fi
    if [ -z "$prof" ] || ! grep -q "^\[profile $prof\]" "$real" 2>/dev/null; then
        echo "[aws-turbo] WARN: profile '${prof:-<unset>}' not found in $real; keeping default transfer settings" >&2
        unset AWS_CONFIG_FILE
        return 0
    fi
    sed "/^\[profile $prof\]/a s3 =\n    max_concurrent_requests = 64\n    multipart_threshold = 8MB\n    multipart_chunksize = 8MB" \
        "$real" > "$dst"
    # Only trust the clone if it's non-empty and still carries the profile;
    # otherwise fall back to the default config so aws never breaks.
    if [ -s "$dst" ] && grep -q "^\[profile $prof\]" "$dst"; then
        export AWS_CONFIG_FILE="$dst"
    else
        echo "[aws-turbo] WARN: clone $dst invalid; keeping default transfer settings" >&2
        rm -f "$dst"
        unset AWS_CONFIG_FILE
    fi
}
aws_turbo_env
