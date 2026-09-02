# Source this to point AWS_CONFIG_FILE at a clone of ~/.aws/config with
# aggressive S3 transfer settings injected into the active profile.
# On the SZ box <-> us-west-2 link, 64 concurrent 8MB range-fetches measured
# ~6.3 MiB/s vs ~0.3 MiB/s with CLI defaults (2026-09-03 staging of
# siemens_simple_d405_v4). No-op (with a warning) if AWS_PROFILE is unset or
# the profile section is missing, so callers degrade to default settings.
aws_turbo_env() {
    local src="${AWS_CONFIG_FILE:-$HOME/.aws/config}"
    local prof="${AWS_PROFILE:-}"
    local dst="${TMPDIR:-/tmp}/aws_turbo_config.$$"
    if [ -z "$prof" ] || ! grep -q "^\[profile $prof\]" "$src" 2>/dev/null; then
        echo "[aws-turbo] WARN: profile '${prof:-<unset>}' not found in $src; keeping default transfer settings" >&2
        return 0
    fi
    sed "/^\[profile $prof\]/a s3 =\n    max_concurrent_requests = 64\n    multipart_threshold = 8MB\n    multipart_chunksize = 8MB" \
        "$src" > "$dst"
    export AWS_CONFIG_FILE="$dst"
}
aws_turbo_env
