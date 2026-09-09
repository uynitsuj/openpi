# SZ box ↔ S3 transfer playbook (xdof-dgx01)

How to reliably move data between the SZ training box (`xdof-dgx01`, 8×H100) and
S3, where the network is flaky/throttled and naive `aws s3` commands stall or crawl.
Written 2026-09-10 after a long debugging session staging the siemens datasets +
uploading pi0.5 checkpoints. See also the `sz-box-network` project memory.

## TL;DR
1. **Crank concurrency first.** `~/.aws/config` can silently pin
   `max_concurrent_requests` low (found it at **3** — a ~20x throttle). Set it to
   **64** in *both* `[default]` and the active profile's `s3` block. This alone
   fixed most "S3 is slow / hangs" symptoms.
2. **Pick the fast path empirically.** Direct vs the SZ proxy vary by time of day
   and can each be dead. Timed-test all three before a big transfer.
3. **Downloads:** proxy is usually fast; for a single *large* file over a flaky
   link use chunked byte-range GETs (never let `aws s3 cp` restart-from-zero loop).
4. **Uploads (45 GB checkpoints):** 64-way concurrency + short-timeout retry loop;
   gate on file *count*, not byte sums.
5. **NFS (`/nfs_exp`) is authoritative for checkpoints.** S3 upload is best-effort.

## 1. The two paths — direct vs SZ proxy (test both!)
- **Direct** (strip proxy): `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy`.
  Strongly **diurnal**: daytime/evening can be ~11 KB/s (effectively dead) up to
  ~9 MB/s; peaks ~**175 MB/s around local midnight**.
- **SZ office proxy**: `export HTTPS_PROXY=http://xdof-sz-tcl-proxy.local:10080`
  (alt: `xdof-sz-tcl-proxy-1.local:10080`). Often much faster than direct when
  direct is slow. Both proxies can pass a probe yet still stall a transfer.
- **Always timed-test all three** right before a big transfer — they can be equal
  or one can be dead:
  ```bash
  head -c 314572800 /dev/urandom > /tmp/uptest.bin   # 300MB
  for P in DIRECT "http://xdof-sz-tcl-proxy.local:10080" "http://xdof-sz-tcl-proxy-1.local:10080"; do
    [ "$P" = DIRECT ] && unset HTTPS_PROXY || export HTTPS_PROXY="$P"
    time aws s3 cp /tmp/uptest.bin s3://<bucket>/tmp/t.bin   # compare
  done
  ```
- `lab42/tools/traj_tool/sz_traj.py` auto-selects a working proxy (probes real S3).

## 2. Concurrency = "many workers" (the #1 lever)
- Check it: `grep -A4 -E '\[default\]|\[profile' ~/.aws/config` — look for an
  `s3 =` block with `max_concurrent_requests`.
- **Set high in BOTH `[default]` and the active profile** (aws falls back to
  `[default]` if the profile's block is missing/ignored):
  ```
  [default]
  s3 =
      max_concurrent_requests = 64
      multipart_threshold = 32MB
      multipart_chunksize = 32MB
  [profile EngineerAccess-266735817792]
  s3 =
      max_concurrent_requests = 64
      multipart_threshold = 32MB
      multipart_chunksize = 32MB
  ```
- Verify workers are actually parallel: `ss -tn | grep -c ESTAB` → ~100 conns
  during a transfer (not ~3).
- With low concurrency, **one stalled chunk hangs the whole transfer** (symptom:
  the `aws` proc is state `S`, 0 CPU, 0 disk I/O for minutes — that's *hung*, not
  slow). 64-way concurrency makes a single stalled chunk non-fatal.

## 3. Downloads
- **Datasets (many files):** `aws s3 sync` — it **resumes** (skips completed
  files), so it's safe to wrap in a retry loop. Via proxy + 64-way this staged
  ~8 GB LeRobot datasets in ~90 s.
- **A single large file (e.g. an 849 MB parquet) over a flaky link:** `aws s3 cp`
  has **no intra-file resume — it restarts from zero on any interruption.** If the
  link resets often (or you keep killing it), it never finishes. Two fixes:
  - high concurrency + a *long, uninterrupted* timeout (don't kill it), or
  - **chunked byte-range GETs**, each retried independently, then concatenate —
    small chunks complete between resets:
    ```bash
    aws s3api get-object --bucket B --key K --range "bytes=$start-$end" part.NNNN
    # ...loop 32MB chunks (8 parallel), then: cat part.* > file
    ```
- Interrupted multipart downloads leave temp files. Clean before verifying:
  `find "$DIR" \( -name '*.parquet.*' -o -name '*.mp4.*' \) -delete`.

## 4. Uploads (pi0.5 checkpoints ~45 GB)
- Large multipart **uploads** can stall through the proxy even when downloads are
  fine — 64-way concurrency mitigates. At ~9 MB/s a 45 GB checkpoint ≈ **85 min**.
- Use short per-attempt timeouts in a retry loop so a stalled connection recycles
  fast; `aws s3 sync` resumes across files each pass:
  ```bash
  for r in $(seq 1 60); do
    timeout 400 aws s3 sync "$CK" "$S3" --cli-read-timeout 60 --only-show-errors
    got=$(aws s3 ls "$S3/" --recursive | grep -c "/")   # gate on COUNT
    [ "$got" -ge "$WANT_FILES" ] && break; sleep 6
  done
  ```
- **Gate completeness on file COUNT, not byte totals** — `awk '{s+=$3} END{printf "%d",s}'`
  overflows to `2147483647` (2^31-1) on ~45 GB (32-bit `%d`), giving false "done".
- `aws s3 ls` only lists a multipart object once it *fully* completes, so a big
  file mid-upload shows no byte progress — don't mistake that for a stall; check
  the `aws` proc's CPU/disk I/O instead.

## 5. AWS SSO auth on this box
- `aws sso login` → 1 h access token that **auto-refreshes** (refresh token) for
  the IdP session (~8 h). Session expiry is a **hard, interactive re-login** —
  cannot be scripted. Plan long jobs around it.
- Misleading: `aws sts`/`aws s3` can keep working off cached role creds after the
  SSO token expires, while `sky jobs launch` (needs a fresh token) fails —
  **`sky` is the truth for auth health**, not `aws sts`.
- **Checkpoints: `/nfs_exp` is authoritative** (`--keep-period` keeps locals);
  S3 sync is best-effort. If uploads fail on expired token, re-auth and bulk-sync
  from NFS later — never lose a checkpoint to a failed upload.

## 6. Standard env preamble
```bash
# direct:
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy
# or proxy:
export HTTPS_PROXY=http://xdof-sz-tcl-proxy.local:10080 HTTP_PROXY=http://xdof-sz-tcl-proxy.local:10080
export no_proxy=127.0.0.1,localhost
unset AWS_CONFIG_FILE                       # NEVER set it to "" — that breaks profile lookup
export AWS_PROFILE=EngineerAccess-266735817792
```
Note: `AWS_CONFIG_FILE=` (empty) causes `The config profile could not be found` —
always `unset` it, don't assign empty.

## 7. Architecture note
Bulk data work (>10 GB) is best run in a **us-west-2 SkyPilot job next to the
buckets** (fast intra-AWS), not over the SZ link. Conversions already do this
(`sky/convert_siemens_*.yaml`). But training must run on dgx01 (the GPUs), so
datasets have to come *down* over the SZ link — that's when this playbook applies.
Nightly (~midnight local) is the cheapest window for big SZ↔S3 transfers.
