#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}}"
REPROMO_ROOT="${REPROMO_ROOT:-/home/karim/RORM}"
SCRIPT="${SCRIPT:-scripts/data/write_repromo_annotations.py}"
S3_ROOT="${S3_ROOT:-s3://xdof-internal-research/repromo/datasets}"
PY="${PY:-}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"

RUNS=(
  "sim_hang_the_mug_on_the_mug_rack_30hz_gop10:2026-06-17_hang_mug_rabc"
  "sim_load_the_plates_into_the_dish_rack_30hz_gop10:2026-06-17_load_plates_rabc"
  "sim_put_the_plastic_bottles_in_the_bin_30hz_gop10:2026-06-17_put_bottles_rabc"
  "sim_throw_plastic_bottles_in_bin_30hz_gop10:2026-06-17_throw_bottles_rabc"
  "sim_turn_the_mug_right_side_up_30hz_gop10:2026-06-17_turn_mug_rabc"
  "sim_sweep_away_paper_scraps_from_the_table_30hz_gop10:2026-06-17_sweep_paper_rabc"
)

run_repromo() {
  if [[ -n "${PY}" ]]; then
    "${PY}" "${REPROMO_ROOT}/${SCRIPT}" "$@"
  else
    (cd "${REPROMO_ROOT}" && uv run "${SCRIPT}" "$@")
  fi
}

for entry in "${RUNS[@]}"; do
  dataset="${entry%%:*}"
  version="${entry#*:}"
  dataset_dir="${DATA_ROOT}/${dataset}"
  sidecar_dir="${dataset_dir}/repromo_annotations/${version}"

  echo
  echo "=== ${dataset} (${version}) ==="

  mkdir -p "${DATA_ROOT}"
  if [[ -L "${dataset_dir}" ]]; then
    echo "[info] Replacing cache symlink with a writable local copy: ${dataset_dir}"
    rm "${dataset_dir}"
  fi

  echo "[sync] ${S3_ROOT}/${dataset} -> ${dataset_dir}"
  aws s3 sync "${S3_ROOT}/${dataset}" "${dataset_dir}"

  if [[ ! -w "${dataset_dir}/meta/info.json" ]]; then
    echo "[error] Dataset is not writable by $(whoami): ${dataset_dir}" >&2
    echo "        Fix with: sudo chown -R $(id -u):$(id -g) '${dataset_dir}'" >&2
    echo "        Or set DATA_ROOT to a writable LeRobot dataset directory." >&2
    exit 1
  fi

  if [[ ! -f "${sidecar_dir}/dense_predictions.parquet" ]]; then
    echo "[sync] ${S3_ROOT}/${dataset}/repromo_annotations/${version} -> ${sidecar_dir}"
    aws s3 sync "${S3_ROOT}/${dataset}/repromo_annotations/${version}" "${sidecar_dir}"
  fi

  if [[ ! -f "${sidecar_dir}/dense_predictions.parquet" ]]; then
    echo "[error] Missing dense_predictions.parquet after sync: ${sidecar_dir}" >&2
    exit 1
  fi

  args=(
    --mode inject
    --lerobot-repo "${dataset_dir}"
    --sidecar-dir "${sidecar_dir}"
  )
  if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
    args+=(--allow-partial)
  fi

  run_repromo "${args[@]}"
done
