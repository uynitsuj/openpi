#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}"
DATASET_ROOT="${DATASET_ROOT:-${HF_LEROBOT_HOME}}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train.py}"
EXP_PREFIX="${EXP_PREFIX:-repromo_sim_rabc}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${HOME}/checkpoints/openpi}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${REPO_ROOT}/assets}"
FORCE_LINKS="${FORCE_LINKS:-0}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/repromo_rabc_${TIMESTAMP}}"
DRY_RUN=0

# Match the SkyPilot launch path (sky_utils.generate_sky_config defaults to 0.95).
# JAX's default is 0.75, which caps the BFC pool at ~61 GiB on an 80 GB A100 and
# OOMs batch_size=32 runs even though the card has spare memory. Each run gets a
# dedicated GPU here, so a high fraction is safe.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

RUNS=(
  "hang_mug|pi0_sim_hang_mug_rabc_finalaction_thr100_nomax|sim_hang_the_mug_on_the_mug_rack_30hz_gop10"
  "load_plates|pi0_sim_load_plates_rabc_finalaction_thr100_nomax|sim_load_the_plates_into_the_dish_rack_30hz_gop10"
  "put_bottles|pi0_sim_put_bottles_rabc_finalaction_thr100_nomax|sim_put_the_plastic_bottles_in_the_bin_30hz_gop10"
  "sweep_paper|pi0_sim_sweep_paper_rabc_finalaction_thr100_nomax|sim_sweep_away_paper_scraps_from_the_table_30hz_gop10"
  "throw_bottles|pi0_sim_throw_bottles_rabc_finalaction_thr100_nomax|sim_throw_plastic_bottles_in_bin_30hz_gop10"
  "turn_mug|pi0_sim_turn_mug_rabc_finalaction_thr100_nomax|sim_turn_the_mug_right_side_up_30hz_gop10"
)

TRAIN_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --dry-run)
      DRY_RUN=1
      ;;
    --disable-wandb)
      TRAIN_ARGS+=("--no-wandb-enabled")
      ;;
    *)
      TRAIN_ARGS+=("${arg}")
      ;;
  esac
done

cd "${REPO_ROOT}"

mkdir -p "${HF_LEROBOT_HOME}"

detect_gpus() {
  if [[ -n "${GPU_IDS:-}" ]]; then
    IFS="," read -r -a GPU_LIST <<< "${GPU_IDS}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS="," read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader)
  else
    GPU_LIST=()
  fi

  if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      GPU_LIST=("0")
    else
      echo "[error] No GPUs detected. Set GPU_IDS=0,1,... to override." >&2
      exit 1
    fi
  fi
}

prepare_dataset() {
  local config_name="$1"
  local repo_id="$2"
  local dataset_dir="${DATASET_ROOT}/${repo_id}"
  local cache_path="${HF_LEROBOT_HOME}/${repo_id}"
  local asset_dir="${ASSETS_BASE_DIR}/${config_name}/${repo_id}"

  if [[ ! -d "${dataset_dir}" ]]; then
    echo "[error] Missing dataset directory: ${dataset_dir}" >&2
    exit 1
  fi

  if [[ -e "${cache_path}" || -L "${cache_path}" ]]; then
    if [[ "$(realpath "${cache_path}")" != "$(realpath "${dataset_dir}")" ]]; then
      if [[ "${FORCE_LINKS}" == "1" && -L "${cache_path}" ]]; then
        rm "${cache_path}"
        ln -s "${dataset_dir}" "${cache_path}"
      else
        echo "[error] ${cache_path} exists and does not point at ${dataset_dir}" >&2
        echo "        Set FORCE_LINKS=1 to replace an existing symlink." >&2
        exit 1
      fi
    fi
  else
    ln -s "${dataset_dir}" "${cache_path}"
  fi

  if [[ ! -f "${dataset_dir}/norm_stats.json" ]]; then
    echo "[error] Missing norm stats: ${dataset_dir}/norm_stats.json" >&2
    exit 1
  fi
  mkdir -p "${asset_dir}"
  cp "${dataset_dir}/norm_stats.json" "${asset_dir}/norm_stats.json"
}

wait_for_batch() {
  local failed=0
  local i
  for i in "${!BATCH_PIDS[@]}"; do
    if wait "${BATCH_PIDS[$i]}"; then
      echo "[ok] ${BATCH_NAMES[$i]}"
    else
      echo "[error] ${BATCH_NAMES[$i]} failed; see ${BATCH_LOGS[$i]}" >&2
      failed=1
    fi
  done
  BATCH_PIDS=()
  BATCH_NAMES=()
  BATCH_LOGS=()
  return "${failed}"
}

detect_gpus
echo "[info] Using GPUs: ${GPU_LIST[*]}"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
  echo "[info] Logs: ${LOG_DIR}"
fi

BATCH_PIDS=()
BATCH_NAMES=()
BATCH_LOGS=()
FAILED=0
RUN_INDEX=0

for run in "${RUNS[@]}"; do
  IFS="|" read -r short_name config_name repo_id <<< "${run}"
  exp_name="${EXP_PREFIX}_${short_name}_${TIMESTAMP}"
  gpu_id="${GPU_LIST[$((RUN_INDEX % ${#GPU_LIST[@]}))]}"
  log_file="${LOG_DIR}/${short_name}.log"
  cmd=(
    uv run "${TRAIN_SCRIPT}" "${config_name}"
    --exp-name "${exp_name}"
    --assets-base-dir "${ASSETS_BASE_DIR}"
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
    "${TRAIN_ARGS[@]}"
  )

  echo
  echo "[run] ${config_name}"
  echo "      dataset: ${DATASET_ROOT}/${repo_id}"
  echo "      exp:     ${exp_name}"
  echo "      gpu:     ${gpu_id}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    echo "      log:     ${log_file}"
  fi
  printf '      command:'
  printf ' CUDA_VISIBLE_DEVICES=%q' "${gpu_id}"
  printf ' %q' "${cmd[@]}"
  echo

  if [[ "${DRY_RUN}" == "1" ]]; then
    RUN_INDEX=$((RUN_INDEX + 1))
    continue
  fi

  prepare_dataset "${config_name}" "${repo_id}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    "${cmd[@]}"
  ) > "${log_file}" 2>&1 &

  BATCH_PIDS+=("$!")
  BATCH_NAMES+=("${config_name}")
  BATCH_LOGS+=("${log_file}")

  if [[ "${#BATCH_PIDS[@]}" -eq "${#GPU_LIST[@]}" ]]; then
    if ! wait_for_batch; then
      FAILED=1
    fi
  fi

  RUN_INDEX=$((RUN_INDEX + 1))
done

if [[ "${DRY_RUN}" != "1" && "${#BATCH_PIDS[@]}" -gt 0 ]]; then
  if ! wait_for_batch; then
    FAILED=1
  fi
fi

exit "${FAILED}"
