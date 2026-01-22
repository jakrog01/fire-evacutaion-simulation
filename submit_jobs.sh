#!/usr/bin/env bash
set -euo pipefail

N_RUNS="${N_RUNS:-10000}"
SEED="${SEED:-12345}"
T_END="${T_END:-30.0}"
DT="${DT:-0.001}"
SAMPLE_DT="${SAMPLE_DT:-0.05}"

MATRICES="${MATRICES:-sim_output/matrices}"
ROI_JSON="${ROI_JSON:-scripts/output/roi_groups_important.json}"
OBJECTIVES_JSON="${OBJECTIVES_JSON:-tuning/objectives.json}"

MAX_PARALLEL="${MAX_PARALLEL:-400}"
PARTITION="${PARTITION:-topola}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-tuning/grid_runs/gs_${STAMP}}"
MANIFEST="${OUT_ROOT}/manifest.jsonl"
OFFSETS="${OUT_ROOT}/manifest_offsets.npy"

mkdir -p "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}/_slurm"

python grid_search.py generate --out_root "${OUT_ROOT}" --matrices "${MATRICES}" --roi_json "${ROI_JSON}" --objectives_json "${OBJECTIVES_JSON}" --n_runs "${N_RUNS}" --seed "${SEED}" --t_end "${T_END}" --dt "${DT}" --sample_dt "${SAMPLE_DT}"

ARRAY_MAX=$((N_RUNS-1))

sbatch \
  --partition="${PARTITION}" \
  --array="0-${ARRAY_MAX}%${MAX_PARALLEL}" \
  --output="${OUT_ROOT}/_slurm/%A_%a.out" \
  --error="${OUT_ROOT}/_slurm/%A_%a.err" \
  --export=ALL,OUT_ROOT="${OUT_ROOT}",MANIFEST="${MANIFEST}",OFFSETS="${OFFSETS}" \
  slurm_array.sbatch

echo "${OUT_ROOT}"
