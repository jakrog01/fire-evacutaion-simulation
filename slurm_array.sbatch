#!/bin/bash
#SBATCH --job-name=ons_fire_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:25:00

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTHONHASHSEED=0

if [ -z "${NUMBA_CACHE_DIR:-}" ]; then
  export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID}"
fi

python grid_search.py run --out_root "${OUT_ROOT}" --manifest "${MANIFEST}" --offsets "${OFFSETS}" --index "${SLURM_ARRAY_TASK_ID}" --retries "${RETRIES:-1}"
