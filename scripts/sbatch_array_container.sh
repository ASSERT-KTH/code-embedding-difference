#!/bin/bash
#SBATCH -A naiss2025-5-243
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 10:10:00
#SBATCH -J lora_grid
#SBATCH --array=0-2,6,10
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

# -------------------------
# Paths
# -------------------------
WORKDIR="/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/Decoder_RBR"
PY_FILE="baseline_sft_lora.py"
SIF="${WORKDIR}/decoder_rbr.sif"

# trial configs
CFG_DIR="${WORKDIR}/configs/baseline_sft/grid"

# indices & outputs
INDICES_DIR="/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/Decoder_RBR/saved_indices"
OUT_BASE="/mimer/NOBACKUP/groups/naiss2025-5-243/youya/CodeRepair_JEPA/baseline_sft"

# external deps (pip --target installs here)
PYDEPS="${WORKDIR}/pydeps"

# -------------------------
# W&B (optional)
# -------------------------
export WANDB_ENTITY="assert-kth"
export WANDB_PROJECT="CodeRepair_JEPA"
export WANDB_MODE="online"

# -------------------------
# HF cache
# -------------------------
export HF_HOME="/mimer/NOBACKUP/groups/naiss2025-5-243/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# -------------------------
# Runtime env
# -------------------------
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=warn
export NCCL_ASYNC_ERROR_HANDLING=1
unset PYTHONPATH || true
export PYTHONNOUSERSITE=1

# add external deps to python path
mkdir -p "${PYDEPS}"
export PYTHONPATH="${PYDEPS}:${PYTHONPATH:-}"

mkdir -p "${WORKDIR}/logs" "${OUT_BASE}"
cd "${WORKDIR}"

# -------------------------
# Pick config by array task id
# -------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}
CONFIG=$(printf "%s/trial_%03d.yaml" "${CFG_DIR}" "${TASK_ID}")

if [ ! -f "${CONFIG}" ]; then
  echo "[Error] Config not found: ${CONFIG}"
  exit 1
fi

CONFIG_BASENAME="$(basename "${CONFIG}" .yaml)"

# output dir for this trial (stable across retries/resubmits)
OUTDIR="${OUT_BASE}/${CONFIG_BASENAME}"
mkdir -p "${OUTDIR}"

# Optional: skip completed trials (requires you to create done.txt at end of training)
# if [ -f "${OUTDIR}/done.txt" ]; then
#   echo "[Info] Already done, skipping: ${OUTDIR}"
#   exit 0
# fi

echo "[Info] TASK_ID=${TASK_ID}"
echo "[Info] CONFIG=${CONFIG}"
echo "[Info] OUTDIR=${OUTDIR}"
echo "[Info] PYDEPS=${PYDEPS}"
echo "[Info] CWD=$(pwd)"
echo "[Info] NodeList=${SLURM_NODELIST:-NA}  GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-NA}"
echo "[Info] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"

# -------------------------
# torchrun rendezvous (single node)
# -------------------------
MASTER_ADDR=$(hostname)
MASTER_PORT=$((15000 + RANDOM % 20000))
export MASTER_ADDR MASTER_PORT
echo "[Info] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

NPROC=${SLURM_GPUS_ON_NODE:-1}
echo "[Info] NPROC_PER_NODE=$NPROC"

# -------------------------
# Sanity check (optional but helpful)
# -------------------------
srun -N 1 -n 1 --ntasks-per-node=1 \
  apptainer exec --cleanenv --nv \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    --bind "${WORKDIR}:${WORKDIR}" \
    --bind "${HF_HOME}:${HF_HOME}" \
    --bind "${PYDEPS}:${PYDEPS}" \
    --env PYTHONPATH="${PYDEPS}:${PYTHONPATH:-}" \
    "$SIF" bash -lc \
    'echo "[sanity] host=$(hostname) CVD=${CUDA_VISIBLE_DEVICES:-}";
     python -c "import os, torch; print(\"[sanity] CVD_env=\", os.environ.get(\"CUDA_VISIBLE_DEVICES\")); print(\"cuda\", torch.cuda.is_available(), \"ngpu\", torch.cuda.device_count())"'

# -------------------------
# Main run
# -------------------------
srun -N 1 -n 1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  apptainer exec --cleanenv --nv \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    --env MASTER_ADDR="$MASTER_ADDR" \
    --env MASTER_PORT="$MASTER_PORT" \
    --env TOKENIZERS_PARALLELISM=false \
    --env NCCL_ASYNC_ERROR_HANDLING=1 \
    --env NCCL_DEBUG=warn \
    --env PYTHONUNBUFFERED=1 \
    --env WANDB_ENTITY="${WANDB_ENTITY}" \
    --env WANDB_PROJECT="${WANDB_PROJECT}" \
    --env WANDB_MODE="${WANDB_MODE}" \
    --env PYTHONPATH="${PYDEPS}:${PYTHONPATH:-}" \
    --bind "${WORKDIR}:${WORKDIR}" \
    --bind "${HF_HOME}:${HF_HOME}" \
    --bind "${PYDEPS}:${PYDEPS}" \
    "$SIF" bash -lc "
      set -euo pipefail
      cd '${WORKDIR}'
      echo \"[in-container] host=\$(hostname) CVD=\${CUDA_VISIBLE_DEVICES:-}\"
      python -c \"import os; print('[in-container] CVD_env=', os.environ.get('CUDA_VISIBLE_DEVICES'))\"
      echo '[in-container] PYTHONPATH=' \$PYTHONPATH

      python -m torch.distributed.run \
        --nnodes=1 \
        --nproc_per_node=${NPROC} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \
        '${PY_FILE}' \
          --config '${CONFIG}' \
          --indices_dir '${INDICES_DIR}' \
          --output_dir '${OUTDIR}' 
    "

echo "DONE"

