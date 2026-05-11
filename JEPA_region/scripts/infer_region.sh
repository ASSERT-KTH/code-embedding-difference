#!/bin/bash
#SBATCH -A NAISS2026-3-261
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 06:00:00
#SBATCH -J infer_last_change_region
#SBATCH -o logs/encoder/%x_%A.out
#SBATCH -e logs/encoder/%x_%A.err

set -euo pipefail

WORKDIR="/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/Decoder_RBR/JEPA_region"
SIF="/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/Decoder_RBR/decoder_rbr.sif"
PYDEPS="/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/Decoder_RBR/pydeps"
PY_FILE="${WORKDIR}/src/infer.py"
EXP_CONFIG="${WORKDIR}/configs/base.yaml"

SEQ_MODE="change_region"
CKPT_PATH="/mimer/NOBACKUP/groups/naiss2025-5-243/youya/CodeRepair_JEPA_region/checkpoints/YOUR_CHANGE_REGION_RUN/checkpoints/ckpt_last.pt"
RUN_NAME="change_region_pooled_last"
OUT_BASE="/mimer/NOBACKUP/groups/naiss2025-5-243/youya/CodeRepair_JEPA_region/pooled_embeddings"
SAVE_PATH="${OUT_BASE}/${RUN_NAME}"

GLOBAL_TARGET_DIR="/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/Decoder_RBR/saved_indices"
SPLIT_DIR="/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/Decoder_RBR/subset_indices"
TRAIN_INDICES="train_idx.npy"
VAL_INDICES="val_idx.npy"
TEST_INDICES="test_idx.npy"

HF_HOME="/mimer/NOBACKUP/groups/naiss2025-5-243/hf_cache"
TRANSFORMERS_CACHE="${HF_HOME}/transformers"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

INFER_BATCH_SIZE=32
SAVE_DTYPE="float16"
MAX_ITEMS=-1
NPROC=4

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=warn
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONNOUSERSITE=1

mkdir -p "${WORKDIR}/logs/encoder" "${OUT_BASE}"
cd "${WORKDIR}"

MASTER_ADDR=$(hostname)
MASTER_PORT=$((15000 + RANDOM % 20000))
export MASTER_ADDR MASTER_PORT

echo "[Info] SEQ_MODE=${SEQ_MODE}"
echo "[Info] CKPT_PATH=${CKPT_PATH}"
echo "[Info] SAVE_PATH=${SAVE_PATH}"
echo "[Info] GLOBAL_TARGET_DIR=${GLOBAL_TARGET_DIR}"
echo "[Info] SPLIT_DIR=${SPLIT_DIR}"
echo "[Info] TRAIN_INDICES=${TRAIN_INDICES} VAL_INDICES=${VAL_INDICES} TEST_INDICES=${TEST_INDICES}"
echo "[Info] INFER_BATCH_SIZE=${INFER_BATCH_SIZE} SAVE_DTYPE=${SAVE_DTYPE} MAX_ITEMS=${MAX_ITEMS}"
echo "[Info] NodeList=${SLURM_NODELIST:-NA} GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-NA}"

srun -N 1 -n 1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  apptainer exec --cleanenv --nv \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    --env MASTER_ADDR="${MASTER_ADDR}" \
    --env MASTER_PORT="${MASTER_PORT}" \
    --env TOKENIZERS_PARALLELISM=false \
    --env NCCL_ASYNC_ERROR_HANDLING=1 \
    --env NCCL_DEBUG=warn \
    --env PYTHONUNBUFFERED=1 \
    --env PYTHONPATH="${PYDEPS}:${PYTHONPATH:-}" \
    --env TORCHDYNAMO_DISABLE=1 \
    --env TORCHINDUCTOR_DISABLE=1 \
    --env TORCH_COMPILE_DISABLE=1 \
    --bind "${WORKDIR}:${WORKDIR}" \
    --bind "${HF_HOME}:${HF_HOME}" \
    --bind "${PYDEPS}:${PYDEPS}" \
    "${SIF}" bash -lc "
      set -euo pipefail
      cd '${WORKDIR}'
      echo \"[in-container] host=\$(hostname) CVD=\${CUDA_VISIBLE_DEVICES:-}\"

      python -m torch.distributed.run \
        --nnodes=1 \
        --nproc_per_node=${NPROC} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \
        '${PY_FILE}' \
          --config '${EXP_CONFIG}' \
          --set data.indices.global_target_dir='${GLOBAL_TARGET_DIR}' \
          --set data.indices.split_dir='${SPLIT_DIR}' \
          --set infer.ckpt_path='${CKPT_PATH}' \
          --set infer.save_path='${SAVE_PATH}' \
          --set infer.splits=[train,val,test] \
          --set infer.indices_file_by_split.train='${TRAIN_INDICES}' \
          --set infer.indices_file_by_split.val='${VAL_INDICES}' \
          --set infer.indices_file_by_split.test='${TEST_INDICES}' \
          --set infer.batch_size=${INFER_BATCH_SIZE} \
          --set infer.save_dtype='${SAVE_DTYPE}' \
          --set infer.max_items=${MAX_ITEMS} \
          --set ddp.enabled=true \
          --set data.num_workers=8
    "

echo "DONE"
