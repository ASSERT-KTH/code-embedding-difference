#!/bin/bash
#SBATCH -A naiss2025-5-243
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 06:00:00
#SBATCH -J region_lora_sweep
#SBATCH --array=0-11
#SBATCH -o logs/encoder/%x_%A_%a.out
#SBATCH -e logs/encoder/%x_%A_%a.err

set -euo pipefail

# -------------------------
# Paths
# -------------------------
WORKDIR="/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/Decoder_RBR/JEPA_region"
SIF="/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/Decoder_RBR/decoder_rbr.sif"

PY_FILE="${WORKDIR}/src/train.py"
EXP_CONFIG="${WORKDIR}/configs/base.yaml"

OUT_BASE="/mimer/NOBACKUP/groups/naiss2025-5-243/youya/CodeRepair_JEPA_region/checkpoints"
PYDEPS="/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/Decoder_RBR/pydeps"

# -------------------------
# Indices
# -------------------------
GLOBAL_TARGET_DIR="/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/Decoder_RBR/saved_indices"
SPLIT_DIR="/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/Decoder_RBR/subset_indices"

# -------------------------
# W&B
# -------------------------
export WANDB_MODE="online"
export WANDB_ENTITY="assert-kth"
export WANDB_PROJECT="CodeRepair_JEPA_region"
export WANDB_GROUP="encoder_region_lora"
export WANDB_RESUME="never"

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
export PYTHONNOUSERSITE=1

mkdir -p "${WORKDIR}/logs/encoder" "${OUT_BASE}"
cd "${WORKDIR}"

# -------------------------
# Sweep: 2 LR sets x 2 predictor LRs x 3 w_cov = 12 jobs
# -------------------------
LR_E_SET=(1e-5 2e-5)
LR_P_SET=(5e-5 1e-4)
W_COV_SET=(0.0 0.01 0.1)

TASK_ID=${SLURM_ARRAY_TASK_ID}

N_P=${#LR_P_SET[@]}
N_C=${#W_COV_SET[@]}
BLOCK=$((N_P * N_C))

LR_E_IDX=$(( TASK_ID / BLOCK ))
REM=$(( TASK_ID % BLOCK ))
LR_P_IDX=$(( REM / N_C ))
W_COV_IDX=$(( REM % N_C ))

LR_E=${LR_E_SET[$LR_E_IDX]}
LR_P=${LR_P_SET[$LR_P_IDX]}
W_COV=${W_COV_SET[$W_COV_IDX]}

# LoRA defaults for the first sweep
LORA_R=32
LORA_ALPHA=$((2 * LORA_R))
LORA_DROPOUT=0.05
TAU=0.996

# Predictor / supervision setup
SUPERVISION_TARGET="change_region"   # change_region / change_plus_shared / full_sequence
SHARED_WEIGHT=0.1
USE_RESIDUAL=true                    # true / false
RESIDUAL_SCALE=1.0

# Training defaults for the first sweep
BATCH_SIZE=4
GRAD_ACCUM=2
EPOCHS=1
RESUME_CKPT=""

RES_TAG="residual"
if [ "${USE_RESIDUAL}" = "false" ]; then
  RES_TAG="no_residual"
fi
RUN_NAME="region_${SUPERVISION_TARGET}_lrE${LR_E}_lrP${LR_P}_wcov${W_COV}_r${LORA_R}_${RES_TAG}"
if [ "${SUPERVISION_TARGET}" = "change_plus_shared" ]; then
  RUN_NAME="${RUN_NAME}_wshared${SHARED_WEIGHT}"
fi

echo "[Info] TASK_ID=${TASK_ID}"
echo "[Info] LR_E=${LR_E} LR_P=${LR_P} W_COV=${W_COV}"
echo "[Info] LORA_R=${LORA_R} LORA_ALPHA=${LORA_ALPHA} TAU=${TAU}"
echo "[Info] SUPERVISION_TARGET=${SUPERVISION_TARGET} SHARED_WEIGHT=${SHARED_WEIGHT} USE_RESIDUAL=${USE_RESIDUAL} RESIDUAL_SCALE=${RESIDUAL_SCALE}"
echo "[Info] BATCH_SIZE=${BATCH_SIZE} GRAD_ACCUM=${GRAD_ACCUM} EPOCHS=${EPOCHS}"
echo "[Info] RESUME_CKPT=${RESUME_CKPT:-NONE}"
echo "[Info] GLOBAL_TARGET_DIR=${GLOBAL_TARGET_DIR}"
echo "[Info] SPLIT_DIR=${SPLIT_DIR}"
echo "[Info] RUN_NAME=${RUN_NAME}"
echo "[Info] SAVE_ROOT=${OUT_BASE}"
echo "[Info] NodeList=${SLURM_NODELIST:-NA} GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-NA}"
echo "[Info] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"

# -------------------------
# torchrun rendezvous
# -------------------------
MASTER_ADDR=$(hostname)
MASTER_PORT=$((15000 + RANDOM % 20000))
export MASTER_ADDR MASTER_PORT
NPROC=${SLURM_GPUS_ON_NODE:-1}

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
    --env WANDB_MODE="${WANDB_MODE}" \
    --env WANDB_ENTITY="${WANDB_ENTITY}" \
    --env WANDB_PROJECT="${WANDB_PROJECT}" \
    --env WANDB_GROUP="${WANDB_GROUP}" \
    --env WANDB_RESUME="${WANDB_RESUME}" \
    --env PYTHONPATH="${PYDEPS}:${PYTHONPATH:-}" \
    --env TORCHDYNAMO_DISABLE=1 \
    --env TORCHINDUCTOR_DISABLE=1 \
    --env TORCH_COMPILE_DISABLE=1 \
    --bind "${WORKDIR}:${WORKDIR}" \
    --bind "${HF_HOME}:${HF_HOME}" \
    --bind "${PYDEPS}:${PYDEPS}" \
    "$SIF" bash -lc "
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
          --set run.save_dir='${OUT_BASE}' \
          --set run.run_name='${RUN_NAME}' \
          --set data.indices.global_target_dir='${GLOBAL_TARGET_DIR}' \
          --set data.indices.split_dir='${SPLIT_DIR}' \
          --set wandb.enabled=true \
          --set wandb.entity='${WANDB_ENTITY}' \
          --set wandb.project='${WANDB_PROJECT}' \
          --set wandb.group='${WANDB_GROUP}' \
          --set wandb.run_name='${RUN_NAME}' \
          --set wandb.id='' \
          --set wandb.resume='${WANDB_RESUME}' \
          --set encoder.train_mode=lora \
          --set encoder.lora.enabled=true \
          --set encoder.lora.r=${LORA_R} \
          --set encoder.lora.alpha=${LORA_ALPHA} \
          --set encoder.lora.dropout=${LORA_DROPOUT} \
          --set encoder.lora.target_modules=[Wqkv,Wo] \
          --set predictor.use_residual=${USE_RESIDUAL} \
          --set predictor.residual_scale=${RESIDUAL_SCALE} \
          --set train.batch_size=${BATCH_SIZE} \
          --set train.grad_accum=${GRAD_ACCUM} \
          --set train.epochs=${EPOCHS} \
          --set train.lr_encoder=${LR_E} \
          --set train.lr_predictor=${LR_P} \
          --set train.resume_from='${RESUME_CKPT}' \
          --set loss.supervision_target='${SUPERVISION_TARGET}' \
          --set loss.shared_weight=${SHARED_WEIGHT} \
          --set loss.w_align=1.0 \
          --set loss.w_var=1.0 \
          --set loss.w_cov=${W_COV} \
          --set ema.tau=${TAU} \
          --set train.eval_every_steps=800 \
          --set train.save_every_steps=2000 \
          --set data.num_workers=8
    "

echo "DONE"
