# JEPA Region

Region-based JEPA encoder training and embedding inference for buggy/fixed code pairs.

## Structure

- `src/train.py`: train the region JEPA encoder and predictor
- `src/infer.py`: export pooled embeddings from a trained checkpoint
- `src/models.py`: Hugging Face encoder and lightweight predictor modules
- `src/losses.py`: alignment, variance, and covariance losses
- `src/utils.py`: config loading, DDP helpers, logging, and region pooling
- `configs/`: experiment configs
- `scripts/`: runnable Slurm entry points

## Method

1. Load buggy/fixed code pairs from the configured dataset.
2. Tokenize both versions and locate the changed token region.
3. Encode buggy code with the context encoder.
4. Encode fixed code with the EMA target encoder.
5. Predict fixed-code representations from buggy-code representations.
6. Pool the selected supervision region.
7. Train with cosine alignment plus optional variance/covariance regularization.

## Main Entry Points

- `src/train.py`: main training script
- `src/infer.py`: embedding inference and export script
- `scripts/train_region_lora_sweep.sh`: Slurm LoRA sweep
- `scripts/infer_region.sh`: Slurm embedding export

## Config

Main config file:

- `configs/base.yaml`

Important options:

- `encoder.train_mode`: `lora` or `full`
- `loss.supervision_target`: `change_region`, `change_plus_shared`, or `full_sequence`
- `train.resume_from`: checkpoint path for resuming training
- `infer.ckpt_path`: checkpoint path for inference
- `infer.save_path`: output directory for exported embeddings
