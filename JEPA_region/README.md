# JEPA Region / E3

Code for the E3 setting in the thesis: JEPA-style sequence-level predictive representations for buggy/fixed code pairs.

## Structure

- `src/train.py`: train the E3 context encoder and sequence predictor
- `src/infer.py`: export pooled embeddings from a trained checkpoint
- `src/models.py`: Hugging Face encoder and latent transformer predictor
- `src/losses.py`: alignment, variance, and covariance losses
- `src/utils.py`: config loading, DDP helpers, logging, and region pooling
- `configs/`: experiment configs
- `scripts/`: runnable Slurm entry points

## Method

1. Load buggy/fixed code pairs from the configured dataset.
2. Tokenize both versions and locate the changed token region.
3. Encode buggy code as a full hidden-state sequence with the context encoder.
4. Encode fixed code with an EMA target encoder.
5. Predict a fixed-code hidden-state sequence with a latent transformer predictor.
6. Pool the selected supervision region from the predicted and target sequences.
7. Train with cosine alignment plus optional variance/covariance regularization.

The supported E3 supervision variants are:

- `change_region`: E3-S1, pool only the detected change span
- `change_plus_shared`: E3-S2, up-weight the change span and keep shared-token context
- `full_sequence`: E3-S3, pool uniformly over all non-padding tokens

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
