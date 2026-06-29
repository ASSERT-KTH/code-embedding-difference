# JEPA Downstream

Downstream probing tasks for evaluating JEPA-style code embeddings.

## Structure

- `src/bug_detection/train.py`: train a binary buggy-vs-fixed classifier
- `src/bug_detection/test.py`: evaluate a saved bug detection classifier
- `src/edit_type/train.py`: train a multi-label edit-type classifier
- `src/edit_type/test.py`: evaluate a saved edit-type classifier
- `src/common/io.py`: load offline embeddings or extract embeddings online from checkpoints
- `src/common/features.py`: build task-specific embedding features
- `src/common/models.py`: MLP classifier modules

## Tasks

1. Bug detection
   - Input: buggy and fixed embeddings
   - Target: classify whether an embedding comes from buggy or fixed code
   - Metrics: accuracy, precision, recall, F1

2. Edit-type classification
   - Input: embedding features such as `tgt_minus_buggy`, `tgt`, or `buggy`
   - Target: predict one or more edit labels for each repair
   - Metrics: micro-F1 and macro-F1

## Embedding Sources

The scripts support two embedding backends:

- `offline_pt`: load precomputed `.pt` embedding files
- `online_checkpoint`: extract embeddings directly from a JEPA checkpoint

Supported method names:

- `e1`: frozen pooled embeddings
- `e2`: pooled JEPA-style embeddings
- `seq_emb`: sequence-level / E3 pooled embeddings

## Main Entry Points

- `src/bug_detection/train.py`: train bug detection probe
- `src/bug_detection/test.py`: test bug detection probe
- `src/edit_type/train.py`: train edit-type probe
- `src/edit_type/test.py`: test edit-type probe

## Important Options

- `--backend`: `offline_pt` or `online_checkpoint`
- `--method`: `e1`, `e2`, or `seq_emb`
- `--embedding-view`: `ctx`, `tgt`, or `ctx_tgt`
- `--feature-mode`: edit-type feature mode, usually `tgt_minus_buggy`
- `--train-dir`, `--val-dir`, `--test-dir`: embedding split directories
- `--output-dir`: directory for checkpoints, metrics, and predictions
