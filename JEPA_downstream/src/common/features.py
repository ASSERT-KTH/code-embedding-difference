from __future__ import annotations

import numpy as np


EDIT_TYPE_FEATURE_MODES = ("tgt_minus_buggy", "tgt", "buggy")
BINARY_FEATURE_MODES = ("buggy",)


def build_edit_type_feature(buggy_emb: np.ndarray, tgt_emb: np.ndarray, feature_mode: str) -> np.ndarray:
    if feature_mode == "tgt_minus_buggy":
        x = tgt_emb - buggy_emb
    elif feature_mode == "tgt":
        x = tgt_emb
    elif feature_mode == "buggy":
        x = buggy_emb
    else:
        raise ValueError(f"Unsupported edit_type feature_mode: {feature_mode}")
    return np.asarray(x, dtype=np.float32)


def build_binary_feature(buggy_emb: np.ndarray, feature_mode: str) -> np.ndarray:
    if feature_mode != "buggy":
        raise ValueError(f"Unsupported bug_detection feature_mode: {feature_mode}")
    return np.asarray(buggy_emb, dtype=np.float32)


def infer_edit_type_input_dim(feature_mode: str, emb_dim: int) -> int:
    if feature_mode in EDIT_TYPE_FEATURE_MODES:
        return int(emb_dim)
    raise ValueError(f"Unsupported edit_type feature_mode: {feature_mode}")


def infer_binary_input_dim(feature_mode: str, emb_dim: int) -> int:
    if feature_mode in BINARY_FEATURE_MODES:
        return int(emb_dim)
    raise ValueError(f"Unsupported bug_detection feature_mode: {feature_mode}")
