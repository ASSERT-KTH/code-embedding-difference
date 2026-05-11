from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset


DEFAULT_E1_FILES = {
    "buggy_file_name": "z_buggy.pt",
    "buggy_key": "z_buggy",
    "pred_file_name": "z_pred.pt",
    "pred_key": "z_pred",
    "tgt_file_name": "z_gt.pt",
    "tgt_key": "z_gt",
}

DEFAULT_E2_FILES = {
    "buggy_file_name": "z_ctx.pt",
    "buggy_key": "z_ctx",
    "pred_file_name": "z_pred.pt",
    "pred_key": "z_pred",
    "tgt_file_name": "z_tgt.pt",
    "tgt_key": "z_tgt",
}

DEFAULT_POOLED_FILES = {
    "buggy_ctx_file_name": "buggy_ctx.pt",
    "buggy_ctx_key": "emb",
    "fixed_ctx_file_name": "fixed_ctx.pt",
    "fixed_ctx_key": "emb",
    "buggy_tgt_file_name": "buggy_tgt.pt",
    "buggy_tgt_key": "emb",
    "fixed_tgt_file_name": "fixed_tgt.pt",
    "fixed_tgt_key": "emb",
    "pred_file_name": "pred.pt",
    "pred_key": "emb",
}


@dataclass
class LoadedSplitEmbeddings:
    buggy_emb: np.ndarray
    pred_emb: np.ndarray
    tgt_emb: np.ndarray
    global_indices: np.ndarray


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_tensor_file(path: str) -> Any:
    suffix = Path(path).suffix.lower()
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu", weights_only=False)
    raise ValueError(f"Unsupported file suffix for tensor file: {path}")


def extract_dict_array(data: Any, path: str, key: str, fallback_keys: Optional[Sequence[str]] = None) -> np.ndarray:
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a dict, got: {type(data)}")
    candidate_keys = [key]
    if fallback_keys is not None:
        candidate_keys.extend(str(k) for k in fallback_keys)
    for candidate in candidate_keys:
        if candidate in data:
            return to_numpy(data[candidate])
    raise KeyError(
        f"Missing keys={candidate_keys} in {path}. Available keys: {list(data.keys())}"
    )


def squeeze_embedding_array(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D] or [N, 1, D], got: {arr.shape}")
    return arr.astype(np.float32)


def resolve_method_files(method: str) -> Dict[str, str]:
    method_name = str(method).lower()
    if method_name == "e1":
        return dict(DEFAULT_E1_FILES)
    if method_name == "e2":
        return dict(DEFAULT_E2_FILES)
    if method_name in {"seq", "seq_emb"}:
        return dict(DEFAULT_POOLED_FILES)
    raise ValueError(f"Unsupported offline method: {method}")


def build_file_spec(
    method: str,
    buggy_file_name: str = "",
    buggy_key: str = "",
    pred_file_name: str = "",
    pred_key: str = "",
    tgt_file_name: str = "",
    tgt_key: str = "",
    buggy_ctx_file_name: str = "",
    buggy_ctx_key: str = "",
    fixed_ctx_file_name: str = "",
    fixed_ctx_key: str = "",
    buggy_tgt_file_name: str = "",
    buggy_tgt_key: str = "",
    fixed_tgt_file_name: str = "",
    fixed_tgt_key: str = "",
) -> Dict[str, str]:
    spec = resolve_method_files(method)
    method_name = str(method).lower()
    if method_name in {"seq", "seq_emb"}:
        if buggy_file_name:
            spec["buggy_ctx_file_name"] = buggy_file_name
        if buggy_key:
            spec["buggy_ctx_key"] = buggy_key
        if tgt_file_name:
            spec["fixed_ctx_file_name"] = tgt_file_name
        if tgt_key:
            spec["fixed_ctx_key"] = tgt_key
        if buggy_ctx_file_name:
            spec["buggy_ctx_file_name"] = buggy_ctx_file_name
        if buggy_ctx_key:
            spec["buggy_ctx_key"] = buggy_ctx_key
        if fixed_ctx_file_name:
            spec["fixed_ctx_file_name"] = fixed_ctx_file_name
        if fixed_ctx_key:
            spec["fixed_ctx_key"] = fixed_ctx_key
        if buggy_tgt_file_name:
            spec["buggy_tgt_file_name"] = buggy_tgt_file_name
        if buggy_tgt_key:
            spec["buggy_tgt_key"] = buggy_tgt_key
        if fixed_tgt_file_name:
            spec["fixed_tgt_file_name"] = fixed_tgt_file_name
        if fixed_tgt_key:
            spec["fixed_tgt_key"] = fixed_tgt_key
        if pred_file_name:
            spec["pred_file_name"] = pred_file_name
        if pred_key:
            spec["pred_key"] = pred_key
        return spec
    if buggy_file_name:
        spec["buggy_file_name"] = buggy_file_name
    if buggy_key:
        spec["buggy_key"] = buggy_key
    if pred_file_name:
        spec["pred_file_name"] = pred_file_name
    if pred_key:
        spec["pred_key"] = pred_key
    if tgt_file_name:
        spec["tgt_file_name"] = tgt_file_name
    if tgt_key:
        spec["tgt_key"] = tgt_key
    return spec


def _extract_embedding_with_global_indices(
    path: Path,
    emb_key: str,
    global_indices_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    data = load_tensor_file(str(path))
    emb = squeeze_embedding_array(
        extract_dict_array(data, str(path), emb_key, fallback_keys=["emb"]),
        f"{path.parent.name}/{path.stem}",
    )
    gidx = extract_dict_array(data, str(path), global_indices_key).astype(np.int64)
    if len(emb) != len(gidx):
        raise ValueError(f"{path}: global_indices length does not match embeddings")
    return emb, gidx


def _resolve_view_file_pair(
    split_path: Path,
    spec: Dict[str, str],
    embedding_view: str,
) -> tuple[Path, str, Path, str]:
    view = str(embedding_view).lower()
    if view == "ctx":
        return (
            split_path / spec["buggy_ctx_file_name"],
            spec["buggy_ctx_key"],
            split_path / spec["fixed_ctx_file_name"],
            spec["fixed_ctx_key"],
        )
    if view == "tgt":
        return (
            split_path / spec["buggy_tgt_file_name"],
            spec["buggy_tgt_key"],
            split_path / spec["fixed_tgt_file_name"],
            spec["fixed_tgt_key"],
        )
    if view == "ctx_tgt":
        return (
            split_path / spec["buggy_ctx_file_name"],
            spec["buggy_ctx_key"],
            split_path / spec["fixed_tgt_file_name"],
            spec["fixed_tgt_key"],
        )
    raise ValueError(f"Unsupported embedding_view: {embedding_view}")


def _supports_pooled_view_files(split_path: Path) -> bool:
    return all(
        (split_path / name).exists()
        for name in ["buggy_ctx.pt", "fixed_ctx.pt", "buggy_tgt.pt", "fixed_tgt.pt", "pred.pt"]
    )


def load_method_split_embeddings(
    split_dir: str,
    method: str,
    embedding_view: str = "ctx",
    global_indices_key: str = "global_indices",
    buggy_file_name: str = "",
    buggy_key: str = "",
    pred_file_name: str = "",
    pred_key: str = "",
    tgt_file_name: str = "",
    tgt_key: str = "",
    buggy_ctx_file_name: str = "",
    buggy_ctx_key: str = "",
    fixed_ctx_file_name: str = "",
    fixed_ctx_key: str = "",
    buggy_tgt_file_name: str = "",
    buggy_tgt_key: str = "",
    fixed_tgt_file_name: str = "",
    fixed_tgt_key: str = "",
) -> LoadedSplitEmbeddings:
    split_path = Path(split_dir)
    spec = build_file_spec(
        method,
        buggy_file_name=buggy_file_name,
        buggy_key=buggy_key,
        pred_file_name=pred_file_name,
        pred_key=pred_key,
        tgt_file_name=tgt_file_name,
        tgt_key=tgt_key,
        buggy_ctx_file_name=buggy_ctx_file_name,
        buggy_ctx_key=buggy_ctx_key,
        fixed_ctx_file_name=fixed_ctx_file_name,
        fixed_ctx_key=fixed_ctx_key,
        buggy_tgt_file_name=buggy_tgt_file_name,
        buggy_tgt_key=buggy_tgt_key,
        fixed_tgt_file_name=fixed_tgt_file_name,
        fixed_tgt_key=fixed_tgt_key,
    )
    method_name = str(method).lower()

    if method_name in {"seq", "seq_emb"} or (method_name == "e2" and _supports_pooled_view_files(split_path)):
        pooled_spec = dict(DEFAULT_POOLED_FILES)
        if method_name in {"seq", "seq_emb"}:
            pooled_spec.update(spec)
        buggy_path, buggy_key_name, tgt_path, tgt_key_name = _resolve_view_file_pair(
            split_path, pooled_spec, embedding_view
        )
        pred_path = split_path / pooled_spec["pred_file_name"]
        if not buggy_path.exists():
            raise FileNotFoundError(f"Missing buggy embedding file: {buggy_path}")
        if not tgt_path.exists():
            raise FileNotFoundError(f"Missing target embedding file: {tgt_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing pred embedding file: {pred_path}")

        buggy_emb, buggy_global = _extract_embedding_with_global_indices(
            buggy_path, buggy_key_name, global_indices_key
        )
        tgt_emb, tgt_global = _extract_embedding_with_global_indices(
            tgt_path, tgt_key_name, global_indices_key
        )
        pred_emb, pred_global = _extract_embedding_with_global_indices(
            pred_path, pooled_spec["pred_key"], global_indices_key
        )

        num_samples = len(buggy_emb)
        if len(tgt_emb) != num_samples or len(pred_emb) != num_samples:
            raise ValueError(
                f"{split_dir}: embedding size mismatch: buggy={len(buggy_emb)} pred={len(pred_emb)} tgt={len(tgt_emb)}"
            )
        if not np.array_equal(buggy_global, pred_global) or not np.array_equal(buggy_global, tgt_global):
            raise ValueError(f"{split_dir}: global_indices differ across embedding files")

        return LoadedSplitEmbeddings(
            buggy_emb=buggy_emb,
            pred_emb=pred_emb,
            tgt_emb=tgt_emb,
            global_indices=buggy_global,
        )

    buggy_path = split_path / spec["buggy_file_name"]
    pred_path = split_path / spec["pred_file_name"]
    tgt_path = split_path / spec["tgt_file_name"]

    if not buggy_path.exists():
        raise FileNotFoundError(f"Missing buggy embedding file: {buggy_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing pred embedding file: {pred_path}")
    if not tgt_path.exists():
        raise FileNotFoundError(f"Missing target embedding file: {tgt_path}")

    buggy_data = load_tensor_file(str(buggy_path))
    pred_data = load_tensor_file(str(pred_path))
    tgt_data = load_tensor_file(str(tgt_path))

    buggy_emb = squeeze_embedding_array(
        extract_dict_array(buggy_data, str(buggy_path), spec["buggy_key"], fallback_keys=["emb"]),
        f"{split_path.name}/buggy_emb",
    )
    pred_emb = squeeze_embedding_array(
        extract_dict_array(pred_data, str(pred_path), spec["pred_key"], fallback_keys=["emb"]),
        f"{split_path.name}/pred_emb",
    )
    tgt_emb = squeeze_embedding_array(
        extract_dict_array(tgt_data, str(tgt_path), spec["tgt_key"], fallback_keys=["emb"]),
        f"{split_path.name}/tgt_emb",
    )

    buggy_global = extract_dict_array(buggy_data, str(buggy_path), global_indices_key).astype(np.int64)
    pred_global = extract_dict_array(pred_data, str(pred_path), global_indices_key).astype(np.int64)
    tgt_global = extract_dict_array(tgt_data, str(tgt_path), global_indices_key).astype(np.int64)

    num_samples = len(buggy_emb)
    if len(pred_emb) != num_samples or len(tgt_emb) != num_samples:
        raise ValueError(
            f"{split_dir}: embedding size mismatch: buggy={len(buggy_emb)} pred={len(pred_emb)} tgt={len(tgt_emb)}"
        )
    if len(buggy_global) != num_samples or len(pred_global) != num_samples or len(tgt_global) != num_samples:
        raise ValueError(f"{split_dir}: global_indices length does not match embeddings")
    if not np.array_equal(buggy_global, pred_global) or not np.array_equal(buggy_global, tgt_global):
        raise ValueError(f"{split_dir}: global_indices differ across embedding files")

    return LoadedSplitEmbeddings(
        buggy_emb=buggy_emb,
        pred_emb=pred_emb,
        tgt_emb=tgt_emb,
        global_indices=buggy_global,
    )


def normalize_label_item(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    return x


def build_multihot_from_label_lists(
    label_lists: Sequence[Sequence[Any]],
    label_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    cleaned: List[List[Any]] = []
    for labels in label_lists:
        if labels is None:
            cleaned.append([])
        else:
            cleaned.append([normalize_label_item(v) for v in labels])

    if label_to_idx is None:
        all_labels = sorted({str(label) for sample in cleaned for label in sample})
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    y = np.zeros((len(cleaned), len(label_to_idx)), dtype=np.float32)
    for i, labels in enumerate(cleaned):
        for label in labels:
            key = str(label)
            if key in label_to_idx:
                y[i, label_to_idx[key]] = 1.0
    return y, label_to_idx, idx_to_label


def load_label_lists_for_global_indices(
    dataset: Any,
    global_indices: np.ndarray,
    hf_label_field: str,
    split_name: str,
) -> List[List[Any]]:
    if len(global_indices) == 0:
        return []

    global_indices = global_indices.astype(np.int64)
    max_idx = int(np.max(global_indices))
    if max_idx >= len(dataset):
        raise IndexError(
            f"{split_name}: requested max dataset index {max_idx}, but dataset length is {len(dataset)}."
        )

    selected = dataset.select(global_indices.tolist())
    raw_label_lists = selected[hf_label_field]

    output: List[List[Any]] = []
    for labels in raw_label_lists:
        if labels is None:
            output.append([])
        else:
            output.append([normalize_label_item(v) for v in labels])
    return output


def load_multilabel_targets(
    global_indices: np.ndarray,
    hf_dataset_name: str,
    hf_split: str,
    hf_label_field: str,
    split_name: str,
    label_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    dataset = load_dataset(hf_dataset_name, split=hf_split)
    label_lists = load_label_lists_for_global_indices(dataset, global_indices, hf_label_field, split_name)
    return build_multihot_from_label_lists(label_lists, label_to_idx=label_to_idx)
