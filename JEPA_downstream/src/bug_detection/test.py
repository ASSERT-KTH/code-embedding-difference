from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from common import BinaryMLP, build_binary_feature, load_method_split_embeddings, load_online_split_embeddings, save_json


class BinaryFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, meta: List[Dict[str, Any]]) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.meta = meta

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.features[idx]).float(), torch.tensor(self.labels[idx], dtype=torch.float32), self.meta[idx]


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def build_binary_split(features_buggy: np.ndarray, features_fixed: np.ndarray, global_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    x_buggy = build_binary_feature(features_buggy, feature_mode="buggy")
    x_fixed = build_binary_feature(features_fixed, feature_mode="buggy")
    y_buggy = np.ones((len(x_buggy),), dtype=np.float32)
    y_fixed = np.zeros((len(x_fixed),), dtype=np.float32)
    x = np.concatenate([x_buggy, x_fixed], axis=0).astype(np.float32)
    y = np.concatenate([y_buggy, y_fixed], axis=0).astype(np.float32)
    meta: List[Dict[str, Any]] = []
    for idx in global_indices:
        meta.append({"global_index": int(idx), "source": "buggy"})
    for idx in global_indices:
        meta.append({"global_index": int(idx), "source": "fixed"})
    return x, y, meta


def load_split_binary_data(
    backend: str,
    method: str,
    split_dir: str,
    embedding_view: str,
    file_kwargs: Dict[str, str],
    online_kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    if backend == "offline_pt":
        loaded = load_method_split_embeddings(
            split_dir,
            method=method,
            embedding_view=embedding_view,
            **file_kwargs,
        )
        return build_binary_split(loaded.buggy_emb, loaded.tgt_emb, loaded.global_indices)
    if backend == "online_checkpoint":
        loaded = load_online_split_embeddings(
            checkpoint_path=str(online_kwargs["jepa_checkpoint"]),
            config_path=str(online_kwargs["jepa_config"]),
            encoder_mode=str(online_kwargs["encoder_mode"]),
            indices_dir=str(online_kwargs["indices_dir"]),
            indices_file=str(online_kwargs["indices_file"]),
            global_target_dir=str(online_kwargs["global_target_dir"]),
            global_target_file=str(online_kwargs["global_target_file"]),
            hf_dataset_name=str(online_kwargs["hf_dataset_name"]),
            hf_split=str(online_kwargs["hf_split"]),
            hf_buggy_field=str(online_kwargs["hf_buggy_field"]),
            hf_fixed_field=str(online_kwargs["hf_fixed_field"]),
            max_len=int(online_kwargs["max_len"]),
            batch_size=int(online_kwargs["extract_batch_size"]),
            num_workers=int(online_kwargs["extract_num_workers"]),
            device=online_kwargs["device"],
        )
        return build_binary_split(loaded.buggy_emb, loaded.tgt_emb, loaded.global_indices)
    raise ValueError(f"Unsupported backend: {backend}")


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    y_true = y_true.astype(np.int32)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test binary buggy-vs-fixed classifier probe")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--backend", type=str, default="", choices=["", "offline_pt", "online_checkpoint"])
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--embedding-view", type=str, default="", choices=["", "ctx", "tgt", "ctx_tgt"])
    parser.add_argument("--jepa-checkpoint", type=str, default="")
    parser.add_argument("--jepa-config", type=str, default="")
    parser.add_argument("--encoder-mode", type=str, default="same_encoder", choices=["same_encoder", "context_target", "target_target"])
    parser.add_argument("--indices-dir", type=str, default="")
    parser.add_argument("--global-target-dir", type=str, default="")
    parser.add_argument("--global-target-file", type=str, default="global_target_indices.npy")
    parser.add_argument("--indices-file", type=str, default="test_idx.npy")
    parser.add_argument("--hf-dataset-name", type=str, default="")
    parser.add_argument("--hf-split", type=str, default="")
    parser.add_argument("--hf-buggy-field", type=str, default="")
    parser.add_argument("--hf-fixed-field", type=str, default="")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--extract-num-workers", type=int, default=4)
    parser.add_argument("--global-indices-key", type=str, default="global_indices")
    parser.add_argument("--buggy-file-name", type=str, default="")
    parser.add_argument("--buggy-key", type=str, default="")
    parser.add_argument("--pred-file-name", type=str, default="")
    parser.add_argument("--pred-key", type=str, default="")
    parser.add_argument("--tgt-file-name", type=str, default="")
    parser.add_argument("--tgt-key", type=str, default="")
    parser.add_argument("--buggy-ctx-file-name", type=str, default="")
    parser.add_argument("--buggy-ctx-key", type=str, default="")
    parser.add_argument("--fixed-ctx-file-name", type=str, default="")
    parser.add_argument("--fixed-ctx-key", type=str, default="")
    parser.add_argument("--buggy-tgt-file-name", type=str, default="")
    parser.add_argument("--buggy-tgt-key", type=str, default="")
    parser.add_argument("--fixed-tgt-file-name", type=str, default="")
    parser.add_argument("--fixed-tgt-key", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backend = args.backend or ckpt.get("backend", "offline_pt")
    method = args.method or ckpt.get("method", "e2")
    embedding_view = args.embedding_view or ckpt.get("embedding_view", "ctx")
    threshold = float(ckpt.get("threshold", 0.5))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_kwargs = {
        "global_indices_key": args.global_indices_key,
        "buggy_file_name": args.buggy_file_name,
        "buggy_key": args.buggy_key,
        "pred_file_name": args.pred_file_name,
        "pred_key": args.pred_key,
        "tgt_file_name": args.tgt_file_name,
        "tgt_key": args.tgt_key,
        "buggy_ctx_file_name": args.buggy_ctx_file_name,
        "buggy_ctx_key": args.buggy_ctx_key,
        "fixed_ctx_file_name": args.fixed_ctx_file_name,
        "fixed_ctx_key": args.fixed_ctx_key,
        "buggy_tgt_file_name": args.buggy_tgt_file_name,
        "buggy_tgt_key": args.buggy_tgt_key,
        "fixed_tgt_file_name": args.fixed_tgt_file_name,
        "fixed_tgt_key": args.fixed_tgt_key,
    }
    online_kwargs = {
        "jepa_checkpoint": args.jepa_checkpoint,
        "jepa_config": args.jepa_config,
        "encoder_mode": args.encoder_mode,
        "indices_dir": args.indices_dir,
        "indices_file": args.indices_file,
        "global_target_dir": args.global_target_dir,
        "global_target_file": args.global_target_file,
        "hf_dataset_name": args.hf_dataset_name,
        "hf_split": args.hf_split,
        "hf_buggy_field": args.hf_buggy_field,
        "hf_fixed_field": args.hf_fixed_field,
        "max_len": args.max_len,
        "extract_batch_size": args.extract_batch_size,
        "extract_num_workers": args.extract_num_workers,
        "device": device,
    }

    x, y, meta = load_split_binary_data(backend, method, args.split_dir, embedding_view, file_kwargs, online_kwargs)
    ds = BinaryFeatureDataset(x, y, meta)
    dl = make_dataloader(ds, args.batch_size, False, args.num_workers)
    model = BinaryMLP(
        ckpt["input_dim"],
        hidden_dim=int(ckpt.get("hidden_dim", 512)),
        dropout=float(ckpt.get("dropout", 0.3)),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    total = 0
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    predictions: List[Dict[str, Any]] = []

    with torch.no_grad():
        for x_batch, y_batch, meta_batch in tqdm(dl, desc="[Eval]", leave=False):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            bs = x_batch.size(0)
            total += bs
            running_loss += loss.item() * bs
            all_probs.append(probs)
            all_targets.append(y_batch.detach().cpu().numpy())

            global_indices = meta_batch["global_index"]
            sources = meta_batch["source"]
            for global_index, source, prob in zip(global_indices, sources, probs):
                predictions.append(
                    {
                        "global_index": int(global_index),
                        "source": source,
                        "probability_buggy": float(prob),
                        "predicted_label": int(prob >= threshold),
                        "target_label": 1 if source == "buggy" else 0,
                    }
                )

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = float(running_loss / max(total, 1))

    output = {
        "backend": backend,
        "method": method,
        "embedding_view": embedding_view,
        "metrics": metrics,
        "predictions": predictions,
    }
    save_json(output, args.output_json)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
