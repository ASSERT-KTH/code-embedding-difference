from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from common import MultiLabelMLP, build_edit_type_feature, load_method_split_embeddings, load_multilabel_targets, load_online_split_embeddings, save_json


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray | None = None) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.features[idx]).float()
        if self.labels is None:
            return x, int(idx)
        y = torch.from_numpy(self.labels[idx]).float()
        return x, y


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


def compute_f1_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    y_true = y_true.astype(np.int32)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"micro_f1": float(micro), "macro_f1": float(macro)}


def load_split_features(
    backend: str,
    method: str,
    split_dir: str,
    embedding_view: str,
    feature_mode: str,
    file_kwargs: Dict[str, str],
    online_kwargs: Dict[str, Any],
):
    if backend == "offline_pt":
        loaded = load_method_split_embeddings(
            split_dir,
            method=method,
            embedding_view=embedding_view,
            **file_kwargs,
        )
        features = build_edit_type_feature(loaded.buggy_emb, loaded.tgt_emb, feature_mode)
        return features, loaded.global_indices
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
        features = build_edit_type_feature(loaded.buggy_emb, loaded.tgt_emb, feature_mode)
        return features, loaded.global_indices
    raise ValueError(f"Unsupported backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test edit-type multi-label classifier probe")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--backend", type=str, default="", choices=["", "offline_pt", "online_checkpoint"])
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--embedding-view", type=str, default="", choices=["", "ctx", "tgt", "ctx_tgt"])
    parser.add_argument("--feature-mode", type=str, default="")
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
    parser.add_argument("--hf-dataset-name", type=str, default="ASSERT-KTH/RunBugRun-Final")
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--hf-label-field", type=str, default="labels")
    parser.add_argument("--hf-buggy-field", type=str, default="")
    parser.add_argument("--hf-fixed-field", type=str, default="")
    parser.add_argument("--jepa-checkpoint", type=str, default="")
    parser.add_argument("--jepa-config", type=str, default="")
    parser.add_argument("--encoder-mode", type=str, default="same_encoder", choices=["same_encoder", "context_target", "target_target"])
    parser.add_argument("--indices-dir", type=str, default="")
    parser.add_argument("--global-target-dir", type=str, default="")
    parser.add_argument("--global-target-file", type=str, default="global_target_indices.npy")
    parser.add_argument("--indices-file", type=str, default="test_idx.npy")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--extract-num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backend = args.backend or ckpt.get("backend", "offline_pt")
    method = args.method or ckpt.get("method", "e2")
    embedding_view = args.embedding_view or ckpt.get("embedding_view", "ctx")
    feature_mode = args.feature_mode or ckpt.get("feature_mode", "tgt_minus_buggy")
    threshold = float(ckpt.get("threshold", 0.5))
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    features, global_indices = load_split_features(backend, method, args.split_dir, embedding_view, feature_mode, file_kwargs, online_kwargs)
    labels, _, idx_to_label = load_multilabel_targets(
        global_indices,
        args.hf_dataset_name,
        args.hf_split,
        args.hf_label_field,
        split_name=Path(args.split_dir).name,
    )

    ds = FeatureDataset(features, labels)
    dl = make_dataloader(ds, args.batch_size, False, args.num_workers)

    model = MultiLabelMLP(
        ckpt["input_dim"],
        ckpt["num_labels"],
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

    with torch.no_grad():
        for x, y in tqdm(dl, desc="[Eval]", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)
            bs = x.size(0)
            total += bs
            running_loss += loss.item() * bs
            all_probs.append(probs.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    metrics = compute_f1_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = float(running_loss / max(total, 1))

    predictions: List[Dict[str, Any]] = []
    for global_index, prob in zip(global_indices, y_prob):
        pred_indices = np.where(prob >= threshold)[0].tolist()
        predictions.append(
            {
                "global_index": int(global_index),
                "predicted_label_indices": pred_indices,
                "predicted_labels": [idx_to_label[i] for i in pred_indices],
                "probabilities": prob.tolist(),
            }
        )

    output = {
        "backend": backend,
        "method": method,
        "embedding_view": embedding_view,
        "feature_mode": feature_mode,
        "metrics": metrics,
        "predictions": predictions,
    }
    save_json(output, args.output_json)
    print(json.dumps(output["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
