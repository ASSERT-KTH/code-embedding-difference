from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from common import (
    EDIT_TYPE_FEATURE_MODES,
    MultiLabelMLP,
    build_edit_type_feature,
    infer_edit_type_input_dim,
    load_method_split_embeddings,
    load_multilabel_targets,
    load_online_split_embeddings,
    save_json,
    seed_everything,
)


try:
    import wandb  # type: ignore
except Exception:
    wandb = None


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        if self.labels is not None and len(self.features) != len(self.labels):
            raise ValueError("features and labels must have the same length")

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
) -> Tuple[np.ndarray, np.ndarray]:
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
            indices_file=str(online_kwargs["indices_file_by_split"][split_dir]),
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


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    split_name: str,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    total = 0
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=f"[{split_name}]", leave=False):
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

    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 0), dtype=np.float32)
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, 0), dtype=np.float32)
    metrics = compute_f1_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = float(running_loss / max(total, 1))
    return metrics


def build_predictions(
    model: nn.Module,
    features: np.ndarray,
    global_indices: np.ndarray,
    idx_to_label: Dict[int, str],
    batch_size: int,
    device: torch.device,
    threshold: float,
) -> List[Dict[str, Any]]:
    model.eval()
    ds = FeatureDataset(features, labels=None)
    dl = make_dataloader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs: List[Dict[str, Any]] = []

    cursor = 0
    with torch.no_grad():
        for x, _ in tqdm(dl, desc="[Predict]", leave=False):
            x = x.to(device, non_blocking=True)
            probs = torch.sigmoid(model(x)).detach().cpu().numpy()
            bs = probs.shape[0]
            for global_index, prob in zip(global_indices[cursor:cursor + bs], probs):
                pred_indices = np.where(prob >= threshold)[0].tolist()
                outputs.append(
                    {
                        "global_index": int(global_index),
                        "predicted_label_indices": pred_indices,
                        "predicted_labels": [idx_to_label[i] for i in pred_indices],
                        "probabilities": prob.tolist(),
                    }
                )
            cursor += bs
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train edit-type multi-label classifier probe")
    parser.add_argument("--backend", type=str, default="offline_pt", choices=["offline_pt", "online_checkpoint"])
    parser.add_argument("--method", type=str, default="e2", choices=["e1", "e2", "seq_emb"])
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--embedding-view", type=str, default="ctx", choices=["ctx", "tgt", "ctx_tgt"])
    parser.add_argument("--feature-mode", type=str, default="tgt_minus_buggy", choices=list(EDIT_TYPE_FEATURE_MODES))
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
    parser.add_argument("--train-indices", type=str, default="train_idx.npy")
    parser.add_argument("--val-indices", type=str, default="val_idx.npy")
    parser.add_argument("--test-indices", type=str, default="test_idx.npy")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--extract-batch-size", type=int, default=128)
    parser.add_argument("--extract-num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--use-wandb", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="CodeRepair_JEPA_downstream")
    parser.add_argument("--wandb-entity", type=str, default="assert-kth")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="edit_type")
    parser.add_argument("--wandb-id", type=str, default="")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

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
        "indices_file_by_split": {
            args.train_dir: args.train_indices,
            args.val_dir: args.val_indices,
            args.test_dir: args.test_indices,
        },
    }

    train_x, train_global = load_split_features(args.backend, args.method, args.train_dir, args.embedding_view, args.feature_mode, file_kwargs, online_kwargs)
    val_x, val_global = load_split_features(args.backend, args.method, args.val_dir, args.embedding_view, args.feature_mode, file_kwargs, online_kwargs)
    test_x, test_global = load_split_features(args.backend, args.method, args.test_dir, args.embedding_view, args.feature_mode, file_kwargs, online_kwargs)

    y_train, label_to_idx, idx_to_label = load_multilabel_targets(
        train_global, args.hf_dataset_name, args.hf_split, args.hf_label_field, "train"
    )
    y_val, _, _ = load_multilabel_targets(
        val_global, args.hf_dataset_name, args.hf_split, args.hf_label_field, "val", label_to_idx=label_to_idx
    )
    y_test, _, _ = load_multilabel_targets(
        test_global, args.hf_dataset_name, args.hf_split, args.hf_label_field, "test", label_to_idx=label_to_idx
    )

    save_json(
        {
            "label_to_idx": label_to_idx,
            "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
            "num_labels": len(label_to_idx),
            "backend": args.backend,
            "method": args.method,
            "embedding_view": args.embedding_view,
            "feature_mode": args.feature_mode,
        },
        Path(args.output_dir) / "label_map.json",
    )

    train_ds = FeatureDataset(train_x, y_train)
    val_ds = FeatureDataset(val_x, y_val)
    test_ds = FeatureDataset(test_x, y_test)

    train_dl = make_dataloader(train_ds, args.batch_size, True, args.num_workers)
    val_dl = make_dataloader(val_ds, args.batch_size, False, args.num_workers)
    test_dl = make_dataloader(test_ds, args.batch_size, False, args.num_workers)

    input_dim = infer_edit_type_input_dim(args.feature_mode, train_x.shape[1])
    model = MultiLabelMLP(input_dim, len(label_to_idx), hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_micro = -1.0
    best_epoch = -1
    history: List[Dict[str, Any]] = []
    best_ckpt_path = Path(args.output_dir) / "best_model.pt"
    last_ckpt_path = Path(args.output_dir) / "last_model.pt"

    if args.use_wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project or None,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            group=args.wandb_group or None,
            id=args.wandb_id or None,
            resume="allow" if args.wandb_id else None,
            config={
                "backend": args.backend,
                "method": args.method,
                "embedding_view": args.embedding_view,
                "feature_mode": args.feature_mode,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for x, y in tqdm(train_dl, desc=f"[Train {epoch}/{args.epochs}]", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total += bs
            running_loss += loss.item() * bs

        train_loss = float(running_loss / max(total, 1))
        val_metrics = evaluate(model, val_dl, criterion, device, args.threshold, "Val")

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_micro_f1": val_metrics["micro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:03d} | train={train_loss:.4f} "
            f"val_micro={val_metrics['micro_f1']:.4f} val_macro={val_metrics['macro_f1']:.4f}"
        )

        if args.use_wandb and wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_metrics["loss"],
                    "val/micro_f1": val_metrics["micro_f1"],
                    "val/macro_f1": val_metrics["macro_f1"],
                },
                step=epoch,
            )

        ckpt = {
            "model": model.state_dict(),
            "input_dim": input_dim,
            "num_labels": len(label_to_idx),
            "threshold": args.threshold,
            "epoch": epoch,
            "backend": args.backend,
            "method": args.method,
            "embedding_view": args.embedding_view,
            "feature_mode": args.feature_mode,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        }
        torch.save(ckpt, last_ckpt_path)

        if val_metrics["micro_f1"] > best_val_micro:
            best_val_micro = val_metrics["micro_f1"]
            best_epoch = epoch
            ckpt["best_val_micro_f1"] = best_val_micro
            torch.save(ckpt, best_ckpt_path)

    save_json(history, Path(args.output_dir) / "history.json")

    best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    best_model = MultiLabelMLP(
        best_ckpt["input_dim"],
        best_ckpt["num_labels"],
        hidden_dim=int(best_ckpt.get("hidden_dim", args.hidden_dim)),
        dropout=float(best_ckpt.get("dropout", args.dropout)),
    )
    best_model.load_state_dict(best_ckpt["model"])
    best_model = best_model.to(device)

    val_metrics = evaluate(best_model, val_dl, criterion, device, args.threshold, "Val-best")
    test_metrics = evaluate(best_model, test_dl, criterion, device, args.threshold, "Test-best")
    test_predictions = build_predictions(
        best_model,
        test_x,
        test_global,
        idx_to_label,
        args.batch_size,
        device,
        args.threshold,
    )

    final_metrics = {
        "best_val_micro_f1": best_val_micro,
        "best_epoch": best_epoch,
        "val": val_metrics,
        "test": test_metrics,
    }
    save_json(final_metrics, Path(args.output_dir) / "final_metrics.json")
    save_json(test_predictions, Path(args.output_dir) / "test_predictions.json")

    if args.use_wandb and wandb is not None and wandb.run is not None:
        wandb.run.summary["best_val_micro_f1"] = best_val_micro
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["test_micro_f1"] = test_metrics["micro_f1"]
        wandb.run.summary["test_macro_f1"] = test_metrics["macro_f1"]
        wandb.finish()

    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
