from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from common import BinaryMLP, build_binary_feature, load_method_split_embeddings, load_online_split_embeddings, save_json, seed_everything


try:
    import wandb  # type: ignore
except Exception:
    wandb = None


class BinaryFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, meta: Optional[List[Dict[str, Any]]] = None) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.meta = meta
        if len(self.features) != len(self.labels):
            raise ValueError("features and labels must have the same length")
        if self.meta is not None and len(self.meta) != len(self.features):
            raise ValueError("meta and features must have the same length")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.features[idx]).float()
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.meta is None:
            return x, y
        return x, y, self.meta[idx]


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
        for batch in tqdm(dataloader, desc=f"[{split_name}]", leave=False):
            x, y = batch[:2]
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

    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0,), dtype=np.float32)
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0,), dtype=np.float32)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = float(running_loss / max(total, 1))
    return metrics


def build_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
) -> List[Dict[str, Any]]:
    model.eval()
    outputs: List[Dict[str, Any]] = []

    with torch.no_grad():
        for x, _, meta_batch in tqdm(dataloader, desc="[Predict]", leave=False):
            x = x.to(device, non_blocking=True)
            probs = torch.sigmoid(model(x)).detach().cpu().numpy()
            global_indices = meta_batch["global_index"]
            sources = meta_batch["source"]
            for global_index, source, prob in zip(global_indices, sources, probs):
                pred_label = int(prob >= threshold)
                outputs.append(
                    {
                        "global_index": int(global_index),
                        "source": source,
                        "probability_buggy": float(prob),
                        "predicted_label": pred_label,
                        "target_label": 1 if source == "buggy" else 0,
                    }
                )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train binary buggy-vs-fixed classifier probe")
    parser.add_argument("--backend", type=str, default="offline_pt", choices=["offline_pt", "online_checkpoint"])
    parser.add_argument("--method", type=str, default="e2", choices=["e1", "e2", "seq_emb"])
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--embedding-view", type=str, default="ctx", choices=["ctx", "tgt", "ctx_tgt"])
    parser.add_argument("--jepa-checkpoint", type=str, default="")
    parser.add_argument("--jepa-config", type=str, default="")
    parser.add_argument("--encoder-mode", type=str, default="same_encoder", choices=["same_encoder", "context_target", "target_target"])
    parser.add_argument("--indices-dir", type=str, default="")
    parser.add_argument("--global-target-dir", type=str, default="")
    parser.add_argument("--global-target-file", type=str, default="global_target_indices.npy")
    parser.add_argument("--train-indices", type=str, default="train_idx.npy")
    parser.add_argument("--val-indices", type=str, default="val_idx.npy")
    parser.add_argument("--test-indices", type=str, default="test_idx.npy")
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
    parser.add_argument("--wandb-group", type=str, default="bug_detection")
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

    train_x, train_y, _ = load_split_binary_data(args.backend, args.method, args.train_dir, args.embedding_view, file_kwargs, online_kwargs)
    val_x, val_y, _ = load_split_binary_data(args.backend, args.method, args.val_dir, args.embedding_view, file_kwargs, online_kwargs)
    test_x, test_y, test_meta = load_split_binary_data(args.backend, args.method, args.test_dir, args.embedding_view, file_kwargs, online_kwargs)

    train_ds = BinaryFeatureDataset(train_x, train_y)
    val_ds = BinaryFeatureDataset(val_x, val_y)
    test_ds = BinaryFeatureDataset(test_x, test_y)
    pred_test_ds = BinaryFeatureDataset(test_x, test_y, meta=test_meta)

    train_dl = make_dataloader(train_ds, args.batch_size, True, args.num_workers)
    val_dl = make_dataloader(val_ds, args.batch_size, False, args.num_workers)
    test_dl = make_dataloader(test_ds, args.batch_size, False, args.num_workers)
    pred_test_dl = make_dataloader(pred_test_ds, args.batch_size, False, args.num_workers)

    input_dim = int(train_x.shape[1])
    model = BinaryMLP(input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
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
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:03d} | train={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if args.use_wandb and wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/precision": val_metrics["precision"],
                    "val/recall": val_metrics["recall"],
                    "val/f1": val_metrics["f1"],
                },
                step=epoch,
            )

        ckpt = {
            "model": model.state_dict(),
            "input_dim": input_dim,
            "threshold": args.threshold,
            "epoch": epoch,
            "backend": args.backend,
            "method": args.method,
            "embedding_view": args.embedding_view,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        }
        torch.save(ckpt, last_ckpt_path)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            ckpt["best_val_f1"] = best_val_f1
            torch.save(ckpt, best_ckpt_path)

    save_json(history, Path(args.output_dir) / "history.json")

    best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    best_model = BinaryMLP(
        best_ckpt["input_dim"],
        hidden_dim=int(best_ckpt.get("hidden_dim", args.hidden_dim)),
        dropout=float(best_ckpt.get("dropout", args.dropout)),
    )
    best_model.load_state_dict(best_ckpt["model"])
    best_model = best_model.to(device)

    val_metrics = evaluate(best_model, val_dl, criterion, device, args.threshold, "Val-best")
    test_metrics = evaluate(best_model, test_dl, criterion, device, args.threshold, "Test-best")
    test_predictions = build_predictions(best_model, pred_test_dl, device, args.threshold)

    final_metrics = {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "val": val_metrics,
        "test": test_metrics,
    }
    save_json(final_metrics, Path(args.output_dir) / "final_metrics.json")
    save_json(test_predictions, Path(args.output_dir) / "test_predictions.json")

    if args.use_wandb and wandb is not None and wandb.run is not None:
        wandb.run.summary["best_val_f1"] = best_val_f1
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["test_accuracy"] = test_metrics["accuracy"]
        wandb.run.summary["test_f1"] = test_metrics["f1"]
        wandb.finish()

    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
