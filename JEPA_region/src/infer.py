from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from models import build_encoder, build_predictor, load_encoder_tunable_state
from utils import (
    apply_overrides,
    ddp_barrier,
    ddp_cleanup,
    ddp_enabled,
    ddp_local_rank,
    ddp_setup,
    deep_update,
    is_main,
    load_yaml,
    masked_mean_pool,
    rank,
    save_json,
)


def resolve_config(exp_config_path: str, overrides: List[str]) -> Dict[str, Any]:
    base_path = os.path.join(os.path.dirname(exp_config_path), "base.yaml")
    base_cfg = load_yaml(base_path)
    exp_cfg = load_yaml(exp_config_path)
    cfg = deep_update(base_cfg, exp_cfg)
    cfg = apply_overrides(cfg, overrides)
    return cfg


def flatten_overrides(raw_overrides: List[Any]) -> List[str]:
    flat: List[str] = []
    for item in raw_overrides:
        if isinstance(item, (list, tuple)):
            flat.extend(str(x) for x in item)
        else:
            flat.append(str(item))
    return flat


def load_indices(indices_dir: str, filename: str) -> np.ndarray:
    return np.load(os.path.join(indices_dir, filename))


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def save_pt(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


def load_pt(path: str) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def resolve_indices_filename(idx_cfg: Dict[str, Any], infer_cfg: Dict[str, Any], split: str) -> str:
    indices_map = infer_cfg.get("indices_file_by_split", {})
    if isinstance(indices_map, dict) and indices_map.get(split):
        return str(indices_map[split])
    if infer_cfg.get("indices_file"):
        return str(infer_cfg["indices_file"])

    split = str(split).lower()
    if split == "train":
        return str(idx_cfg.get("train", "train_idx.npy"))
    if split == "val":
        return str(idx_cfg.get("val", "val_idx.npy"))
    if split == "test":
        return str(idx_cfg.get("test", "test_idx.npy"))
    raise ValueError(f"Unknown infer.split='{split}'")


def _normalize_dtype_tensor(x: torch.Tensor, save_dtype: str) -> torch.Tensor:
    if save_dtype == "float16":
        return x.half()
    if save_dtype == "bfloat16":
        return x.bfloat16()
    return x.float()


def _sort_payload_by_indices(payload: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    gidx = payload["global_indices"]
    order = torch.argsort(gidx)
    return {
        "emb": payload["emb"][order],
        "global_indices": gidx[order],
    }


def try_merge_on_rank0(split_dir: str, world: int, file_names: List[str]) -> None:
    if not is_main():
        return

    def merge(file_name: str) -> None:
        parts = []
        for r in range(world):
            path = os.path.join(split_dir, f"{file_name}.rank{r}.pt")
            if not os.path.exists(path):
                print(f"[Merge] missing shard: {path}")
                return
            parts.append(load_pt(path))

        payload = {
            "emb": torch.cat([part["emb"] for part in parts], dim=0),
            "global_indices": torch.cat([part["global_indices"] for part in parts], dim=0),
        }
        payload = _sort_payload_by_indices(payload)
        torch.save(payload, os.path.join(split_dir, f"{file_name}.pt"))
        print(f"[Merge] wrote: {os.path.join(split_dir, f'{file_name}.pt')}")

    for file_name in file_names:
        merge(file_name)


def merge_runtime_with_checkpoint_cfg(runtime_cfg: Dict[str, Any], ckpt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deep_update({}, ckpt_cfg)
    for key in ["infer", "ddp"]:
        if key in runtime_cfg:
            cfg[key] = deep_update(cfg.get(key, {}), runtime_cfg[key])
    if "data" in runtime_cfg:
        cfg["data"] = deep_update(cfg.get("data", {}), runtime_cfg["data"])
    return cfg


@torch.no_grad()
def run_infer(cfg: Dict[str, Any]) -> None:
    infer_cfg = cfg.get("infer", {})
    ckpt_path = str(infer_cfg.get("ckpt_path", ""))
    if not ckpt_path:
        raise ValueError("infer.ckpt_path must be set.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("cfg")
    if isinstance(ckpt_cfg, dict):
        cfg = merge_runtime_with_checkpoint_cfg(cfg, ckpt_cfg)
        infer_cfg = cfg.get("infer", {})

    use_ddp = bool(cfg.get("ddp", {}).get("enabled", False)) and ddp_enabled()
    if use_ddp:
        ddp_setup(cfg.get("ddp", {}).get("backend", "nccl"))

    device = torch.device("cuda", ddp_local_rank()) if torch.cuda.is_available() else torch.device("cpu")
    save_path = str(infer_cfg.get("save_path", "./infer_outputs"))
    save_dtype = str(infer_cfg.get("save_dtype", "float16")).lower()
    max_items = int(infer_cfg.get("max_items", -1))
    os.makedirs(save_path, exist_ok=True)
    split_list = infer_cfg.get("splits")
    if split_list:
        splits = [str(x).lower() for x in split_list]
    else:
        splits = [str(infer_cfg.get("split", "test")).lower()]

    file_names = ["buggy_ctx", "fixed_ctx", "buggy_tgt", "fixed_tgt", "pred"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"]["name"], use_fast=True)

    enc_ctx, hidden_dim = build_encoder(cfg, device=device)
    enc_tgt, _ = build_encoder(cfg, device=device)
    predictor = build_predictor(cfg, hidden_dim=hidden_dim, device=device)

    if "enc_ctx" in ckpt:
        enc_ctx.load_state_dict(ckpt["enc_ctx"], strict=True)
    elif "enc_ctx_tunable" in ckpt:
        load_encoder_tunable_state(enc_ctx, ckpt["enc_ctx_tunable"], strict=False)
    else:
        raise KeyError("Checkpoint is missing encoder context weights.")

    if "enc_tgt" in ckpt:
        enc_tgt.load_state_dict(ckpt["enc_tgt"], strict=True)
    elif "enc_tgt_tunable" in ckpt:
        load_encoder_tunable_state(enc_tgt, ckpt["enc_tgt_tunable"], strict=False)
    elif "enc_ctx" in ckpt:
        enc_tgt.load_state_dict(ckpt["enc_ctx"], strict=True)
    elif "enc_ctx_tunable" in ckpt:
        load_encoder_tunable_state(enc_tgt, ckpt["enc_ctx_tunable"], strict=False)
    else:
        raise KeyError("Checkpoint is missing encoder target weights.")

    predictor.load_state_dict(ckpt["predictor"], strict=True)
    for p in enc_tgt.parameters():
        p.requires_grad = False
    enc_ctx.eval()
    enc_tgt.eval()
    predictor.eval()

    manifest = {
        "splits": splits,
        "files": file_names,
        "format": {"emb_key": "emb", "global_indices_key": "global_indices"},
        "save_dtype": save_dtype,
        "checkpoint": os.path.abspath(ckpt_path),
        "cfg_source": "checkpoint_cfg+runtime_infer_overrides" if isinstance(ckpt_cfg, dict) else "runtime_cfg",
    }

    for split in splits:
        split_dir = os.path.join(save_path, split)
        os.makedirs(split_dir, exist_ok=True)
        dl, _ = build_dataloader(cfg, tokenizer, use_ddp, infer_cfg, split)

        buffers = {name: [] for name in file_names}
        gidx_all = []
        total_saved = 0

        iterator = tqdm(dl, desc=f"InferLast-{split}", dynamic_ncols=True) if is_main() else dl
        for batch in iterator:
            tok_buggy, tok_fixed, global_indices = batch
            tok_buggy = to_device(tok_buggy, device)
            tok_fixed = to_device(tok_fixed, device)

            buggy_ctx_seq = enc_ctx(tok_buggy["input_ids"], tok_buggy["attention_mask"])
            fixed_ctx_seq = enc_ctx(tok_fixed["input_ids"], tok_fixed["attention_mask"])
            buggy_tgt_seq = enc_tgt(tok_buggy["input_ids"], tok_buggy["attention_mask"])
            fixed_tgt_seq = enc_tgt(tok_fixed["input_ids"], tok_fixed["attention_mask"])
            pred_seq = predictor(buggy_ctx_seq, attention_mask=tok_buggy["attention_mask"])

            buggy_mask = tok_buggy["attention_mask"].bool()
            fixed_mask = tok_fixed["attention_mask"].bool()

            pooled = {
                "buggy_ctx": masked_mean_pool(buggy_ctx_seq, buggy_mask),
                "fixed_ctx": masked_mean_pool(fixed_ctx_seq, fixed_mask),
                "buggy_tgt": masked_mean_pool(buggy_tgt_seq, buggy_mask),
                "fixed_tgt": masked_mean_pool(fixed_tgt_seq, fixed_mask),
                "pred": masked_mean_pool(pred_seq, buggy_mask),
            }

            if max_items > 0:
                remaining = max_items - total_saved
                if remaining <= 0:
                    break
                for key in pooled:
                    pooled[key] = pooled[key][:remaining]
                global_indices = global_indices[:remaining]

            for key in pooled:
                buffers[key].append(_normalize_dtype_tensor(pooled[key], save_dtype).cpu())
            gidx_all.append(global_indices.cpu())
            total_saved += int(global_indices.numel())

        if not gidx_all:
            raise RuntimeError(f"No samples were processed during inference for split='{split}'.")

        global_indices_tensor = torch.cat(gidx_all, dim=0)
        for key in file_names:
            payload = {
                "emb": torch.cat(buffers[key], dim=0),
                "global_indices": global_indices_tensor,
            }
            payload = _sort_payload_by_indices(payload)
            save_pt(payload, os.path.join(split_dir, f"{key}.rank{rank()}.pt"))

        if use_ddp:
            ddp_barrier()
        try_merge_on_rank0(split_dir, world=int(os.environ.get("WORLD_SIZE", "1")), file_names=file_names)
        if use_ddp:
            ddp_barrier()

    if is_main():
        save_json(manifest, os.path.join(save_path, "manifest.json"))
    if use_ddp:
        ddp_cleanup()


def build_dataloader(cfg: Dict[str, Any], tokenizer, use_ddp: bool, infer_cfg: Dict[str, Any], split: str):
    assert cfg["data"]["source"] == "hf", "This infer script expects data.source=hf."
    hf_cfg = cfg["data"]["hf"]
    idx_cfg = cfg["data"]["indices"]

    dataset_id = hf_cfg["dataset_id"]
    split_name = hf_cfg.get("split", "train")
    buggy_key = hf_cfg["fields"]["buggy"]
    fixed_key = hf_cfg["fields"]["fixed"]
    max_len = int(cfg["encoder"]["max_len"])

    global_target_dir = idx_cfg["global_target_dir"]
    split_dir = idx_cfg["split_dir"]
    global_target = load_indices(global_target_dir, idx_cfg["global_target"])
    idx_filename = resolve_indices_filename(idx_cfg, infer_cfg, split)
    subset_idx = load_indices(split_dir, idx_filename)
    global_indices_all = global_target[subset_idx]

    ds_full = load_dataset(dataset_id, split=split_name)
    ds_subset = ds_full.select(global_target.tolist())
    ds_selected = ds_subset.select(subset_idx.tolist())

    def collate_fn(batch):
        buggy = [str(x.get(buggy_key, "") or "") for x in batch]
        fixed = [str(x.get(fixed_key, "") or "") for x in batch]
        indices = torch.tensor([int(x["_global_index"]) for x in batch], dtype=torch.long)
        tok_buggy = tokenizer(
            buggy,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        tok_fixed = tokenizer(
            fixed,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return tok_buggy, tok_fixed, indices

    ds_selected = ds_selected.add_column("_global_index", global_indices_all.tolist())
    sampler = DistributedSampler(ds_selected, shuffle=False) if use_ddp else None
    dl = DataLoader(
        ds_selected,
        batch_size=int(infer_cfg.get("batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        sampler=sampler,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return dl, sampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--set", nargs="*", action="append", default=[])
    args = parser.parse_args()

    cfg = resolve_config(args.config, flatten_overrides(args.set))
    run_infer(cfg)


if __name__ == "__main__":
    main()
