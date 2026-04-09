#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embed_jepa.py

General embedding extractor for JEPA exp2 (train/val/test/any split indices).

It reuses the same logic as your test_jepa.py, but:
- you choose which indices file to embed via cfg["embed"]["split"] ("train"/"val"/"test")
  OR cfg["embed"]["indices_file"] (explicit .npy filename under indices_dir)
- output keys/filenames are generic: global_indices (not "global_test_indices")
- saves dicts exactly in your preferred format:
    torch.save({"z_pred": emb, "global_indices": idx}, "z_pred.rank{r}.pt")
    torch.save({"z_tgt": emb,  "global_indices": idx}, "z_tgt.rank{r}.pt")
- merges on rank0 into:
    z_pred.pt / z_tgt.pt / metrics.json

Typical use:
  python -m torch.distributed.run --nproc_per_node=4 embed_jepa.py \
    --config configs/exp2_lora_encoder_predictor.yaml \
    --set embed.ckpt_path=/.../ckpt_best.pt \
    --set embed.save_path=/.../emb_train \
    --set embed.split=train

Or explicitly:
    --set embed.indices_file=train_idx.npy
"""

from __future__ import annotations

import os
import time
import argparse
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from jepa.models import build_encoder, build_predictor
from jepa.losses import retrieval_top1_acc, emb_std_mean
from jepa.utils import (
    apply_overrides,
    ddp_all_reduce_sum,
    ddp_cleanup,
    ddp_setup,
    deep_update,
    is_ddp_env,
    is_main,
    load_json,
    load_pt,
    load_yaml,
    local_rank,
    rank,
    save_json,
    save_pt,
    unwrap_ddp,
)


# -------------------------
# Config resolve (same as train/test)
# -------------------------
def resolve_config(exp_config_path: str, overrides: List[str]) -> Dict[str, Any]:
    base_path = os.path.join(os.path.dirname(exp_config_path), "base.yaml")
    base_cfg = load_yaml(base_path)
    exp_cfg = load_yaml(exp_config_path)
    cfg = deep_update(base_cfg, exp_cfg)
    cfg = apply_overrides(cfg, overrides)
    return cfg


# -------------------------
# Data helpers
# -------------------------
def load_indices(path: str) -> np.ndarray:
    return np.load(path)

def to_device(batch_tok: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch_tok.items()}


def try_merge_on_rank0(save_path: str, world: int) -> None:
    if not is_main():
        return

    def merge_dict_shards(prefix: str, out_name: str, emb_key: str) -> None:
        parts: List[Dict[str, Any]] = []
        for r in range(world):
            p = os.path.join(save_path, f"{prefix}.rank{r}.pt")
            if not os.path.exists(p):
                print(f"[Merge] missing shard: {p}")
                return
            d = load_pt(p)
            if not isinstance(d, dict) or ("global_indices" not in d) or (emb_key not in d):
                raise ValueError(
                    f"[Merge] bad shard format: {p} keys={list(d.keys()) if isinstance(d, dict) else type(d)}"
                )
            parts.append(d)

        gidx = torch.cat([d["global_indices"] for d in parts], dim=0)
        emb = torch.cat([d[emb_key] for d in parts], dim=0)

        out_file = os.path.join(save_path, out_name)
        torch.save({emb_key: emb, "global_indices": gidx}, out_file)
        print(f"[Merge] wrote: {out_file}")

    merge_dict_shards("z_ctx", "z_ctx.pt", "z_ctx")
    # merge_dict_shards("z_pred", "z_pred.pt", "z_pred")
    # merge_dict_shards("z_tgt", "z_tgt.pt", "z_tgt")

    # merge metrics
    m_parts = []
    for r in range(world):
        mp = os.path.join(save_path, f"metrics.rank{r}.json")
        if os.path.exists(mp):
            m_parts.append(load_json(mp))
    if m_parts:
        merged_m = {"per_rank": m_parts}
        keys = set.intersection(*[set(m.keys()) for m in m_parts]) if len(m_parts) > 1 else set(m_parts[0].keys())
        avg = {}
        for k in keys:
            if isinstance(m_parts[0][k], (int, float)):
                avg[k] = float(sum(m[k] for m in m_parts)) / max(1, len(m_parts))
        if avg:
            merged_m["avg"] = avg
        save_json(merged_m, os.path.join(save_path, "metrics.json"))
        print(f"[Merge] wrote: {os.path.join(save_path, 'metrics.json')}")


# -------------------------
# Core: choose indices file
# -------------------------
def resolve_indices_filename(idx_cfg: Dict[str, Any], embed_cfg: Dict[str, Any]) -> str:
    """
    Priority:
      1) embed.indices_file (explicit filename)
      2) embed.split in {"train","val","test"} mapping to idx_cfg keys or defaults
    """
    if embed_cfg.get("indices_file"):
        return str(embed_cfg["indices_file"])

    split = str(embed_cfg.get("split", "test")).lower()
    # Prefer config-provided filenames if present
    if split == "train":
        return str(idx_cfg.get("train", "train_idx.npy"))
    if split == "val":
        return str(idx_cfg.get("val", "val_idx.npy"))
    if split == "test":
        return str(idx_cfg.get("test", "test_idx.npy"))

    raise ValueError(f"Unknown embed.split='{split}'. Use train/val/test or set embed.indices_file.")


# -------------------------
# Embed
# -------------------------
@torch.no_grad()
def run_embed(cfg: Dict[str, Any]) -> None:
    ddp_enabled = bool(cfg.get("ddp", {}).get("enabled", False)) and is_ddp_env()
    if ddp_enabled:
        ddp_setup(cfg.get("ddp", {}).get("backend", "nccl"))

    device = torch.device("cuda", local_rank()) if torch.cuda.is_available() else torch.device("cpu")
    world = int(os.environ.get("WORLD_SIZE", "1"))

    embed_cfg = cfg.get("embed", {})
    ckpt_path = str(embed_cfg.get("ckpt_path", ""))
    if not ckpt_path:
        raise ValueError("Missing cfg.embed.ckpt_path (path to ckpt_last.pt / ckpt_best.pt / ckpt_epoch*.pt).")

    save_path = str(embed_cfg.get("save_path", "./embed_outputs"))
    save_dtype = str(embed_cfg.get("save_dtype", "float16")).lower()
    max_items = int(embed_cfg.get("max_items", -1))

    os.makedirs(save_path, exist_ok=True)

    # data
    assert cfg["data"]["source"] == "hf", "This script expects data.source=hf."
    hf_cfg = cfg["data"]["hf"]
    idx_cfg = cfg["data"]["indices"]

    dataset_id = hf_cfg["dataset_id"]
    split_name = hf_cfg.get("split", "train")
    buggy_key = hf_cfg["fields"]["buggy"]
    fixed_key = hf_cfg["fields"]["fixed"]

    indices_dir = idx_cfg["dir"]
    global_target = load_indices(os.path.join(indices_dir, idx_cfg["global_target"]))

    idx_filename = resolve_indices_filename(idx_cfg, embed_cfg)
    subset_idx = load_indices(os.path.join(indices_dir, idx_filename))

    # ds_full global indices aligned with ds_subset_selected order
    global_indices_all = global_target[subset_idx]  # np.ndarray [N_subset]

    # load HF dataset
    ds_full = load_dataset(dataset_id, split=split_name)
    ds_subset = ds_full.select(global_target.tolist())
    ds_sel = ds_subset.select(subset_idx.tolist())

    # tokenizer (encoder tokenizer)
    enc_name = cfg["encoder"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(enc_name, use_fast=True)

    def collate_fn(batch):
        buggy = [str(x.get(buggy_key, "")) if x.get(buggy_key, None) is not None else "" for x in batch]
        fixed = [str(x.get(fixed_key, "")) if x.get(fixed_key, None) is not None else "" for x in batch]
        tok_buggy = tokenizer(
            buggy,
            padding=True,
            truncation=True,
            max_length=int(cfg["encoder"]["max_len"]),
            return_tensors="pt",
        )
        tok_fixed = tokenizer(
            fixed,
            padding=True,
            truncation=True,
            max_length=int(cfg["encoder"]["max_len"]),
            return_tensors="pt",
        )
        return tok_buggy, tok_fixed

    sampler = DistributedSampler(ds_sel, shuffle=False) if ddp_enabled else None
    dl = DataLoader(
        ds_sel,
        batch_size=int(embed_cfg.get("batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        sampler=sampler,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # build models
    enc_ctx, emb_dim = build_encoder(cfg, device=device)
    if enc_ctx is None:
        raise ValueError("End2end embed expects encoder != None.")
    enc_tgt, _ = build_encoder(cfg, device=device)
    predictor = build_predictor(cfg, emb_dim=emb_dim, device=device)

    if ddp_enabled:
        find_unused = bool(cfg.get("ddp", {}).get("find_unused_parameters", False))
        enc_ctx = DDP(enc_ctx, device_ids=[local_rank()], find_unused_parameters=find_unused)
        predictor = DDP(predictor, device_ids=[local_rank()], find_unused_parameters=find_unused)

    # load checkpoint
    if is_main():
        print(f"[Embed] Loading ckpt: {ckpt_path}")
        print(f"[Embed] indices_file: {idx_filename}  (N={len(ds_sel)})")
        print(f"[Embed] save_path: {save_path}")

    if ddp_enabled and torch.distributed.is_initialized():
        torch.distributed.barrier()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ddp_enabled and torch.distributed.is_initialized():
        torch.distributed.barrier()

    unwrap_ddp(enc_ctx).load_state_dict(ckpt["enc_ctx"], strict=True)
    enc_tgt.load_state_dict(ckpt["enc_tgt"], strict=True)
    unwrap_ddp(predictor).load_state_dict(ckpt["predictor"], strict=True)

    enc_ctx.eval()
    predictor.eval()
    enc_tgt.eval()
    for p in enc_tgt.parameters():
        p.requires_grad = False

    if save_dtype == "float16":
        out_cast = torch.float16
    elif save_dtype == "float32":
        out_cast = torch.float32
    else:
        raise ValueError("cfg.embed.save_dtype must be 'float16' or 'float32'")

    # storage
    out_ctx: List[torch.Tensor] = []
    # out_pred: List[torch.Tensor] = []
    # out_tgt: List[torch.Tensor] = []
    out_gidx: List[int] = []


    # metrics meters (rank-local)
    m_top1_sum = 0.0
    m_cos_sum = 0.0
    m_std_pred_sum = 0.0
    m_std_tgt_sum = 0.0
    m_n = 0

    def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.float()
        b = b.float()
        a = torch.nn.functional.normalize(a, dim=-1)
        b = torch.nn.functional.normalize(b, dim=-1)
        return (a * b).sum(dim=-1).mean().item()

    if ddp_enabled and sampler is not None:
        sampler.set_epoch(0)

    it_dl = tqdm(dl, desc="Embed", dynamic_ncols=True) if is_main() else dl

    # IMPORTANT: track position within THIS rank's sampler order (rank-local)
    pos_in_rank = 0

    for (tok_buggy, tok_fixed) in it_dl:
        if max_items > 0 and m_n >= max_items:
            break

        tok_buggy = to_device(tok_buggy, device)
        tok_fixed = to_device(tok_fixed, device)

        z_ctx = enc_ctx(tok_buggy["input_ids"], tok_buggy["attention_mask"])
        # z_pred = predictor(z_ctx)
        # z_tgt = enc_tgt(tok_fixed["input_ids"], tok_fixed["attention_mask"])

        # metrics (batch-level)
        # top1 = retrieval_top1_acc(z_pred, z_tgt).item()
        # cos = cosine_mean(z_pred, z_tgt)
        # std_pred = emb_std_mean(z_pred).item()
        # std_tgt = emb_std_mean(z_tgt).item()

        bsz = int(z_ctx.size(0))
        # m_top1_sum += top1 * bsz
        # m_cos_sum += cos * bsz
        # m_std_pred_sum += std_pred * bsz
        # m_std_tgt_sum += std_tgt * bsz
        m_n += bsz

        # save tensors
        out_ctx.append(z_ctx.detach().to("cpu", dtype=out_cast))
        # out_pred.append(z_pred.detach().to("cpu", dtype=out_cast))
        # out_tgt.append(z_tgt.detach().to("cpu", dtype=out_cast))

        # save ds_full GLOBAL indices aligned with embeddings
        ds_pos = np.arange(pos_in_rank, pos_in_rank + bsz, dtype=np.int64)
        pos_in_rank += bsz
        gidx = global_indices_all[ds_pos]
        out_gidx.extend(gidx.tolist())

    # concat
    z_ctx_all = torch.cat(out_ctx, dim=0) if out_ctx else torch.empty((0, emb_dim), dtype=out_cast)
    # z_pred_all = torch.cat(out_pred, dim=0) if out_pred else torch.empty((0, emb_dim), dtype=out_cast)
    # z_tgt_all = torch.cat(out_tgt, dim=0) if out_tgt else torch.empty((0, emb_dim), dtype=out_cast)
    gidx_all = torch.tensor(out_gidx, dtype=torch.long)

    # save per-rank shards as dicts (format aligned with your request)
    ctx_path = os.path.join(save_path, f"z_ctx.rank{rank()}.pt")
    # pred_path = os.path.join(save_path, f"z_pred.rank{rank()}.pt")
    # tgt_path = os.path.join(save_path, f"z_tgt.rank{rank()}.pt")

    save_pt({"z_ctx": z_ctx_all, "global_indices": gidx_all}, ctx_path)
    # save_pt({"z_pred": z_pred_all, "global_indices": gidx_all}, pred_path)
    # save_pt({"z_tgt": z_tgt_all, "global_indices": gidx_all}, tgt_path)

    if is_main():
        # print(f"[Embed] wrote shards:\n  {ctx_path}\n  {pred_path}\n  {tgt_path}")
        print(f"[Embed] wrote shards:\n  {ctx_path}")

    # reduce metrics across ranks
    # t = torch.tensor(
    #     [m_top1_sum, m_cos_sum, m_std_pred_sum, m_std_tgt_sum, float(m_n)],
    #     device=device,
    #     dtype=torch.float64,
    # )
    # ddp_all_reduce_sum(t)
    # top1_avg = (t[0] / t[4]).item() if t[4].item() > 0 else 0.0
    # cos_avg = (t[1] / t[4]).item() if t[4].item() > 0 else 0.0
    # std_pred_avg = (t[2] / t[4]).item() if t[4].item() > 0 else 0.0
    # std_tgt_avg = (t[3] / t[4]).item() if t[4].item() > 0 else 0.0

    # metrics = {
    #     "top1": float(top1_avg),
    #     "cos": float(cos_avg),
    #     "std_pred": float(std_pred_avg),
    #     "std_tgt": float(std_tgt_avg),
    #     "n": int(t[4].item()),
    #     "indices_file": idx_filename,
    # }

    t = torch.tensor([float(m_n)], device=device, dtype=torch.float64)
    ddp_all_reduce_sum(t)

    metrics = {
        "n": int(t[0].item()),
        "indices_file": idx_filename,
    }

    save_json(metrics, os.path.join(save_path, f"metrics.rank{rank()}.json"))
    if is_main():
        print(f"[Embed] metrics: {metrics}")

    # merge on rank0
    if ddp_enabled and torch.distributed.is_initialized():
        torch.distributed.barrier()
    try_merge_on_rank0(save_path=save_path, world=world)

    if ddp_enabled:
        ddp_cleanup()


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to exp config YAML")
    ap.add_argument("--set", type=str, action="append", default=[], help="Override config, e.g. --set embed.split=train")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = resolve_config(args.config, args.set)
    run_embed(cfg)

if __name__ == "__main__":
    main()
