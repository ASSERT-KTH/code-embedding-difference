#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_jepa.py (refactored)

End-to-end JEPA-style training (exp2):
- Student/context encoder: ModernBERT (full / LoRA)
- Teacher/target encoder: EMA copy (stop-grad)
- Predictor: vit1d or mlp
- Loss: exp2 minimal (cosine align + variance regularizer)

Features:
- Config merge: configs/base.yaml + configs/exp*.yaml + CLI --set overrides
- Split by saved_indices/*.npy
- DDP via torchrun
- W&B logging (project/group/run_name placeholders allowed)

Expected files:
  models.py, losses.py, utils.py
"""

from __future__ import annotations

import os
import json
import time
import argparse
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm

from jepa.models import build_encoder, build_predictor
from jepa.losses import build_loss, retrieval_top1_acc, emb_std_mean
from jepa.utils import (
    load_yaml,
    deep_update,
    apply_overrides,
    save_resolved_config,
    is_ddp_env,
    is_main,
    rank,
    local_rank,
    ddp_setup,
    ddp_cleanup,
    ddp_all_reduce_sum,
    seed_everything,
    ema_update,
    unwrap_ddp,
    AverageMeter,
)

from datasets import load_dataset


# W&B is optional; only used if cfg["wandb"]["enabled"] = True
try:
    import wandb  # type: ignore
except Exception:
    wandb = None


# -------------------------
# Config resolve
# -------------------------
def resolve_config(exp_config_path: str, overrides: List[str]) -> Dict[str, Any]:
    base_path = os.path.join(os.path.dirname(exp_config_path), "base.yaml")
    base_cfg = load_yaml(base_path)
    exp_cfg = load_yaml(exp_config_path)
    cfg = deep_update(base_cfg, exp_cfg)
    cfg = apply_overrides(cfg, overrides)
    return cfg


# -------------------------
# Data
# -------------------------
def load_indices(indices_dir: str, filename: str) -> np.ndarray:
    return np.load(os.path.join(indices_dir, filename))


def to_device(batch_tok: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch_tok.items()}


# -------------------------
# RNG helpers (for more faithful resume)
# -------------------------
def get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        # per-device RNG states
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    else:
        state["cuda_rng_state_all"] = None
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    try:
        if state.get("python_random_state", None) is not None:
            random.setstate(state["python_random_state"])
        if state.get("numpy_random_state", None) is not None:
            np.random.set_state(state["numpy_random_state"])
        if state.get("torch_rng_state", None) is not None:
            torch.random.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state_all", None) is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    except Exception as e:
        # Best-effort: RNG restore is optional
        if is_main():
            print(f"[Resume] Warning: failed to restore RNG state: {e}")


# -------------------------
# Checkpoint
# -------------------------
def save_checkpoint(
    path: str,
    enc_ctx: nn.Module,
    enc_tgt: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    step: int,
    epoch: int,
    it: int,
    cfg: Dict[str, Any],
) -> None:
    ckpt = {
        "step": int(step),
        "epoch": int(epoch),
        "it": int(it),  # dataloader batch index inside the epoch (best-effort)
        "cfg": cfg,
        "enc_ctx": unwrap_ddp(enc_ctx).state_dict(),
        "enc_tgt": unwrap_ddp(enc_tgt).state_dict() if hasattr(enc_tgt, "state_dict") else enc_tgt.state_dict(),
        "predictor": unwrap_ddp(predictor).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": get_rng_state(),
    }
    torch.save(ckpt, path)


class JSONLLogger:
    def __init__(self, path: str):
        self.path = path
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -------------------------
# W&B
# -------------------------
def wandb_init_if_needed(cfg: Dict[str, Any], out_dir: str) -> None:
    wcfg = cfg.get("wandb", {})
    if not bool(wcfg.get("enabled", False)):
        return
    if wandb is None:
        raise RuntimeError("wandb enabled but wandb not installed.")
    if not is_main():
        return

    project = wcfg.get("project") or None
    group = wcfg.get("group") or None
    name = wcfg.get("run_name") or None
    entity = wcfg.get("entity") or None

    run_id = wcfg.get("id") or None
    resume_mode = wcfg.get("resume", "auto")  # "auto" or "must"

    wandb.init(
        entity=entity,
        project=project,
        group=group,
        name=name,
        id=run_id,
        resume=resume_mode,
        config=cfg,
        dir=out_dir,
    )


def wandb_log_if_needed(cfg: Dict[str, Any], metrics: Dict[str, Any], step: int) -> None:
    if not bool(cfg.get("wandb", {}).get("enabled", False)):
        return
    if wandb is None or not is_main():
        return
    wandb.log(metrics, step=step)


def wandb_finish_if_needed(cfg: Dict[str, Any]) -> None:
    if not bool(cfg.get("wandb", {}).get("enabled", False)):
        return
    if wandb is None or not is_main():
        return
    wandb.finish()


# -------------------------
# Train
# -------------------------
def train(cfg: Dict[str, Any]) -> None:
    # DDP init
    ddp_enabled = bool(cfg.get("ddp", {}).get("enabled", False)) and is_ddp_env()
    if ddp_enabled:
        ddp_setup(cfg.get("ddp", {}).get("backend", "nccl"))

    device = torch.device("cuda", local_rank()) if torch.cuda.is_available() else torch.device("cpu")

    # Seed (rank-shift)
    seed = int(cfg.get("seed", 42)) + rank()
    seed_everything(seed)

    # Output dir per run (good for sbatch arrays)
    run_name = cfg.get("run", {}).get("run_name", "run")
    save_root = cfg.get("run", {}).get("save_dir", "./checkpoints_jepa")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_folder = f"{run_name}_{ts}" + (f"_job{job_id}" if job_id else "")
    out_dir = os.path.join(save_root, run_folder)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    metrics_logger = None
    if is_main():
        metrics_logger = JSONLLogger(os.path.join(out_dir, "metrics.jsonl"))

    best_val_top1 = -1.0
    best_path = os.path.join(ckpt_dir, "ckpt_best.pt")
    last_path = os.path.join(ckpt_dir, "ckpt_last.pt")

    if is_main():
        os.makedirs(ckpt_dir, exist_ok=True)
        save_resolved_config(cfg, out_dir)

    # W&B
    wandb_init_if_needed(cfg, out_dir)

    # Tokenizer
    enc_name = cfg["encoder"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(enc_name, use_fast=True)

    # Data (HF + two-level indices)
    assert cfg["data"]["source"] == "hf", "This train script currently expects data.source=hf."

    hf_cfg = cfg["data"]["hf"]
    idx_cfg = cfg["data"]["indices"]

    dataset_id = hf_cfg["dataset_id"]
    split_name = hf_cfg.get("split", "train")

    indices_dir = idx_cfg["dir"]
    global_target_idx = load_indices(indices_dir, idx_cfg["global_target"])
    train_idx = load_indices(indices_dir, idx_cfg["train"])
    val_idx = load_indices(indices_dir, idx_cfg["val"])

    ds_full = load_dataset(dataset_id, split=split_name)
    ds_subset = ds_full.select(global_target_idx.tolist())
    ds_train = ds_subset.select(train_idx.tolist())
    ds_val = ds_subset.select(val_idx.tolist())

    buggy_key = hf_cfg["fields"]["buggy"]
    fixed_key = hf_cfg["fields"]["fixed"]

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

    train_sampler = DistributedSampler(ds_train, shuffle=True) if ddp_enabled else None
    val_sampler = DistributedSampler(ds_val, shuffle=False) if ddp_enabled else None

    dl_train = DataLoader(
        ds_train,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Models
    enc_ctx, emb_dim = build_encoder(cfg, device=device)
    if enc_ctx is None:
        raise ValueError("This train_jepa.py is end2end; encoder must not be None.")

    # Teacher encoder (EMA target): SAME architecture as student (including LoRA), but frozen
    enc_tgt, _ = build_encoder(cfg, device=device)
    enc_tgt.load_state_dict(unwrap_ddp(enc_ctx).state_dict(), strict=True)
    for p in enc_tgt.parameters():
        p.requires_grad = False
    enc_tgt.eval()

    predictor = build_predictor(cfg, emb_dim=emb_dim, device=device)

    # Wrap student + predictor with DDP
    if ddp_enabled:
        find_unused = bool(cfg.get("ddp", {}).get("find_unused_parameters", False))
        enc_ctx = DDP(enc_ctx, device_ids=[local_rank()], find_unused_parameters=find_unused)
        predictor = DDP(predictor, device_ids=[local_rank()], find_unused_parameters=find_unused)

    # -------------------------
    # Resume (part 1): load weights + global_step (+ epoch/it if present)
    # -------------------------
    resume_path = cfg["train"].get("resume_from", "") or ""
    resume_strict = bool(cfg["train"].get("resume_strict", True))
    restore_rng = bool(cfg["train"].get("resume_restore_rng", False))  # optional
    ckpt = None

    if resume_path:
        if is_main():
            print(f"[Resume] Loading checkpoint: {resume_path}")
        # ckpt = torch.load(resume_path, map_location="cpu")

        # (optional) DDP barrier for more stable shared-fs IO
        if ddp_enabled and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)

        if ddp_enabled and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        unwrap_ddp(enc_ctx).load_state_dict(ckpt["enc_ctx"], strict=resume_strict)
        enc_tgt.load_state_dict(ckpt["enc_tgt"], strict=resume_strict)
        unwrap_ddp(predictor).load_state_dict(ckpt["predictor"], strict=resume_strict)

        global_step = int(ckpt.get("step", 0))
        resume_epoch = int(ckpt.get("epoch", 0))
        resume_it = int(ckpt.get("it", 0))

        if restore_rng and ckpt.get("rng_state", None) is not None:
            set_rng_state(ckpt["rng_state"])

        if is_main():
            print(f"[Resume] global_step restored to {global_step}")
            print(f"[Resume] epoch/it from ckpt: epoch={resume_epoch}, it={resume_it}")
    else:
        global_step = 0
        resume_epoch = 0
        resume_it = 0

    # Loss
    loss_fn = build_loss(cfg).to(device)

    # Optimizer (separate LR for encoder vs predictor)
    lr_encoder = float(cfg["train"].get("lr_encoder", cfg["train"].get("lr", 2e-5)))
    lr_predictor = float(cfg["train"].get("lr_predictor", cfg["train"].get("lr", 2e-5)))
    wd = float(cfg["train"].get("weight_decay", 0.01))
    betas = cfg.get("optim", {}).get("betas", [0.9, 0.999])
    eps = float(cfg.get("optim", {}).get("eps", 1e-8))

    enc_params = [p for p in unwrap_ddp(enc_ctx).parameters() if p.requires_grad]
    pred_params = [p for p in unwrap_ddp(predictor).parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": lr_encoder},
            {"params": pred_params, "lr": lr_predictor},
        ],
        weight_decay=wd,
        betas=tuple(betas),
        eps=eps,
    )

    # AMP
    use_fp16 = bool(cfg["train"].get("fp16", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # -------------------------
    # Resume (part 2): load optimizer/scaler (must be after creation)
    # -------------------------
    if ckpt is not None:
        if ckpt.get("optimizer", None) is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler", None) is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if is_main():
            print("[Resume] optimizer/scaler restored.")



    # Train settings
    epochs = int(cfg["train"]["epochs"])
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    log_every = int(cfg["train"].get("log_every", 50))
    tau = float(cfg.get("ema", {}).get("tau", 0.996))
    save_ckpt = bool(cfg["train"].get("save_ckpt", True))

    init_path = os.path.join(ckpt_dir, "ckpt_init.pt")

    if save_ckpt and is_main() and ckpt is None:
        save_checkpoint(
            init_path,
            enc_ctx,
            enc_tgt,
            predictor,
            optimizer,
            scaler,
            step=0,
            epoch=0,
            it=0,
            cfg=cfg,
        )
        print(f"[Init] Saved initial checkpoint to {init_path}")

    save_every_epoch = bool(cfg["train"].get("save_every_epoch", True))
    eval_every_steps = int(cfg["train"].get("eval_every_steps", 0))  # 0 means disabled
    save_every_steps = int(cfg["train"].get("save_every_steps", 0))  # 0 means disabled

    # ---------------------------------------------------------
    # IMPORTANT FIX:
    # Do NOT reset global_step here. Previously you had: global_step = 0
    # That overwrote the resumed step and made it look like training restarted.
    # ---------------------------------------------------------

    def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
        return F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()

    def run_validation(epoch: int, global_step: int) -> Dict[str, float]:
        enc_ctx.eval()
        predictor.eval()
        enc_tgt.eval()
        if ddp_enabled and val_sampler is not None:
            val_sampler.set_epoch(epoch)

        v_loss = AverageMeter()
        v_align = AverageMeter()
        v_var = AverageMeter()
        v_top1 = AverageMeter()
        v_std_ctx = AverageMeter()
        v_std_tgt = AverageMeter()
        v_cos = AverageMeter()

        with torch.no_grad():
            for tok_buggy, tok_fixed in dl_val:
                tok_buggy = to_device(tok_buggy, device)
                tok_fixed = to_device(tok_fixed, device)

                z_ctx = enc_ctx(tok_buggy["input_ids"], tok_buggy["attention_mask"])
                z_tgt = enc_tgt(tok_fixed["input_ids"], tok_fixed["attention_mask"])
                z_pred = predictor(z_ctx)

                out = loss_fn(z_ctx=z_ctx, z_pred=z_pred, z_tgt=z_tgt)

                bsz = int(z_ctx.size(0))
                top1 = retrieval_top1_acc(z_pred, z_tgt).item()
                std_ctx = emb_std_mean(z_ctx).item()
                std_tgt = emb_std_mean(z_tgt).item()
                cos = cosine_mean(z_pred, z_tgt)

                v_loss.update(out["loss"].item(), bsz)
                v_align.update(out["align"].item(), bsz)
                v_var.update(out["var"].item(), bsz)
                v_top1.update(top1, bsz)
                v_std_ctx.update(std_ctx, bsz)
                v_std_tgt.update(std_tgt, bsz)
                v_cos.update(cos, bsz)

        # DDP reduce
        t = torch.tensor(
            [
                v_loss.sum, v_loss.count,
                v_align.sum, v_align.count,
                v_var.sum, v_var.count,
                v_top1.sum, v_top1.count,
                v_std_ctx.sum, v_std_ctx.count,
                v_std_tgt.sum, v_std_tgt.count,
                v_cos.sum, v_cos.count,
            ],
            device=device,
            dtype=torch.float64,
        )
        ddp_all_reduce_sum(t)

        vals = t.tolist()

        def avg(s, c):
            return s / max(1.0, c)

        metrics = {
            "val_loss": avg(vals[0], vals[1]),
            "val_align": avg(vals[2], vals[3]),
            "val_var": avg(vals[4], vals[5]),
            "val_top1": avg(vals[6], vals[7]),
            "val_std_ctx": avg(vals[8], vals[9]),
            "val_std_tgt": avg(vals[10], vals[11]),
            "val_cos": avg(vals[12], vals[13]),
        }

        if is_main():
            print(
                f"[ep {epoch} step {global_step}] VAL "
                f"loss={metrics['val_loss']:.4f} align={metrics['val_align']:.4f} var={metrics['val_var']:.4f} "
                f"top1={metrics['val_top1']:.3f} std_ctx={metrics['val_std_ctx']:.3f} std_tgt={metrics['val_std_tgt']:.3f} "
                f"cos={metrics['val_cos']:.3f}"
            )

            wandb_log_if_needed(
                cfg,
                {
                    "val/loss": metrics["val_loss"],
                    "val/align": metrics["val_align"],
                    "val/var": metrics["val_var"],
                    "val/top1": metrics["val_top1"],
                    "val/std_ctx": metrics["val_std_ctx"],
                    "val/std_tgt": metrics["val_std_tgt"],
                    "val/cos": metrics["val_cos"],
                    "epoch": epoch,
                },
                step=global_step,
            )

            if metrics_logger is not None:
                metrics_logger.log(
                    {
                        "split": "val",
                        "step": global_step,
                        "epoch": epoch,
                        "loss": metrics["val_loss"],
                        "align": metrics["val_align"],
                        "var": metrics["val_var"],
                        "top1": metrics["val_top1"],
                        "std_ctx": metrics["val_std_ctx"],
                        "std_tgt": metrics["val_std_tgt"],
                        "cos": metrics["val_cos"],
                        "time": time.time(),
                    }
                )

        return metrics

    # ---------------------------------------------------------
    # Resume-aware epoch + dataloader skipping
    # ---------------------------------------------------------
    # global_step counts optimizer steps (after grad_accum).
    steps_per_epoch = max(1, (len(dl_train) // max(1, grad_accum)))
    start_epoch = global_step // steps_per_epoch
    start_step_in_epoch = global_step % steps_per_epoch

    if is_main():
        print(f"[Resume] steps_per_epoch={steps_per_epoch} (len(dl_train)={len(dl_train)}, grad_accum={grad_accum})")
        print(f"[Resume] start_epoch={start_epoch}, start_step_in_epoch={start_step_in_epoch}")

    for epoch in range(start_epoch, epochs):
        if ddp_enabled:
            train_sampler.set_epoch(epoch)

        enc_ctx.train()
        predictor.train()

        meter_loss = AverageMeter()
        meter_align = AverageMeter()
        meter_var = AverageMeter()
        meter_top1 = AverageMeter()
        meter_std_ctx = AverageMeter()
        meter_std_tgt = AverageMeter()
        meter_cos = AverageMeter()

        optimizer.zero_grad(set_to_none=True)

        # Only skip for the very first resumed epoch.
        skip_batches = 0
        if epoch == start_epoch and start_step_in_epoch > 0:
            skip_batches = start_step_in_epoch * grad_accum
            if is_main():
                print(f"[Resume] Skipping {skip_batches} batches to align with global_step={global_step}.")

        if is_main():
            iter_dl = tqdm(dl_train, desc=f"Epoch {epoch}", dynamic_ncols=True)
        else:
            iter_dl = dl_train

        for it, (tok_buggy, tok_fixed) in enumerate(iter_dl):
            if skip_batches > 0 and it < skip_batches:
                continue

            tok_buggy = to_device(tok_buggy, device)
            tok_fixed = to_device(tok_fixed, device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                z_ctx = enc_ctx(tok_buggy["input_ids"], tok_buggy["attention_mask"])
                with torch.no_grad():
                    z_tgt = enc_tgt(tok_fixed["input_ids"], tok_fixed["attention_mask"])
                z_pred = predictor(z_ctx)

                out = loss_fn(z_ctx=z_ctx, z_pred=z_pred, z_tgt=z_tgt)
                loss = out["loss"] / grad_accum

            scaler.scale(loss).backward()

            # Step
            if (it + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # EMA update teacher
                ema_update(enc_tgt, unwrap_ddp(enc_ctx), tau=tau)

                global_step += 1

                # ---- step-based checkpoint saving ----
                if save_ckpt and save_every_steps > 0 and (global_step % save_every_steps == 0) and is_main():
                    save_checkpoint(
                        os.path.join(ckpt_dir, f"ckpt_step{global_step}.pt"),
                        enc_ctx, enc_tgt, predictor, optimizer, scaler,
                        global_step, epoch, it, cfg
                    )

                # ---- step-based validation + best checkpoint ----
                if eval_every_steps > 0 and (global_step % eval_every_steps == 0):
                    metrics = run_validation(epoch=epoch, global_step=global_step)

                    if save_ckpt and is_main() and metrics["val_top1"] > best_val_top1:
                        best_val_top1 = metrics["val_top1"]
                        save_checkpoint(
                            best_path,
                            enc_ctx, enc_tgt, predictor, optimizer, scaler,
                            global_step, epoch, it, cfg
                        )
                        print(f"[ep {epoch} step {global_step}] New BEST ckpt: val_top1={best_val_top1:.3f}")
                        wandb_log_if_needed(cfg, {"best/val_top1": best_val_top1, "best/epoch": epoch}, step=global_step)

                    # resume train mode
                    enc_ctx.train()
                    predictor.train()

                # Metrics (local)
                with torch.no_grad():
                    top1 = retrieval_top1_acc(z_pred, z_tgt).item()
                    std_ctx = emb_std_mean(z_ctx).item()
                    std_tgt = emb_std_mean(z_tgt).item()
                    cos = cosine_mean(z_pred, z_tgt)

                bsz = int(z_ctx.size(0))
                meter_loss.update(out["loss"].item(), bsz)
                meter_align.update(out["align"].item(), bsz)
                meter_var.update(out["var"].item(), bsz)
                meter_top1.update(top1, bsz)
                meter_std_ctx.update(std_ctx, bsz)
                meter_std_tgt.update(std_tgt, bsz)
                meter_cos.update(cos, bsz)

                if global_step % log_every == 0:
                    # DDP reduce meters (sum,count) using a tensor
                    t = torch.tensor(
                        [
                            meter_loss.sum, meter_loss.count,
                            meter_align.sum, meter_align.count,
                            meter_var.sum, meter_var.count,
                            meter_top1.sum, meter_top1.count,
                            meter_std_ctx.sum, meter_std_ctx.count,
                            meter_std_tgt.sum, meter_std_tgt.count,
                            meter_cos.sum, meter_cos.count,
                        ],
                        device=device,
                        dtype=torch.float64,
                    )
                    ddp_all_reduce_sum(t)

                    if is_main():
                        vals = t.tolist()

                        def avg(s, c):
                            return s / max(1.0, c)

                        tr_loss = avg(vals[0], vals[1])
                        tr_align = avg(vals[2], vals[3])
                        tr_var = avg(vals[4], vals[5])
                        tr_top1 = avg(vals[6], vals[7])
                        tr_std_ctx = avg(vals[8], vals[9])
                        tr_std_tgt = avg(vals[10], vals[11])
                        tr_cos = avg(vals[12], vals[13])

                        print(
                            f"[ep {epoch} step {global_step}] "
                            f"train loss={tr_loss:.4f} align={tr_align:.4f} var={tr_var:.4f} "
                            f"top1={tr_top1:.3f} std_ctx={tr_std_ctx:.3f} std_tgt={tr_std_tgt:.3f} "
                            f"cos={tr_cos:.3f}"
                        )

                        wandb_log_if_needed(
                            cfg,
                            {
                                "train/loss": tr_loss,
                                "train/align": tr_align,
                                "train/var": tr_var,
                                "train/top1": tr_top1,
                                "train/std_ctx": tr_std_ctx,
                                "train/std_tgt": tr_std_tgt,
                                "train/cos": tr_cos,
                                "epoch": epoch,
                                "lr/encoder": optimizer.param_groups[0]["lr"],
                                "lr/predictor": optimizer.param_groups[1]["lr"],
                            },
                            step=global_step,
                        )

                        if metrics_logger is not None:
                            metrics_logger.log(
                                {
                                    "split": "train",
                                    "step": global_step,
                                    "epoch": epoch,
                                    "loss": tr_loss,
                                    "align": tr_align,
                                    "var": tr_var,
                                    "top1": tr_top1,
                                    "std_ctx": tr_std_ctx,
                                    "std_tgt": tr_std_tgt,
                                    "cos": tr_cos,
                                    "lr_encoder": optimizer.param_groups[0]["lr"],
                                    "lr_predictor": optimizer.param_groups[1]["lr"],
                                    "time": time.time(),
                                }
                            )

        # after finishing this epoch, no more skipping
        start_step_in_epoch = 0

        # Optional: save last checkpoint each epoch
        if save_ckpt and is_main() and save_every_epoch:
            save_checkpoint(
                os.path.join(ckpt_dir, f"ckpt_epoch{epoch}.pt"),
                enc_ctx, enc_tgt, predictor, optimizer, scaler,
                global_step, epoch, it, cfg
            )
            # also update a "last" symlink-ish file
            save_checkpoint(
                last_path,
                enc_ctx, enc_tgt, predictor, optimizer, scaler,
                global_step, epoch, it, cfg
            )
            if is_main():
                print(f"[ep {epoch}] Saved epoch ckpt + last ckpt at step={global_step}")

    if is_main():
        print("Training done.")
    wandb_finish_if_needed(cfg)

    if ddp_enabled:
        ddp_cleanup()


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to exp config YAML (e.g. configs/exp2_lora_vit.yaml)")
    ap.add_argument("--set", type=str, action="append", default=[], help="Override config, e.g. --set train.lr=1e-5")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = resolve_config(args.config, args.set)
    train(cfg)


if __name__ == "__main__":
    main()
