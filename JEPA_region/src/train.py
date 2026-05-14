from __future__ import annotations

import argparse
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from losses import build_loss
from models import build_encoder, build_predictor, export_encoder_tunable_state, load_encoder_tunable_state
from utils import (
    AverageMeter,
    JSONLLogger,
    apply_overrides,
    ddp_all_reduce_sum,
    ddp_cleanup,
    ddp_enabled,
    ddp_local_rank,
    ddp_setup,
    deep_update,
    ema_update,
    find_change_regions_in_batch,
    is_main,
    load_yaml,
    pool_change_regions,
    rank,
    save_resolved_config,
    seed_everything,
    span_mask,
    unwrap_ddp,
)


try:
    import wandb  # type: ignore
except Exception:
    wandb = None


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


def get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    else:
        state["cuda_rng_state_all"] = None
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    if state.get("python_random_state") is not None:
        random.setstate(state["python_random_state"])
    if state.get("numpy_random_state") is not None:
        np.random.set_state(state["numpy_random_state"])
    if state.get("torch_rng_state") is not None:
        torch.random.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def grad_global_norm(module: torch.nn.Module) -> float:
    total = 0.0
    found = False
    for param in module.parameters():
        if param.grad is None:
            continue
        g = param.grad.detach()
        total += float(torch.sum(g.float() * g.float()).item())
        found = True
    if not found:
        return 0.0
    return total ** 0.5


def save_checkpoint(
    path: str,
    enc_ctx: torch.nn.Module,
    enc_tgt: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    empty_region_emb: Optional[torch.Tensor],
    step: int,
    epoch: int,
    it: int,
    cfg: Dict[str, Any],
) -> None:
    ckpt = {
        "step": int(step),
        "epoch": int(epoch),
        "it": int(it),
        "cfg": cfg,
        "enc_ctx": unwrap_ddp(enc_ctx).state_dict(),
        "enc_tgt": unwrap_ddp(enc_tgt).state_dict(),
        "predictor": unwrap_ddp(predictor).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": get_rng_state(),
        "empty_region_emb": empty_region_emb.detach().to("cpu") if empty_region_emb is not None else None,
    }
    torch.save(ckpt, path)


def save_inference_checkpoint(
    path: str,
    enc_ctx: torch.nn.Module,
    enc_tgt: torch.nn.Module,
    predictor: torch.nn.Module,
    empty_region_emb: Optional[torch.Tensor],
    step: int,
    epoch: int,
    cfg: Dict[str, Any],
) -> None:
    ckpt = {
        "step": int(step),
        "epoch": int(epoch),
        "cfg": cfg,
        "enc_ctx_tunable": export_encoder_tunable_state(unwrap_ddp(enc_ctx)),
        "enc_tgt_tunable": export_encoder_tunable_state(unwrap_ddp(enc_tgt)),
        "predictor": unwrap_ddp(predictor).state_dict(),
        "empty_region_emb": empty_region_emb.detach().to("cpu") if empty_region_emb is not None else None,
    }
    torch.save(ckpt, path)


def retrieval_top1_acc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = F.normalize(pred.float(), dim=-1, eps=eps)
    target = F.normalize(target.float(), dim=-1, eps=eps)
    sim = pred @ target.t()
    pred_idx = sim.argmax(dim=1)
    gt_idx = torch.arange(sim.size(0), device=sim.device)
    return (pred_idx == gt_idx).float().mean()


def emb_std_mean(x: torch.Tensor) -> torch.Tensor:
    return x.float().std(dim=0, unbiased=False).mean()


def masked_full_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def weighted_mean_pool(hidden: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights.to(dtype=hidden.dtype).unsqueeze(-1)
    summed = (hidden * w).sum(dim=1)
    denom = w.sum(dim=1).clamp_min(1e-8)
    return summed / denom


def build_change_plus_shared_weights(
    attention_mask: torch.Tensor,
    regions,
    side: str,
    shared_weight: float,
) -> torch.Tensor:
    seq_len = attention_mask.size(1)
    weights = attention_mask.to(dtype=torch.float32) * float(shared_weight)
    for i, region in enumerate(regions):
        if side == "buggy":
            start, end = region.buggy_start, region.buggy_end
        elif side == "fixed":
            start, end = region.fixed_start, region.fixed_end
        else:
            raise ValueError(f"Unknown side='{side}'")
        change_mask = span_mask(seq_len, start, end, device=attention_mask.device).to(dtype=torch.float32)
        weights[i] = torch.maximum(weights[i], change_mask)
    return weights


def get_supervision_pair(
    cfg: Dict[str, Any],
    tok_buggy: Dict[str, torch.Tensor],
    tok_fixed: Dict[str, torch.Tensor],
    z_pred: torch.Tensor,
    z_tgt: torch.Tensor,
    empty_region_emb: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    supervision_target = str(cfg.get("loss", {}).get("supervision_target", "change_region")).lower()
    if supervision_target == "change_region":
        regions = find_change_regions_in_batch(
            tok_buggy["input_ids"],
            tok_fixed["input_ids"],
            tok_buggy["attention_mask"],
            tok_fixed["attention_mask"],
            ignore_token_ids=None,
        )
        return pool_change_regions(
            z_pred,
            z_tgt,
            regions,
            pred_empty_embedding=empty_region_emb,
            tgt_empty_embedding=empty_region_emb,
        )
    if supervision_target == "change_plus_shared":
        shared_weight = float(cfg.get("loss", {}).get("shared_weight", 0.1))
        regions = find_change_regions_in_batch(
            tok_buggy["input_ids"],
            tok_fixed["input_ids"],
            tok_buggy["attention_mask"],
            tok_fixed["attention_mask"],
            ignore_token_ids=None,
        )
        pred_weights = build_change_plus_shared_weights(
            tok_buggy["attention_mask"],
            regions,
            side="buggy",
            shared_weight=shared_weight,
        )
        tgt_weights = build_change_plus_shared_weights(
            tok_fixed["attention_mask"],
            regions,
            side="fixed",
            shared_weight=shared_weight,
        )
        pred_mix = weighted_mean_pool(z_pred, pred_weights)
        tgt_mix = weighted_mean_pool(z_tgt, tgt_weights)
        return pred_mix, tgt_mix
    if supervision_target == "full_sequence":
        pred_full = masked_full_mean_pool(z_pred, tok_buggy["attention_mask"])
        tgt_full = masked_full_mean_pool(z_tgt, tok_fixed["attention_mask"])
        return pred_full, tgt_full
    raise ValueError(f"Unknown loss.supervision_target='{supervision_target}'")


def wandb_init_if_needed(cfg: Dict[str, Any], out_dir: str) -> None:
    wcfg = cfg.get("wandb", {})
    if not bool(wcfg.get("enabled", False)):
        return
    if wandb is None:
        raise RuntimeError("wandb is enabled but not installed.")
    if not is_main():
        return

    wandb.init(
        entity=wcfg.get("entity") or None,
        project=wcfg.get("project") or None,
        group=wcfg.get("group") or None,
        name=wcfg.get("run_name") or None,
        id=wcfg.get("id") or None,
        resume=wcfg.get("resume", "auto"),
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


def build_dataloaders(cfg: Dict[str, Any], tokenizer, use_ddp: bool):
    assert cfg["data"]["source"] == "hf", "This train script expects data.source=hf."
    hf_cfg = cfg["data"]["hf"]
    idx_cfg = cfg["data"]["indices"]

    dataset_id = hf_cfg["dataset_id"]
    split_name = hf_cfg.get("split", "train")

    global_target_dir = idx_cfg["global_target_dir"]
    split_dir = idx_cfg["split_dir"]
    global_target_idx = load_indices(global_target_dir, idx_cfg["global_target"])
    train_idx = load_indices(split_dir, idx_cfg["train"])
    val_idx = load_indices(split_dir, idx_cfg["val"])

    ds_full = load_dataset(dataset_id, split=split_name)
    ds_subset = ds_full.select(global_target_idx.tolist())
    ds_train = ds_subset.select(train_idx.tolist())
    ds_val = ds_subset.select(val_idx.tolist())

    buggy_key = hf_cfg["fields"]["buggy"]
    fixed_key = hf_cfg["fields"]["fixed"]
    max_len = int(cfg["encoder"]["max_len"])

    def collate_fn(batch):
        buggy = [str(x.get(buggy_key, "") or "") for x in batch]
        fixed = [str(x.get(fixed_key, "") or "") for x in batch]
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
        return tok_buggy, tok_fixed

    train_sampler = DistributedSampler(ds_train, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(ds_val, shuffle=False) if use_ddp else None

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["data"].get("num_workers", 4))

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return dl_train, dl_val, train_sampler, val_sampler


def run_validation(
    cfg: Dict[str, Any],
    enc_ctx: torch.nn.Module,
    enc_tgt: torch.nn.Module,
    predictor: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dl_val: DataLoader,
    val_sampler: Optional[DistributedSampler],
    device: torch.device,
    epoch: int,
    global_step: int,
    metrics_logger: Optional[JSONLLogger],
    empty_region_emb: Optional[torch.Tensor],
) -> Dict[str, float]:
    enc_ctx.eval()
    predictor.eval()
    enc_tgt.eval()
    if ddp_enabled() and val_sampler is not None:
        val_sampler.set_epoch(epoch)

    meter_loss = AverageMeter()
    meter_align = AverageMeter()
    meter_var = AverageMeter()
    meter_cov = AverageMeter()
    meter_top1 = AverageMeter()
    meter_std_pred = AverageMeter()
    meter_std_tgt = AverageMeter()

    with torch.no_grad():
        for tok_buggy, tok_fixed in dl_val:
            tok_buggy = to_device(tok_buggy, device)
            tok_fixed = to_device(tok_fixed, device)

            z_ctx = enc_ctx(tok_buggy["input_ids"], tok_buggy["attention_mask"])
            z_tgt = enc_tgt(tok_fixed["input_ids"], tok_fixed["attention_mask"])
            z_pred = predictor(z_ctx, attention_mask=tok_buggy["attention_mask"])

            pred_region, tgt_region = get_supervision_pair(
                cfg,
                tok_buggy,
                tok_fixed,
                z_pred,
                z_tgt,
                empty_region_emb=empty_region_emb,
            )
            out = loss_fn(pred_region, tgt_region)

            bsz = pred_region.size(0)
            meter_loss.update(out["loss"].item(), bsz)
            meter_align.update(out["align"].item(), bsz)
            meter_var.update(out["var"].item(), bsz)
            meter_cov.update(out["cov"].item(), bsz)
            meter_top1.update(retrieval_top1_acc(pred_region, tgt_region).item(), bsz)
            meter_std_pred.update(emb_std_mean(pred_region).item(), bsz)
            meter_std_tgt.update(emb_std_mean(tgt_region).item(), bsz)

    t = torch.tensor(
        [
            meter_loss.sum, meter_loss.count,
            meter_align.sum, meter_align.count,
            meter_var.sum, meter_var.count,
            meter_cov.sum, meter_cov.count,
            meter_top1.sum, meter_top1.count,
            meter_std_pred.sum, meter_std_pred.count,
            meter_std_tgt.sum, meter_std_tgt.count,
        ],
        dtype=torch.float64,
        device=device,
    )
    ddp_all_reduce_sum(t)
    values = t.tolist()

    def avg(s: float, c: float) -> float:
        return s / max(1.0, c)

    metrics = {
        "val_loss": avg(values[0], values[1]),
        "val_align": avg(values[2], values[3]),
        "val_var": avg(values[4], values[5]),
        "val_cov": avg(values[6], values[7]),
        "val_top1": avg(values[8], values[9]),
        "val_std_pred": avg(values[10], values[11]),
        "val_std_tgt": avg(values[12], values[13]),
    }

    if is_main():
        print(
            f"[ep {epoch} step {global_step}] VAL "
            f"loss={metrics['val_loss']:.4f} align={metrics['val_align']:.4f} "
            f"var={metrics['val_var']:.4f} cov={metrics['val_cov']:.4f} "
            f"top1={metrics['val_top1']:.3f}"
        )
        if metrics_logger is not None:
            metrics_logger.log({"split": "val", "step": global_step, "epoch": epoch, **metrics, "time": time.time()})
        wandb_log_if_needed(
            cfg,
            {
                "val/loss": metrics["val_loss"],
                "val/align": metrics["val_align"],
                "val/var": metrics["val_var"],
                "val/cov": metrics["val_cov"],
                "val/top1": metrics["val_top1"],
                "val/std_pred": metrics["val_std_pred"],
                "val/std_tgt": metrics["val_std_tgt"],
                "epoch": epoch,
            },
            step=global_step,
        )
    return metrics


def train(cfg: Dict[str, Any]) -> None:
    use_ddp = bool(cfg.get("ddp", {}).get("enabled", False)) and ddp_enabled()
    if use_ddp:
        ddp_setup(cfg.get("ddp", {}).get("backend", "nccl"))

    device = torch.device("cuda", ddp_local_rank()) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(int(cfg.get("seed", 42)) + rank())

    run_name = cfg.get("run", {}).get("run_name", "run")
    save_root = cfg.get("run", {}).get("save_dir", "./checkpoints")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(save_root, f"{run_name}_{ts}" + (f"_job{job_id}" if job_id else ""))
    ckpt_dir = os.path.join(out_dir, "checkpoints")

    metrics_logger = None
    if is_main():
        os.makedirs(ckpt_dir, exist_ok=True)
        save_resolved_config(cfg, out_dir)
        metrics_logger = JSONLLogger(os.path.join(out_dir, "metrics.jsonl"))

    wandb_init_if_needed(cfg, out_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"]["name"], use_fast=True)
    dl_train, dl_val, train_sampler, val_sampler = build_dataloaders(cfg, tokenizer, use_ddp)

    enc_ctx, hidden_dim = build_encoder(cfg, device=device)
    enc_tgt, _ = build_encoder(cfg, device=device)
    enc_tgt.load_state_dict(unwrap_ddp(enc_ctx).state_dict(), strict=True)
    for p in enc_tgt.parameters():
        p.requires_grad = False
    enc_tgt.eval()

    predictor = build_predictor(cfg, hidden_dim=hidden_dim, device=device)
    empty_region_emb = torch.nn.Parameter(torch.zeros(hidden_dim, device=device))

    enc_ctx_trainable = sum(p.numel() for p in enc_ctx.parameters() if p.requires_grad)
    predictor_trainable = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    if is_main():
        print(
            f"[Params] encoder.train_mode={cfg['encoder'].get('train_mode', 'frozen')} "
            f"enc_ctx_trainable={enc_ctx_trainable} predictor_trainable={predictor_trainable}"
        )
    if enc_ctx_trainable == 0:
        raise RuntimeError(
            "Context encoder has no trainable parameters. "
            "This usually means encoder.train_mode stayed 'frozen' or LoRA overrides were not applied."
        )
    if predictor_trainable == 0:
        raise RuntimeError("Predictor has no trainable parameters.")

    if use_ddp:
        find_unused = bool(cfg.get("ddp", {}).get("find_unused_parameters", False))
        enc_ctx = DDP(enc_ctx, device_ids=[ddp_local_rank()], find_unused_parameters=find_unused)
        predictor = DDP(predictor, device_ids=[ddp_local_rank()], find_unused_parameters=find_unused)

    loss_fn = build_loss(cfg).to(device)

    lr_encoder = float(cfg["train"].get("lr_encoder", cfg["train"].get("lr", 2e-5)))
    lr_predictor = float(cfg["train"].get("lr_predictor", cfg["train"].get("lr", 1e-4)))
    wd = float(cfg["train"].get("weight_decay", 0.01))
    betas = tuple(cfg.get("optim", {}).get("betas", [0.9, 0.999]))
    eps = float(cfg.get("optim", {}).get("eps", 1e-8))

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in unwrap_ddp(enc_ctx).parameters() if p.requires_grad], "lr": lr_encoder},
            {"params": [p for p in unwrap_ddp(predictor).parameters() if p.requires_grad], "lr": lr_predictor},
            {"params": [empty_region_emb], "lr": lr_predictor},
        ],
        weight_decay=wd,
        betas=betas,
        eps=eps,
    )

    use_fp16 = bool(cfg["train"].get("fp16", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    global_step = 0
    resume_epoch = 0
    resume_it = 0
    best_val = float("inf")

    resume_path = str(cfg["train"].get("resume_from", "") or "")
    if resume_path:
        if is_main():
            print(f"[Resume] Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "enc_ctx" in ckpt:
            unwrap_ddp(enc_ctx).load_state_dict(ckpt["enc_ctx"], strict=True)
        elif "enc_ctx_tunable" in ckpt:
            load_encoder_tunable_state(unwrap_ddp(enc_ctx), ckpt["enc_ctx_tunable"], strict=False)

        if "enc_tgt" in ckpt:
            enc_tgt.load_state_dict(ckpt["enc_tgt"], strict=True)
        elif "enc_tgt_tunable" in ckpt:
            load_encoder_tunable_state(unwrap_ddp(enc_tgt), ckpt["enc_tgt_tunable"], strict=False)

        unwrap_ddp(predictor).load_state_dict(ckpt["predictor"], strict=True)
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("empty_region_emb") is not None:
            empty_region_emb.data.copy_(ckpt["empty_region_emb"].to(device=device, dtype=empty_region_emb.dtype))
        global_step = int(ckpt.get("step", 0))
        resume_epoch = int(ckpt.get("epoch", 0))
        resume_it = int(ckpt.get("it", 0))
        best_val = float(ckpt.get("best_val_loss", best_val))
        if bool(cfg["train"].get("resume_restore_rng", False)) and ckpt.get("rng_state") is not None:
            set_rng_state(ckpt["rng_state"])

    epochs = int(cfg["train"]["epochs"])
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    log_every = int(cfg["train"].get("log_every", 50))
    eval_every_steps = int(cfg["train"].get("eval_every_steps", 0))
    save_every_steps = int(cfg["train"].get("save_every_steps", 0))
    save_every_epoch = bool(cfg["train"].get("save_every_epoch", False))
    tau = float(cfg.get("ema", {}).get("tau", 0.996))
    best_path = os.path.join(ckpt_dir, "ckpt_best.pt")
    last_path = os.path.join(ckpt_dir, "ckpt_last.pt")

    steps_per_epoch = max(1, len(dl_train) // max(1, grad_accum))
    start_epoch = global_step // steps_per_epoch
    start_step_in_epoch = global_step % steps_per_epoch

    if is_main():
        print(f"[Info] Output dir: {out_dir}")
        print(f"[Info] steps_per_epoch={steps_per_epoch}, start_epoch={start_epoch}, start_step_in_epoch={start_step_in_epoch}")

    for epoch in range(start_epoch, epochs):
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        enc_ctx.train()
        predictor.train()
        optimizer.zero_grad(set_to_none=True)

        skip_batches = start_step_in_epoch * grad_accum if epoch == start_epoch else 0

        meter_loss = AverageMeter()
        meter_align = AverageMeter()
        meter_var = AverageMeter()
        meter_cov = AverageMeter()
        meter_top1 = AverageMeter()

        iterator = tqdm(dl_train, desc=f"Epoch {epoch}", dynamic_ncols=True) if is_main() else dl_train
        for it, (tok_buggy, tok_fixed) in enumerate(iterator):
            if skip_batches > 0 and it < skip_batches:
                continue

            tok_buggy = to_device(tok_buggy, device)
            tok_fixed = to_device(tok_fixed, device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                z_ctx = enc_ctx(tok_buggy["input_ids"], tok_buggy["attention_mask"])
                with torch.no_grad():
                    z_tgt = enc_tgt(tok_fixed["input_ids"], tok_fixed["attention_mask"])
                z_pred = predictor(z_ctx, attention_mask=tok_buggy["attention_mask"])

                pred_region, tgt_region = get_supervision_pair(
                    cfg,
                    tok_buggy,
                    tok_fixed,
                    z_pred,
                    z_tgt,
                    empty_region_emb=empty_region_emb,
                )
                out = loss_fn(pred_region, tgt_region)
                loss = out["loss"] / max(1, grad_accum)

            scaler.scale(loss).backward()

            bsz = pred_region.size(0)
            meter_loss.update(out["loss"].item(), bsz)
            meter_align.update(out["align"].item(), bsz)
            meter_var.update(out["var"].item(), bsz)
            meter_cov.update(out["cov"].item(), bsz)
            meter_top1.update(retrieval_top1_acc(pred_region, tgt_region).item(), bsz)

            if (it + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                enc_ctx_grad_norm = grad_global_norm(unwrap_ddp(enc_ctx))
                predictor_grad_norm = grad_global_norm(unwrap_ddp(predictor))
                grad_norm_ratio = predictor_grad_norm / max(enc_ctx_grad_norm, 1e-12)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema_update(enc_tgt, enc_ctx, tau=tau)
                global_step += 1

                if is_main() and (global_step % log_every == 0):
                    train_metrics = {
                        "train/loss": meter_loss.avg,
                        "train/align": meter_align.avg,
                        "train/var": meter_var.avg,
                        "train/cov": meter_cov.avg,
                        "train/top1": meter_top1.avg,
                        "train/enc_ctx_grad_norm": enc_ctx_grad_norm,
                        "train/predictor_grad_norm": predictor_grad_norm,
                        "train/predictor_to_encoder_grad_ratio": grad_norm_ratio,
                        "epoch": epoch,
                    }
                    print(
                        f"[ep {epoch} step {global_step}] TRAIN "
                        f"loss={meter_loss.avg:.4f} align={meter_align.avg:.4f} "
                        f"var={meter_var.avg:.4f} cov={meter_cov.avg:.4f} "
                        f"top1={meter_top1.avg:.3f} "
                        f"enc_gn={enc_ctx_grad_norm:.4f} pred_gn={predictor_grad_norm:.4f} "
                        f"pred/enc={grad_norm_ratio:.3f}"
                    )
                    wandb_log_if_needed(cfg, train_metrics, step=global_step)
                    if metrics_logger is not None:
                        metrics_logger.log(
                            {
                                "split": "train",
                                "step": global_step,
                                "epoch": epoch,
                                "loss": meter_loss.avg,
                                "align": meter_align.avg,
                                "var": meter_var.avg,
                                "cov": meter_cov.avg,
                                "top1": meter_top1.avg,
                                "enc_ctx_grad_norm": enc_ctx_grad_norm,
                                "predictor_grad_norm": predictor_grad_norm,
                                "predictor_to_encoder_grad_ratio": grad_norm_ratio,
                                "time": time.time(),
                            }
                        )

                if save_every_steps > 0 and (global_step % save_every_steps == 0) and is_main():
                    step_path = os.path.join(ckpt_dir, f"ckpt_step{global_step}.pt")
                    save_checkpoint(step_path, enc_ctx, enc_tgt, predictor, optimizer, scaler, empty_region_emb, global_step, epoch, it, cfg)

                if eval_every_steps > 0 and (global_step % eval_every_steps == 0):
                    val_metrics = run_validation(
                        cfg,
                        enc_ctx,
                        enc_tgt,
                        predictor,
                        loss_fn,
                        dl_val,
                        val_sampler,
                        device,
                        epoch,
                        global_step,
                        metrics_logger,
                        empty_region_emb,
                    )
                    if is_main() and val_metrics["val_loss"] < best_val:
                        best_val = val_metrics["val_loss"]
                        save_inference_checkpoint(best_path, enc_ctx, enc_tgt, predictor, empty_region_emb, global_step, epoch, cfg)
                        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                        ckpt["best_val_loss"] = best_val
                        torch.save(ckpt, best_path)
                        print(f"[Best] Saved best checkpoint to {best_path}")
                    enc_ctx.train()
                    predictor.train()

        val_metrics = run_validation(
            cfg,
            enc_ctx,
            enc_tgt,
            predictor,
            loss_fn,
            dl_val,
            val_sampler,
            device,
            epoch,
            global_step,
            metrics_logger,
            empty_region_emb,
        )
        if is_main() and val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            save_inference_checkpoint(best_path, enc_ctx, enc_tgt, predictor, empty_region_emb, global_step, epoch, cfg)
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            ckpt["best_val_loss"] = best_val
            torch.save(ckpt, best_path)
            print(f"[Best] Saved best checkpoint to {best_path}")

        if is_main() and save_every_epoch:
            epoch_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch + 1}.pt")
            save_checkpoint(epoch_path, enc_ctx, enc_tgt, predictor, optimizer, scaler, empty_region_emb, global_step, epoch, len(dl_train), cfg)
            ckpt = torch.load(epoch_path, map_location="cpu", weights_only=False)
            ckpt["best_val_loss"] = best_val
            torch.save(ckpt, epoch_path)

    if is_main():
        save_checkpoint(last_path, enc_ctx, enc_tgt, predictor, optimizer, scaler, empty_region_emb, global_step, epochs - 1, len(dl_train), cfg)
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        ckpt["best_val_loss"] = best_val
        torch.save(ckpt, last_path)

    wandb_finish_if_needed(cfg)
    if use_ddp:
        ddp_cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--set", nargs="*", action="append", default=[])
    args = parser.parse_args()

    cfg = resolve_config(args.config, flatten_overrides(args.set))
    train(cfg)


if __name__ == "__main__":
    main()
