# utils.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import random
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import yaml
from torch.profiler import profile, ProfilerActivity


# -------------------------
# Config
# -------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def _cast_value(v: str):
    s = v.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("none", "null"):
        return None
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_cast_value(x.strip()) for x in inner.split(",")]
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s

def set_by_dotted_key(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Bad override '{ov}', expected key=value")
        k, v = ov.split("=", 1)
        set_by_dotted_key(cfg, k.strip(), _cast_value(v.strip()))
    return cfg

def save_resolved_config(cfg: Dict[str, Any], out_dir: str, filename: str = "resolved_config.json") -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# -------------------------
# File / JSON helpers
# -------------------------
def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pt(obj: Any, path: str) -> None:
    ensure_parent_dir(path)
    torch.save(obj, path)


def load_pt(path: str) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class JSONLLogger:
    def __init__(self, path: str):
        self.path = path
        ensure_parent_dir(path)

    def log(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -------------------------
# DDP helpers
# -------------------------
def is_ddp_env() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def ddp_enabled() -> bool:
    return is_ddp_env()

def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def ddp_rank() -> int:
    return rank()

def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def ddp_local_rank() -> int:
    return local_rank()

def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def ddp_world() -> int:
    return world_size()

def is_main() -> bool:
    return rank() == 0

def ddp_setup(backend: str = "nccl") -> None:
    if not ddp_enabled():
        return
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())

def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def ddp_barrier() -> None:
    if dist.is_initialized():
        dist.barrier()

def ddp_all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_tf32(allow_tf32: bool) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)


# -------------------------
# EMA / model helpers
# -------------------------
@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, tau: float) -> None:
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(tau).add_(p_s.data, alpha=1.0 - tau)

def unwrap_ddp(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m

def set_requires_grad(m: nn.Module, requires_grad: bool) -> None:
    for p in m.parameters():
        p.requires_grad = requires_grad


def extract_named_trainable_parameters(module: nn.Module) -> Dict[str, torch.Tensor]:
    base = unwrap_ddp(module)
    return {
        name: p.detach().to("cpu")
        for name, p in base.named_parameters()
        if p.requires_grad
    }


def load_named_parameters(module: nn.Module, saved: Dict[str, torch.Tensor], strict: bool = False) -> List[str]:
    base = unwrap_ddp(module)
    current = dict(base.named_parameters())
    missing: List[str] = []
    for name, tensor in saved.items():
        if name in current:
            current[name].data.copy_(tensor)
        elif strict:
            missing.append(name)
    return missing


def extract_from_dict(data: Any, prefer_keys: List[str]) -> Any:
    if isinstance(data, dict):
        for key in prefer_keys:
            if key in data:
                return data[key]
        for value in data.values():
            if isinstance(value, (list, np.ndarray)):
                return value
        raise KeyError(f"Cannot find array in dict keys={list(data.keys())[:50]}")
    return data


# -------------------------
# Simple meters
# -------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


# -------------------------
# FLOPs / Profiler helpers
# -------------------------
def move_batch_to_device(
    batch,
    device: torch.device,
):
    """
    Move a batch of (b_emb, b_ids, b_mask, gidx/anything) to device.
    Keeps the 4th item untouched.
    """
    b_emb, b_ids, b_mask, b_extra = batch
    return (
        b_emb.to(device, non_blocking=True),
        b_ids.to(device, non_blocking=True),
        b_mask.to(device, non_blocking=True),
        b_extra,
    )


def get_total_flops_from_prof(prof) -> float:
    """
    Sum FLOPs over all profiler events.
    Note:
      PyTorch profiler FLOPs are formula-based estimates and do not cover every op.
      Still useful for relative comparison across runs.
    """
    total_flops = 0.0
    for evt in prof.key_averages():
        flops = getattr(evt, "flops", None)
        if flops is not None:
            total_flops += float(flops)
    return total_flops


# -------------------------
# Embedding helpers
# -------------------------
def load_embeddings_pt(path: str, key: str = "z_pred", index_key: str = "global_indices") -> Tuple[torch.Tensor, torch.Tensor]:
    obj = load_pt(path)
    if not isinstance(obj, dict):
        raise ValueError(f"emb_pt must be a dict, got {type(obj)}")
    if key not in obj:
        raise KeyError(f"Missing key '{key}' in {path}. keys={list(obj.keys())[:20]}")
    if index_key not in obj:
        raise KeyError(f"Missing key '{index_key}' in {path}. keys={list(obj.keys())[:20]}")

    emb = obj[key]
    gidx = obj[index_key]
    if not torch.is_tensor(emb) or not torch.is_tensor(gidx):
        raise ValueError("emb and indices must be torch tensors.")
    if emb.dim() != 2:
        raise ValueError(f"emb must be [N,D], got {tuple(emb.shape)}")
    if gidx.dim() != 1 or gidx.numel() != emb.shape[0]:
        raise ValueError(f"indices must be [N], got {tuple(gidx.shape)} vs N={emb.shape[0]}")
    return emb.contiguous(), gidx.long().contiguous()
