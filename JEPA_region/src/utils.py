from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _cast_value(value: str) -> Any:
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"none", "null"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_cast_value(x.strip()) for x in inner.split(",")]
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def set_by_dotted_key(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Bad override '{item}', expected key=value")
        key, value = item.split("=", 1)
        set_by_dotted_key(cfg, key.strip(), _cast_value(value.strip()))
    return cfg


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_resolved_config(cfg: Dict[str, Any], out_dir: str, filename: str = "resolved_config.json") -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_json(cfg, os.path.join(out_dir, filename))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def ddp_all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, tau: float) -> None:
    teacher = unwrap_ddp(teacher)
    student = unwrap_ddp(student)
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(tau).add_(student_param.data, alpha=1.0 - tau)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


class JSONLLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def valid_token_ids(
    input_ids: Sequence[int],
    attention_mask: Optional[Sequence[int]] = None,
    ignore_token_ids: Optional[set[int]] = None,
) -> List[int]:
    ignore_token_ids = ignore_token_ids or set()
    output: List[int] = []
    for idx, token_id in enumerate(input_ids):
        if attention_mask is not None and int(attention_mask[idx]) == 0:
            break
        if int(token_id) in ignore_token_ids:
            continue
        output.append(int(token_id))
    return output


@dataclass
class ChangeRegion:
    left_len: int
    buggy_start: int
    buggy_end: int
    fixed_start: int
    fixed_end: int
    right_len: int

    @property
    def buggy_len(self) -> int:
        return self.buggy_end - self.buggy_start

    @property
    def fixed_len(self) -> int:
        return self.fixed_end - self.fixed_start


def find_change_region(
    buggy_token_ids: Sequence[int],
    fixed_token_ids: Sequence[int],
) -> ChangeRegion:
    buggy = list(buggy_token_ids)
    fixed = list(fixed_token_ids)

    left = 0
    min_len = min(len(buggy), len(fixed))
    while left < min_len and buggy[left] == fixed[left]:
        left += 1

    buggy_right = len(buggy)
    fixed_right = len(fixed)
    while buggy_right > left and fixed_right > left and buggy[buggy_right - 1] == fixed[fixed_right - 1]:
        buggy_right -= 1
        fixed_right -= 1

    right_len = len(buggy) - buggy_right
    return ChangeRegion(
        left_len=left,
        buggy_start=left,
        buggy_end=buggy_right,
        fixed_start=left,
        fixed_end=fixed_right,
        right_len=right_len,
    )


def find_change_region_from_inputs(
    buggy_input_ids: Sequence[int],
    fixed_input_ids: Sequence[int],
    buggy_attention_mask: Optional[Sequence[int]] = None,
    fixed_attention_mask: Optional[Sequence[int]] = None,
    ignore_token_ids: Optional[set[int]] = None,
) -> ChangeRegion:
    buggy = valid_token_ids(
        buggy_input_ids,
        attention_mask=buggy_attention_mask,
        ignore_token_ids=ignore_token_ids,
    )
    fixed = valid_token_ids(
        fixed_input_ids,
        attention_mask=fixed_attention_mask,
        ignore_token_ids=ignore_token_ids,
    )
    return find_change_region(buggy, fixed)


def find_change_regions_in_batch(
    buggy_input_ids: torch.Tensor,
    fixed_input_ids: torch.Tensor,
    buggy_attention_mask: Optional[torch.Tensor] = None,
    fixed_attention_mask: Optional[torch.Tensor] = None,
    ignore_token_ids: Optional[set[int]] = None,
) -> List[ChangeRegion]:
    regions: List[ChangeRegion] = []
    batch_size = buggy_input_ids.size(0)
    for i in range(batch_size):
        region = find_change_region_from_inputs(
            buggy_input_ids[i].tolist(),
            fixed_input_ids[i].tolist(),
            buggy_attention_mask[i].tolist() if buggy_attention_mask is not None else None,
            fixed_attention_mask[i].tolist() if fixed_attention_mask is not None else None,
            ignore_token_ids=ignore_token_ids,
        )
        regions.append(region)
    return regions


def span_mask(length: int, start: int, end: int, device: Optional[torch.device] = None) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    if end > start:
        mask[start:end] = True
    return mask


def batch_span_mask(
    seq_len: int,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    positions = torch.arange(seq_len, device=starts.device).unsqueeze(0)
    return (positions >= starts.unsqueeze(1)) & (positions < ends.unsqueeze(1))


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if hidden.dim() != 3:
        raise ValueError(f"hidden must be [B, L, D], got {tuple(hidden.shape)}")
    if mask.dim() != 2:
        raise ValueError(f"mask must be [B, L], got {tuple(mask.shape)}")
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp(min=eps)
    return summed / counts


def masked_mean_pool_with_empty(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    empty_embedding: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    pooled = masked_mean_pool(hidden, mask, eps=eps)
    if empty_embedding is None:
        return pooled

    empty_rows = ~mask.any(dim=1)
    if not bool(empty_rows.any()):
        return pooled

    empty_vec = empty_embedding.to(device=hidden.device, dtype=hidden.dtype).view(1, -1)
    empty_vec = empty_vec.expand(hidden.size(0), -1)
    empty_mask = empty_rows.to(dtype=hidden.dtype).unsqueeze(-1)
    return pooled * (1.0 - empty_mask) + empty_vec * empty_mask


def pool_regions_by_side(
    hidden: torch.Tensor,
    regions: List[ChangeRegion],
    side: str,
    empty_embedding: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if hidden.dim() != 3:
        raise ValueError("hidden must be [B, L, D]")
    if len(regions) != hidden.size(0):
        raise ValueError("regions length must match batch size")

    seq_len = hidden.size(1)
    masks = []
    for region in regions:
        if side == "buggy":
            masks.append(span_mask(seq_len, region.buggy_start, region.buggy_end, device=hidden.device))
        elif side == "fixed":
            masks.append(span_mask(seq_len, region.fixed_start, region.fixed_end, device=hidden.device))
        else:
            raise ValueError(f"Unknown side='{side}'")
    mask = torch.stack(masks, dim=0)
    return masked_mean_pool_with_empty(hidden, mask, empty_embedding=empty_embedding)


def pool_change_regions(
    pred_hidden: torch.Tensor,
    tgt_hidden: torch.Tensor,
    regions: List[ChangeRegion],
    pred_empty_embedding: Optional[torch.Tensor] = None,
    tgt_empty_embedding: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_hidden.dim() != 3 or tgt_hidden.dim() != 3:
        raise ValueError("pred_hidden and tgt_hidden must be [B, L, D]")
    if len(regions) != pred_hidden.size(0) or len(regions) != tgt_hidden.size(0):
        raise ValueError("regions length must match batch size")

    pred_pooled = pool_regions_by_side(
        pred_hidden,
        regions,
        side="buggy",
        empty_embedding=pred_empty_embedding,
    )
    tgt_pooled = pool_regions_by_side(
        tgt_hidden,
        regions,
        side="fixed",
        empty_embedding=tgt_empty_embedding,
    )
    return pred_pooled, tgt_pooled
