from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .io import LoadedSplitEmbeddings


JEPA_REGION_SRC = Path(__file__).resolve().parents[3] / "JEPA_region" / "src"
if str(JEPA_REGION_SRC) not in sys.path:
    sys.path.insert(0, str(JEPA_REGION_SRC))

from models import build_encoder, load_encoder_tunable_state  # type: ignore  # noqa: E402
from utils import load_yaml  # type: ignore  # noqa: E402


class IndexedCodeDataset(Dataset):
    def __init__(self, dataset: Any, global_indices: np.ndarray, buggy_field: str, fixed_field: str) -> None:
        self.dataset = dataset
        self.global_indices = global_indices.astype(np.int64)
        self.buggy_field = str(buggy_field)
        self.fixed_field = str(fixed_field)

    def __len__(self) -> int:
        return len(self.global_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[int(self.global_indices[idx])]
        return {
            "buggy": str(sample.get(self.buggy_field, "") or ""),
            "fixed": str(sample.get(self.fixed_field, "") or ""),
            "global_index": int(self.global_indices[idx]),
        }


class TokenizingCollator:
    def __init__(self, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = int(max_len)

    def __call__(self, batch):
        buggy = [item["buggy"] for item in batch]
        fixed = [item["fixed"] for item in batch]
        global_indices = torch.tensor([item["global_index"] for item in batch], dtype=torch.long)
        tok_buggy = self.tokenizer(
            buggy,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        tok_fixed = self.tokenizer(
            fixed,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return tok_buggy, tok_fixed, global_indices


def load_jepa_runtime_cfg(ckpt: Dict[str, Any], config_path: str = "") -> Dict[str, Any]:
    if config_path:
        return load_yaml(config_path)
    cfg = ckpt.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("JEPA checkpoint is missing embedded cfg. Please provide --jepa-config.")
    return cfg


def load_indices(indices_dir: str, filename: str) -> np.ndarray:
    return np.load(str(Path(indices_dir) / filename))


def ddp_enabled() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def ddp_rank() -> int:
    return torch.distributed.get_rank() if ddp_enabled() else 0


def ddp_world_size() -> int:
    return torch.distributed.get_world_size() if ddp_enabled() else 1


def ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def ddp_setup(backend: str = "nccl") -> None:
    if ddp_enabled():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if torch.cuda.is_available():
            torch.cuda.set_device(ddp_local_rank())
        torch.distributed.init_process_group(backend=backend)


def ddp_cleanup() -> None:
    if ddp_enabled():
        torch.distributed.destroy_process_group()


def is_main_process() -> bool:
    return ddp_rank() == 0


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _load_encoder_from_ckpt(
    cfg: Dict[str, Any],
    ckpt: Dict[str, Any],
    role: str,
    device: torch.device,
):
    encoder, _ = build_encoder(cfg, device=device)
    if role == "ctx":
        if "enc_ctx" in ckpt:
            encoder.load_state_dict(ckpt["enc_ctx"], strict=True)
        elif "enc_ctx_tunable" in ckpt:
            load_encoder_tunable_state(encoder, ckpt["enc_ctx_tunable"], strict=False)
        else:
            raise KeyError("Checkpoint is missing context encoder weights.")
    elif role == "tgt":
        if "enc_tgt" in ckpt:
            encoder.load_state_dict(ckpt["enc_tgt"], strict=True)
        elif "enc_tgt_tunable" in ckpt:
            load_encoder_tunable_state(encoder, ckpt["enc_tgt_tunable"], strict=False)
        elif "enc_ctx" in ckpt:
            encoder.load_state_dict(ckpt["enc_ctx"], strict=True)
        elif "enc_ctx_tunable" in ckpt:
            load_encoder_tunable_state(encoder, ckpt["enc_ctx_tunable"], strict=False)
        else:
            raise KeyError("Checkpoint is missing target encoder weights.")
    else:
        raise ValueError(f"Unknown encoder role: {role}")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


@torch.no_grad()
def load_online_split_embeddings(
    checkpoint_path: str,
    config_path: str,
    encoder_mode: str,
    indices_dir: str,
    indices_file: str,
    global_target_dir: str,
    global_target_file: str = "global_target_indices.npy",
    hf_dataset_name: str = "",
    hf_split: str = "",
    hf_buggy_field: str = "",
    hf_fixed_field: str = "",
    max_len: int = 512,
    batch_size: int = 128,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
) -> LoadedSplitEmbeddings:
    ddp_setup("nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        device = device or torch.device("cuda", ddp_local_rank())
    else:
        device = device or torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = load_jepa_runtime_cfg(ckpt, config_path=config_path)

    dataset_name = hf_dataset_name or cfg["data"]["hf"]["dataset_id"]
    dataset_split = hf_split or cfg["data"]["hf"].get("split", "train")
    buggy_field = hf_buggy_field or cfg["data"]["hf"]["fields"]["buggy"]
    fixed_field = hf_fixed_field or cfg["data"]["hf"]["fields"]["fixed"]
    max_length = int(max_len or cfg["encoder"]["max_len"])

    split_idx = load_indices(indices_dir, indices_file)
    global_target_idx = load_indices(global_target_dir, global_target_file)
    global_indices = global_target_idx[split_idx]

    world = ddp_world_size()
    rank = ddp_rank()
    sharded_global_indices = global_indices[rank::world]

    dataset = load_dataset(dataset_name, split=dataset_split)
    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"]["name"], use_fast=True)
    ds = IndexedCodeDataset(dataset, sharded_global_indices, buggy_field, fixed_field)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=TokenizingCollator(tokenizer, max_length),
        persistent_workers=bool(num_workers > 0),
    )

    enc_ctx = _load_encoder_from_ckpt(cfg, ckpt, role="ctx", device=device)
    enc_tgt = _load_encoder_from_ckpt(cfg, ckpt, role="tgt", device=device)

    mode = str(encoder_mode).lower()
    if mode not in {"same_encoder", "context_target", "target_target"}:
        raise ValueError(f"Unsupported encoder_mode: {encoder_mode}")

    buggy_emb_all = []
    fixed_emb_all = []
    gidx_all = []

    iterator = tqdm(dl, desc=f"[Extract rank{rank}]", leave=False) if is_main_process() else dl
    for tok_buggy, tok_fixed, gidx in iterator:
        tok_buggy = to_device(tok_buggy, device)
        tok_fixed = to_device(tok_fixed, device)

        if mode == "same_encoder":
            buggy_seq = enc_ctx(tok_buggy["input_ids"], tok_buggy["attention_mask"])
            fixed_seq = enc_ctx(tok_fixed["input_ids"], tok_fixed["attention_mask"])
        elif mode == "context_target":
            buggy_seq = enc_ctx(tok_buggy["input_ids"], tok_buggy["attention_mask"])
            fixed_seq = enc_tgt(tok_fixed["input_ids"], tok_fixed["attention_mask"])
        else:
            buggy_seq = enc_tgt(tok_buggy["input_ids"], tok_buggy["attention_mask"])
            fixed_seq = enc_tgt(tok_fixed["input_ids"], tok_fixed["attention_mask"])

        buggy_emb_all.append(masked_mean_pool(buggy_seq, tok_buggy["attention_mask"]).cpu().float())
        fixed_emb_all.append(masked_mean_pool(fixed_seq, tok_fixed["attention_mask"]).cpu().float())
        gidx_all.append(gidx.cpu().long())

    local_buggy = torch.cat(buggy_emb_all, dim=0).numpy() if buggy_emb_all else np.zeros((0, 1), dtype=np.float32)
    local_fixed = torch.cat(fixed_emb_all, dim=0).numpy() if fixed_emb_all else np.zeros((0, 1), dtype=np.float32)
    local_gidx = torch.cat(gidx_all, dim=0).numpy().astype(np.int64) if gidx_all else np.zeros((0,), dtype=np.int64)

    if ddp_enabled():
        gathered: list[Optional[dict[str, np.ndarray]]] = [None for _ in range(world)]
        torch.distributed.all_gather_object(
            gathered,
            {"buggy": local_buggy, "fixed": local_fixed, "gidx": local_gidx},
        )
        buggy_parts = [item["buggy"] for item in gathered if item is not None]
        fixed_parts = [item["fixed"] for item in gathered if item is not None]
        gidx_parts = [item["gidx"] for item in gathered if item is not None]
        full_buggy = np.concatenate(buggy_parts, axis=0) if buggy_parts else np.zeros((0, 1), dtype=np.float32)
        full_fixed = np.concatenate(fixed_parts, axis=0) if fixed_parts else np.zeros((0, 1), dtype=np.float32)
        full_gidx = np.concatenate(gidx_parts, axis=0).astype(np.int64) if gidx_parts else np.zeros((0,), dtype=np.int64)
        order = np.argsort(full_gidx)
        full_buggy = full_buggy[order]
        full_fixed = full_fixed[order]
        full_gidx = full_gidx[order]
    else:
        full_buggy = local_buggy
        full_fixed = local_fixed
        full_gidx = local_gidx

    return LoadedSplitEmbeddings(
        buggy_emb=full_buggy,
        pred_emb=np.zeros((len(full_gidx), 1), dtype=np.float32),
        tgt_emb=full_fixed,
        global_indices=full_gidx,
    )
