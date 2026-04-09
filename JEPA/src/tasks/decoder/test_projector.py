#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
decoder_only_ddp.py (JSONL format aligned with your ViT pipeline)

Goal:
- Load precomputed embeddings from a .pt file (pred_fixed_emb OR true fixed_emb)
- Decode with StarCoder2-3B + your tunable params (prompt_proj + ln)
- DDP parallel: each rank decodes a shard of embeddings (rows)
- Write JSONL records in the SAME format as your previous ViT pipeline:
    {
      "global_index": ...,
      "problem_id": ...,
      "buggy_submission_id": ...,
      "fixed_submission_id": ...,
      "language": ...,
      "preds": [...],
      "gt_fixed_code": ...
    }
- Save JSONL per-rank, and optionally merge on rank0

Run (single GPU):
  python3 decoder_only_ddp.py --emb_pt pred_fixed_emb_test.pt

Run (multi GPU):
  torchrun --standalone --nproc_per_node=4 decoder_only_ddp.py --emb_pt pred_fixed_emb_test.pt
"""

import os
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from jepa.utils import ddp_barrier, ddp_cleanup, ddp_enabled, ddp_local_rank, ddp_rank, ddp_setup, ddp_world, is_main


# =====================
# Keep your existing constants (DO NOT CHANGE)
# =====================
DECODER_TUNABLE_CKPT = "/mimer/NOBACKUP/groups/naiss2025-5-243/youya/CodeRepair_JEPA/e2_decoder_runs_r1/r1_decA_zpred/checkpoints/ckpt_epoch6.pt"

DECODER_MODEL_ID = "bigcode/starcoder2-3b"
PROMPT_LEN = 128

BATCH_DECODER = 20
MAX_NEW_TOKENS = 512

SAVE_PRED_CODE_DIR = "results/pred_fixed_code_pred"
SAVE_PRED_CODE_JSONL = os.path.join(SAVE_PRED_CODE_DIR, "pred_fixed_code_test.jsonl")

# HF dataset fields (same as your ViT pipeline)
HF_DATASET_ID = "ASSERT-KTH/RunBugRun-Final"
HF_SPLIT = "train"
HF_FIXED_FIELD = "fixed_code"


# =====================
# Decoder wrapper（与你训练逻辑一致）
# =====================
class SoftPromptStarCoderDecoder(nn.Module):
    def __init__(self, cond_dim: int, decoder_model, tokenizer, prompt_len: int = 32):
        super().__init__()
        self.decoder = decoder_model
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.hidden_dim = decoder_model.config.hidden_size

        inter_dim = cond_dim * 4
        self.prompt_proj = nn.Sequential(
            nn.Linear(cond_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(inter_dim, prompt_len * self.hidden_dim)
        )
        self.ln = nn.LayerNorm(self.hidden_dim)

        for p in self.decoder.parameters():
            p.requires_grad = False
        self.decoder.eval()

        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        self.eos_token_id = tokenizer.eos_token_id

    @torch.no_grad()
    def generate_fast(self, cond_emb: torch.Tensor, max_new_tokens: int = 128) -> List[str]:
        self.decoder.eval()
        B = cond_emb.shape[0]

        prompt = self.prompt_proj(cond_emb.to(self.prompt_proj[0].weight.dtype))
        prompt = prompt.view(B, self.prompt_len, self.hidden_dim)
        prompt = self.ln(prompt).to(self.decoder.dtype)

        p_mask = torch.ones(B, self.prompt_len, device=cond_emb.device, dtype=torch.long)

        generated_ids = self.decoder.generate(
            inputs_embeds=prompt,
            attention_mask=p_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def load_tunable_parameters(model, path: str):
    state = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  >>> [Loaded decoder tunable params] <- {path}")
    if len(unexpected) > 0:
        print(f"  [Warn] unexpected keys: {unexpected[:5]} ...")
    if len(missing) > 0:
        print(f"  [Info] missing keys (expected): {missing[:5]} ...")


# =====================
# Embedding loader
# =====================
def load_embeddings_pt(path: str, emb_key: Optional[str] = None) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if torch.is_tensor(obj):
        return obj, None

    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported emb_pt format: {type(obj)}. Expect Tensor or dict.")

    if emb_key:
        if emb_key not in obj:
            raise KeyError(f"--emb_key='{emb_key}' not found in keys={list(obj.keys())[:30]}")
        emb = obj[emb_key]
        if not torch.is_tensor(emb):
            raise ValueError(f"obj['{emb_key}'] is not a tensor.")
        meta = {k: v for k, v in obj.items() if k != emb_key}
        return emb, meta

    candidates = ["pred_fixed_emb_test", "fixed_emb_test", "fixed_emb", "emb", "embeddings", "z_pred", "z_tgt"]
    for k in candidates:
        if k in obj and torch.is_tensor(obj[k]):
            meta = {kk: vv for kk, vv in obj.items() if kk != k}
            return obj[k], meta

    for k, v in obj.items():
        if torch.is_tensor(v):
            meta = {kk: vv for kk, vv in obj.items() if kk != k}
            print(f"[Warn] auto-picked tensor key='{k}' from emb_pt dict.")
            return v, meta

    raise ValueError(f"No tensor found in emb_pt dict keys={list(obj.keys())[:30]}")


def _extract_global_indices(meta: Optional[Dict[str, Any]]) -> Optional[List[int]]:
    if not isinstance(meta, dict):
        return None
    for k in ["global_test_indices", "sample_idx", "global_indices"]:
        if k in meta:
            v = meta[k]
            if torch.is_tensor(v):
                return v.cpu().long().tolist()
            if isinstance(v, (list, tuple)):
                return [int(x) for x in v]
    return None


# =====================
# Dataset for decoding (with HF metadata)
# =====================
class DecodeDataset(Dataset):
    """
    Each item includes:
      emb, global_index, problem_id, buggy_submission_id, fixed_submission_id, language, gt_fixed_code
    """
    def __init__(
        self,
        emb_cpu: torch.Tensor,
        global_indices: List[int],
        problem_ids: List[Any],
        buggy_sids: List[Any],
        fixed_sids: List[Any],
        languages: List[Any],
        gt_fixed_codes: List[str],
    ):
        self.emb = emb_cpu
        self.gidx = global_indices
        self.pid = problem_ids
        self.bsid = buggy_sids
        self.fsid = fixed_sids
        self.lang = languages
        self.gt = gt_fixed_codes

    def __len__(self):
        return int(self.emb.shape[0])

    def __getitem__(self, i: int):
        return (
            self.emb[i],
            int(self.gidx[i]),
            str(self.pid[i]),
            int(self.bsid[i]),
            int(self.fsid[i]),
            str(self.lang[i]),
            str(self.gt[i]),
        )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_pt", type=str, required=True, help="Path to pred_fixed_emb.pt OR fixed_emb.pt")
    ap.add_argument("--emb_key", type=str, default="", help="If emb_pt is a dict, pick this key as embedding tensor")
    ap.add_argument("--out_jsonl", type=str, default="", help="Override output jsonl path (default uses SAVE_PRED_CODE_JSONL)")
    ap.add_argument("--max_items", type=int, default=-1, help="Decode at most N embeddings per rank (debug).")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    # ---- DDP setup ----
    if ddp_enabled():
        ddp_setup("nccl" if torch.cuda.is_available() else "gloo")

    # ---- device ----
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{ddp_local_rank()}")
    else:
        device = torch.device("cpu")

    r = ddp_rank()
    w = ddp_world()
    if ddp_enabled():
        if is_main():
            print(f"[DDP] enabled: world_size={w}")
        print(f"[DDP] rank={r} local_rank={ddp_local_rank()} device={device}")

    # ---- output paths ----
    base_out = args.out_jsonl.strip() or SAVE_PRED_CODE_JSONL
    out_rank = base_out.replace(".jsonl", f".rank{r}.jsonl")
    os.makedirs(os.path.dirname(out_rank), exist_ok=True)

    # ---- load embeddings + global indices ----
    emb_key = args.emb_key.strip() or None
    emb_all, meta = load_embeddings_pt(args.emb_pt, emb_key=emb_key)

    if emb_all.dim() != 2:
        raise ValueError(f"Embedding tensor must be [N, D], got shape={tuple(emb_all.shape)}")

    global_indices_all = _extract_global_indices(meta)
    if global_indices_all is None:
        raise ValueError(
            "Cannot find global indices in emb_pt meta. "
            "Please ensure your embedding pt contains one of: "
            "global_test_indices / sample_idx / global_indices."
        )

    if len(global_indices_all) != emb_all.shape[0]:
        raise ValueError(
            f"Length mismatch: len(global_indices)={len(global_indices_all)} "
            f"but emb rows={emb_all.shape[0]}. They must align 1-to-1."
        )

    # ---- shard by rank (same as before) ----
    emb_shard = emb_all[r::w].contiguous()
    gidx_shard = global_indices_all[r::w]

    if args.max_items > 0:
        emb_shard = emb_shard[: args.max_items].contiguous()
        gidx_shard = gidx_shard[: args.max_items]

    if is_main():
        print(f"[Info] emb_all shape={tuple(emb_all.shape)} from {args.emb_pt}")
    print(f"[Info] rank={r} emb_shard shape={tuple(emb_shard.shape)} gidx_shard={len(gidx_shard)}")

    # ---- load HF dataset metadata for THIS shard only ----
    # This matches your ViT pipeline Step A format fields
    ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT)
    subset = ds.select([int(x) for x in gidx_shard])

    gt_fixed_codes = [str(x) if x is not None else "" for x in subset[HF_FIXED_FIELD]]
    languages = subset["language"]
    problem_ids = subset["problem_id"]
    buggy_sids = subset["buggy_submission_id"]
    fixed_sids = subset["fixed_submission_id"]

    # ---- load decoder tokenizer + model ----
    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Rank {r}] Loading StarCoder decoder (bfloat16)...")
    decoder = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(device)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    cond_dim = int(emb_shard.shape[1])
    model = SoftPromptStarCoderDecoder(cond_dim, decoder, tokenizer, prompt_len=PROMPT_LEN).to(device)

    try:
        model.prompt_proj.to(torch.bfloat16)
        model.ln.to(torch.bfloat16)
    except Exception as e:
        print(f"[Warn] Failed to cast prompt modules to bfloat16: {e}")

    if not os.path.exists(DECODER_TUNABLE_CKPT):
        raise FileNotFoundError(f"Not found: {DECODER_TUNABLE_CKPT}")
    load_tunable_parameters(model, DECODER_TUNABLE_CKPT)

    # ---- DataLoader ----
    ds_decode = DecodeDataset(
        emb_cpu=emb_shard,
        global_indices=gidx_shard,
        problem_ids=problem_ids,
        buggy_sids=buggy_sids,
        fixed_sids=fixed_sids,
        languages=languages,
        gt_fixed_codes=gt_fixed_codes,
    )
    dl = DataLoader(ds_decode, batch_size=BATCH_DECODER, shuffle=False, num_workers=4, pin_memory=True)

    # ---- decode + write JSONL (ViT pipeline format) ----
    with open(out_rank, "w", encoding="utf-8") as f:
        pbar = tqdm(dl, desc=f"Decoding(rank {r})", leave=True)
        for b_emb, b_gidx, b_pid, b_bsid, b_fsid, b_lang, b_gt in pbar:
            b_emb = b_emb.to(device, non_blocking=True)
            preds = model.generate_fast(b_emb, max_new_tokens=MAX_NEW_TOKENS)

            # ensure python types
            if torch.is_tensor(b_gidx): b_gidx = b_gidx.tolist()
            if torch.is_tensor(b_bsid): b_bsid = b_bsid.tolist()
            if torch.is_tensor(b_fsid): b_fsid = b_fsid.tolist()

            # b_pid/b_lang/b_gt are lists (strings) after default collate
            for i in range(len(preds)):
                record = {
                    "global_index": int(b_gidx[i]),
                    "problem_id": str(b_pid[i]),
                    "buggy_submission_id": int(b_bsid[i]),
                    "fixed_submission_id": int(b_fsid[i]),
                    "language": str(b_lang[i]),
                    "preds": [preds[i]],
                    "gt_fixed_code": str(b_gt[i]),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[Rank {r}] wrote -> {out_rank}")

    # ---- merge on rank0 ----
    ddp_barrier()
    if ddp_enabled() and is_main():
        merged_path = base_out
        with open(merged_path, "w", encoding="utf-8") as fout:
            for rr in range(ddp_world()):
                part = base_out.replace(".jsonl", f".rank{rr}.jsonl")
                if not os.path.exists(part):
                    print(f"[Merge] missing shard: {part}")
                    continue
                with open(part, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
        print(f"[Merge] merged jsonl -> {merged_path}")

    if ddp_enabled():
        ddp_cleanup()


if __name__ == "__main__":
    main()
