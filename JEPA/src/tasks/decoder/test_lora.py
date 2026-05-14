#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_exp3_decoder_lora.py

Decode repaired code from embedding pt using:
- StarCoder2-3B
- trained prompt projector
- trained ln
- optional trained LoRA weights

Expected emb_pt format:
{
  "z_pred": Tensor[N, D],
  "global_indices": LongTensor[N]
}

Supported checkpoint formats:
1) tunable-only checkpoint:
{
  "projector": ...,
  "ln": ...,
  "lora": ...
}

2) lightweight resume checkpoint:
{
  "epoch": ...,
  "global_step": ...,
  "best_val": ...,
  "projector": ...,
  "ln": ...,
  "lora": ...,
  "optimizer": ...,
  "scaler": ...
}

Output JSONL format:
{
  "global_index": ...,
  "problem_id": ...,
  "buggy_submission_id": ...,
  "fixed_submission_id": ...,
  "language": ...,
  "preds": [...],
  "gt_fixed_code": ...
}
"""

import os
import json
import argparse
from typing import Any, List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from jepa.utils import (
    ddp_barrier,
    ddp_cleanup,
    ddp_enabled,
    ddp_local_rank,
    ddp_rank,
    ddp_setup,
    ddp_world,
    is_main,
    load_embeddings_pt,
)


# =====================
# Model
# =====================
class SoftPromptStarCoderDecoder(nn.Module):
    def __init__(self, cond_dim: int, decoder_model, tokenizer, prompt_len: int = 128):
        super().__init__()
        self.decoder = decoder_model
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.hidden_dim = int(decoder_model.config.hidden_size)

        inter_dim = cond_dim * 4
        self.prompt_proj = nn.Sequential(
            nn.Linear(cond_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(inter_dim, self.prompt_len * self.hidden_dim),
        )
        self.ln = nn.LayerNorm(self.hidden_dim)

        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        self.eos_token_id = tokenizer.eos_token_id

    @torch.no_grad()
    def generate_fast(self, cond_emb: torch.Tensor, max_new_tokens: int = 512) -> List[str]:
        self.eval()
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
            eos_token_id=self.eos_token_id,
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


# =====================
# Checkpoint loader
# =====================
def load_inference_weights(model: nn.Module, path: str, use_lora: bool) -> Dict[str, Any]:
    """
    Load projector / ln / optional lora weights from either:
    - tunable-only checkpoint
    - lightweight resume checkpoint

    Returns checkpoint metadata if present.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "projector" not in ckpt or "ln" not in ckpt:
        raise KeyError("Checkpoint must contain at least 'projector' and 'ln'")

    metadata = {
        "epoch": ckpt.get("epoch", None),
        "global_step": ckpt.get("global_step", None),
        "best_val": ckpt.get("best_val", None),
    }

    model.prompt_proj.load_state_dict(ckpt["projector"], strict=True)
    model.ln.load_state_dict(ckpt["ln"], strict=True)

    print(f"[Load] projector loaded")
    print(f"[Load] ln loaded")

    if use_lora:
        if "lora" not in ckpt:
            raise KeyError("Checkpoint does not contain 'lora', but --use_lora was specified.")

        current_lora_state = {
            name: p
            for name, p in model.decoder.named_parameters()
            if p.requires_grad
        }

        missing = []
        loaded = 0
        for name, tensor in ckpt["lora"].items():
            if name in current_lora_state:
                current_lora_state[name].data.copy_(tensor)
                loaded += 1
            else:
                missing.append(name)

        print(f"[Load] lora params loaded: {loaded}")
        if missing:
            print(f"[Warn] missing current lora keys for ckpt items: {missing[:10]}")
    else:
        if "lora" in ckpt and is_main():
            print("[Info] checkpoint contains 'lora', but --use_lora was not specified. LoRA weights are ignored.")

    return metadata


# =====================
# Dataset
# =====================
class DecodeDataset(Dataset):
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


# =====================
# Args
# =====================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--emb_pt", type=str, required=True)
    ap.add_argument("--emb_key", type=str, default="z_pred")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)

    ap.add_argument("--decoder_model_id", type=str, default="bigcode/starcoder2-3b")
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_items", type=int, default=-1, help="Per-rank max items. -1 means no limit.")
    ap.add_argument("--use_bf16", action="store_true")

    ap.add_argument("--hf_dataset_id", type=str, default="ASSERT-KTH/RunBugRun-Final")
    ap.add_argument("--hf_split", type=str, default="train")
    ap.add_argument("--hf_fixed_field", type=str, default="fixed_code")

    # LoRA options (must match training if used)
    ap.add_argument("--use_lora", action="store_true", help="Enable LoRA structure before loading checkpoint")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "c_proj"],
    )

    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    if ddp_enabled():
        ddp_setup("nccl" if torch.cuda.is_available() else "gloo")

    device = torch.device(f"cuda:{ddp_local_rank()}") if torch.cuda.is_available() else torch.device("cpu")
    r, w = ddp_rank(), ddp_world()

    if is_main():
        print(f"[DDP] enabled={ddp_enabled()} world_size={w}")
        print(f"[Args] ckpt={args.ckpt}")
        print(f"[Args] emb_pt={args.emb_pt}")
        print(f"[Args] out_jsonl={args.out_jsonl}")
        print(f"[Args] use_lora={args.use_lora}")
    print(f"[Rank {r}] device={device}")

    base_out = args.out_jsonl
    out_rank = base_out.replace(".jsonl", f".rank{r}.jsonl")
    out_dir = os.path.dirname(out_rank)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # load embeddings
    emb_all, gidx_all = load_embeddings_pt(args.emb_pt, key=args.emb_key)
    if is_main():
        print(f"[Data] emb_all shape={tuple(emb_all.shape)} from {args.emb_pt}")

    # shard by rank
    emb_shard = emb_all[r::w].contiguous()
    gidx_shard = gidx_all[r::w].tolist()

    # optional truncation (per-rank)
    if args.max_items > 0:
        emb_shard = emb_shard[:args.max_items].contiguous()
        gidx_shard = gidx_shard[:args.max_items]

    print(f"[Rank {r}] emb_shard shape={tuple(emb_shard.shape)} num_items={len(gidx_shard)}")

    # load HF metadata
    ds = load_dataset(args.hf_dataset_id, split=args.hf_split)
    subset = ds.select([int(x) for x in gidx_shard])

    gt_fixed_codes = [str(x) if x is not None else "" for x in subset[args.hf_fixed_field]]
    languages = subset["language"]
    problem_ids = subset["problem_id"]
    buggy_sids = subset["buggy_submission_id"]
    fixed_sids = subset["fixed_submission_id"]

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # base decoder
    decoder = AutoModelForCausalLM.from_pretrained(
        args.decoder_model_id,
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    # optional LoRA structure
    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        decoder = get_peft_model(decoder, lora_config)
        if is_main():
            print("[Model] LoRA structure rebuilt for inference.")
    else:
        if is_main():
            print("[Model] LoRA disabled for inference.")

    decoder.eval()

    cond_dim = int(emb_shard.shape[1])
    model = SoftPromptStarCoderDecoder(cond_dim, decoder, tokenizer, prompt_len=args.prompt_len).to(device)

    if device.type == "cuda" and args.use_bf16:
        try:
            model.prompt_proj.to(torch.bfloat16)
            model.ln.to(torch.bfloat16)
        except Exception as e:
            print(f"[Warn] bf16 cast failed: {e}")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt_meta = load_inference_weights(model, args.ckpt, use_lora=args.use_lora)

    if is_main():
        if ckpt_meta["epoch"] is not None:
            print(f"[CKPT] epoch = {ckpt_meta['epoch']}")
        if ckpt_meta["global_step"] is not None:
            print(f"[CKPT] global_step = {ckpt_meta['global_step']}")
        if ckpt_meta["best_val"] is not None:
            print(f"[CKPT] best_val = {ckpt_meta['best_val']}")

    ds_decode = DecodeDataset(
        emb_cpu=emb_shard,
        global_indices=gidx_shard,
        problem_ids=problem_ids,
        buggy_sids=buggy_sids,
        fixed_sids=fixed_sids,
        languages=languages,
        gt_fixed_codes=gt_fixed_codes,
    )

    dl = DataLoader(
        ds_decode,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    with open(out_rank, "w", encoding="utf-8") as f:
        pbar = tqdm(dl, desc=f"Decoding(rank {r})", leave=True)
        for b_emb, b_gidx, b_pid, b_bsid, b_fsid, b_lang, b_gt in pbar:
            b_emb = b_emb.to(device, non_blocking=True)
            preds = model.generate_fast(b_emb, max_new_tokens=args.max_new_tokens)

            if torch.is_tensor(b_gidx):
                b_gidx = b_gidx.tolist()
            if torch.is_tensor(b_bsid):
                b_bsid = b_bsid.tolist()
            if torch.is_tensor(b_fsid):
                b_fsid = b_fsid.tolist()

            for i in range(len(preds)):
                rec = {
                    "global_index": int(b_gidx[i]),
                    "problem_id": str(b_pid[i]),
                    "buggy_submission_id": int(b_bsid[i]),
                    "fixed_submission_id": int(b_fsid[i]),
                    "language": str(b_lang[i]),
                    "preds": [preds[i]],
                    "gt_fixed_code": str(b_gt[i]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Rank {r}] wrote -> {out_rank}")

    ddp_barrier()

    # merge outputs
    if ddp_enabled():
        if is_main():
            with open(base_out, "w", encoding="utf-8") as fout:
                total_lines = 0
                for rr in range(w):
                    part = base_out.replace(".jsonl", f".rank{rr}.jsonl")
                    if not os.path.exists(part):
                        print(f"[Merge] missing shard: {part}")
                        continue
                    with open(part, "r", encoding="utf-8") as fin:
                        for line in fin:
                            fout.write(line)
                            total_lines += 1
            print(f"[Merge] merged -> {base_out} (total_lines={total_lines})")
    else:
        if out_rank != base_out:
            with open(out_rank, "r", encoding="utf-8") as fin, open(base_out, "w", encoding="utf-8") as fout:
                total_lines = 0
                for line in fin:
                    fout.write(line)
                    total_lines += 1
            if is_main():
                print(f"[Merge] single-rank copy -> {base_out} (total_lines={total_lines})")

    if ddp_enabled():
        ddp_cleanup()


if __name__ == "__main__":
    main()
