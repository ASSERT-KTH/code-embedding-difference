#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RunBugRun TEST generation using a finetuned LoRA checkpoint (PEFT adapter).
"""

import os
import json
import time
import numpy as np
import torch
import difflib
from tqdm import tqdm
from typing import List, Optional, Union

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sacrebleu import corpus_bleu

from peft import PeftModel


# =====================
# Defaults (override via CLI)
# =====================
BASE_MODEL_ID_DEFAULT = "bigcode/starcoder2-3b"
HF_DATASET_ID = "ASSERT-KTH/RunBugRun-Final"

# fixed tokenization for gt_len / metrics
MAX_LEN_FIXED = 1024

# prompt input max length (prompt + buggy)
MAX_LEN_PROMPT_INPUT = 1280

# generation config
MAX_NEW_TOKENS = 4096
MIN_NEW_TOKENS = 64

# dataloader
BATCH_SIZE = 32
NUM_WORKERS = 1

# sampling config
N_SAMPLES = 1
DO_SAMPLE = True
TEMPERATURE = 0.2
TOP_P = 0.95

# debug
DEBUG_PRINT_FIRST_BATCH = True

# metrics (text-only, candidate0, over shard)
DO_TEXT_METRICS = True


# =====================
# Multi-process helpers (torchrun)
# =====================
def get_dist_info():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world, local_rank


def shard_list(xs: List, rank: int, world: int):
    return xs[rank::world]


def shard_ndarray(x: np.ndarray, rank: int, world: int):
    return x[rank::world]


# =====================
# Diff-Match helpers
# =====================
def normalize_code_lines(code: str) -> List[str]:
    if code is None:
        code = ""
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    return [ln.rstrip() for ln in code.split("\n")]


def diff_match_score(buggy: str, gt: str, pred: str) -> float:
    b_lines = normalize_code_lines(buggy)
    g_lines = normalize_code_lines(gt)
    p_lines = normalize_code_lines(pred)

    sm = difflib.SequenceMatcher(a=b_lines, b=g_lines)
    opcodes = sm.get_opcodes()

    changed_gt_indices = set()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        for j in range(j1, j2):
            changed_gt_indices.add(j)

    if len(changed_gt_indices) == 0:
        return 1.0

    match, total = 0, 0
    for j in changed_gt_indices:
        total += 1
        g_line = g_lines[j] if j < len(g_lines) else ""
        p_line = p_lines[j] if j < len(p_lines) else ""
        if p_line == g_line:
            match += 1

    return match / total if total > 0 else 0.0


# =====================
# Dataset
# =====================
class PromptBaselineTestDataset(Dataset):
    def __init__(
        self,
        fixed_ids, fixed_mask,
        buggy_texts, gt_fixed_texts,
        languages, global_indices,
        problem_ids, buggy_submission_ids, fixed_submission_ids,
    ):
        self.fixed_ids = fixed_ids
        self.fixed_mask = fixed_mask

        self.buggy_texts = buggy_texts
        self.gt_fixed_texts = gt_fixed_texts
        self.languages = languages

        self.global_indices = global_indices
        self.problem_ids = problem_ids
        self.buggy_submission_ids = buggy_submission_ids
        self.fixed_submission_ids = fixed_submission_ids

    def __len__(self):
        return self.fixed_ids.size(0)

    def __getitem__(self, idx):
        return (
            self.fixed_ids[idx],
            self.fixed_mask[idx],
            self.buggy_texts[idx],
            self.gt_fixed_texts[idx],
            str(self.languages[idx]),
            int(self.global_indices[idx]),
            str(self.problem_ids[idx]),
            int(self.buggy_submission_ids[idx]),
            int(self.fixed_submission_ids[idx]),
        )


# =====================
# Load indices + HF subset
# =====================
def load_test_subset_from_saved_indices(indices_dir: str):
    global_target_indices = np.load(os.path.join(indices_dir, "global_target_indices.npy"))
    test_idx = np.load(os.path.join(indices_dir, "test_idx.npy"))
    global_test_indices = global_target_indices[test_idx]

    print(f"[Info] Subset size={len(global_target_indices)} | test size={len(test_idx)}")

    print("[Info] Loading HF dataset...")
    ds = load_dataset(HF_DATASET_ID, split="train")
    subset = ds.select(global_target_indices.tolist())

    buggy_texts_all = [str(x) if x is not None else "" for x in subset["buggy_code"]]
    fixed_texts_all = [str(x) if x is not None else "" for x in subset["fixed_code"]]
    languages_all = subset["language"]
    problem_ids_all = subset["problem_id"]
    buggy_submission_ids_all = subset["buggy_submission_id"]
    fixed_submission_ids_all = subset["fixed_submission_id"]

    buggy_texts = [buggy_texts_all[i] for i in test_idx]
    fixed_texts = [fixed_texts_all[i] for i in test_idx]
    languages = [languages_all[i] for i in test_idx]
    problem_ids = [problem_ids_all[i] for i in test_idx]
    buggy_sids = [buggy_submission_ids_all[i] for i in test_idx]
    fixed_sids = [fixed_submission_ids_all[i] for i in test_idx]

    return (
        buggy_texts, fixed_texts, languages,
        global_test_indices, problem_ids, buggy_sids, fixed_sids
    )


# =====================
# Prompt (aligned to your training script)
# =====================
def build_prompt(buggy_code: str, language: str) -> str:
    buggy_code = (buggy_code or "").rstrip()
    language = (language or "").strip()

    lang_low = language.lower()
    if "python" in lang_low:
        fence_lang = "python"
    elif "c++" in lang_low or "cpp" in lang_low:
        fence_lang = "cpp"
    elif lang_low == "c":
        fence_lang = "c"
    elif "java" in lang_low:
        fence_lang = "java"
    else:
        fence_lang = ""

    return (
        f"Your task is to fix the {language} code.\n"
        "Here is the buggy program:\n"
        "```\n"
        f"{buggy_code}\n"
        "```\n\n"
        "Return only the fixed code in a single Markdown code block (triple backticks). No explanation.\n\n"
        "Here is the fixed program:\n"
        f"```{fence_lang}\n"
    )


# =====================
# Generation helpers
# =====================
StopIds = Union[int, List[int]]

def cut_at_first_stop(ids: torch.Tensor, stop_ids: StopIds) -> torch.Tensor:
    """Cut token sequence at the first occurrence of any stop_id (inclusive)."""
    if stop_ids is None:
        return ids
    if isinstance(stop_ids, int):
        stop_ids = [stop_ids]
    if len(stop_ids) == 0:
        return ids

    mask = torch.zeros_like(ids, dtype=torch.bool)
    for sid in stop_ids:
        if sid is None:
            continue
        mask |= (ids == int(sid))

    pos = mask.nonzero(as_tuple=False)
    if pos.numel() == 0:
        return ids

    first = int(pos[0].item())
    return ids[: first + 1]


def first_stop_pos(ids: torch.Tensor, stop_ids: StopIds) -> int:
    """Return first stop position in ids, or -1."""
    if stop_ids is None:
        return -1
    if isinstance(stop_ids, int):
        stop_ids = [stop_ids]
    if len(stop_ids) == 0:
        return -1

    mask = torch.zeros_like(ids, dtype=torch.bool)
    for sid in stop_ids:
        if sid is None:
            continue
        mask |= (ids == int(sid))
    pos = mask.nonzero(as_tuple=False)
    return int(pos[0].item()) if pos.numel() > 0 else -1


def infer_finish_reason(new_ids_trim: torch.Tensor, stop_ids: StopIds, max_new_tokens: int) -> str:
    """
    Heuristic finish reason:
      - if last token is a stop_id => "stop"
      - elif length hits max_new_tokens => "length"
      - else => "unknown"
    """
    if new_ids_trim is None or int(new_ids_trim.numel()) == 0:
        return "unknown"

    last_id = int(new_ids_trim[-1].item())
    if isinstance(stop_ids, int):
        stop_set = {int(stop_ids)}
    else:
        stop_set = set(int(x) for x in (stop_ids or []) if x is not None)

    if last_id in stop_set:
        return "stop"
    if int(new_ids_trim.numel()) >= int(max_new_tokens):
        return "length"
    return "unknown"


# =====================
# Batch-parallel generation (FULL logging, padding removed from logs)
# =====================
@torch.no_grad()
def generate_full_logs_batch(
    decoder,
    tokenizer,
    buggy_texts: List[str],
    languages: List[str],
    device: str,
    stop_ids: StopIds,
    debug: bool = False
):

    B = len(buggy_texts)
    prompts = [build_prompt(b, l) for b, l in zip(buggy_texts, languages)]

    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN_PROMPT_INPUT,
        return_tensors="pt"
    ).to(device)

    # true prompt length without padding
    prompt_lens_tensor = enc.attention_mask.sum(dim=1).long()
    prompt_len_tokens = [int(x.item()) for x in prompt_lens_tensor]

    gen_out = decoder.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=MIN_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE if DO_SAMPLE else None,
        top_p=TOP_P if DO_SAMPLE else None,
        num_return_sequences=N_SAMPLES,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_ids,
        return_dict_in_generate=True,
        output_scores=False,
    )

    sequences = gen_out.sequences
    if N_SAMPLES > 1:
        sequences = sequences.view(B, N_SAMPLES, -1)
    else:
        sequences = sequences.unsqueeze(1)

    input_len = enc.input_ids.shape[1]

    full_texts: List[List[str]] = []
    continuation_texts: List[List[str]] = []
    full_len_tokens: List[List[int]] = []
    new_len_tokens: List[List[int]] = []
    finish_reason: List[List[str]] = []
    stop_pos_in_new: List[List[int]] = []

    for i in range(B):
        # prompt ids without padding
        prompt_ids_nopad = enc.input_ids[i][enc.attention_mask[i].bool()]

        full_i, cont_i = [], []
        full_l_i, new_l_i = [], []
        fr_i, stoppos_i = [], []

        for k in range(N_SAMPLES):
            seq_ids = sequences[i, k]
            new_ids = seq_ids[input_len:]  # continuation: after padded input_len

            stoppos = first_stop_pos(new_ids, stop_ids)
            new_ids_trim = cut_at_first_stop(new_ids, stop_ids)

            seq_ids_trim = torch.cat([prompt_ids_nopad, new_ids_trim], dim=0)

            full_txt = tokenizer.decode(seq_ids_trim, skip_special_tokens=False).strip()
            cont_txt = tokenizer.decode(new_ids_trim, skip_special_tokens=False).strip()

            full_i.append(full_txt)
            cont_i.append(cont_txt)

            full_l_i.append(int(seq_ids_trim.numel()))
            new_l_i.append(int(new_ids_trim.numel()))

            fr_i.append(infer_finish_reason(new_ids_trim, stop_ids, MAX_NEW_TOKENS))
            stoppos_i.append(int(stoppos))

        full_texts.append(full_i)
        continuation_texts.append(cont_i)
        full_len_tokens.append(full_l_i)
        new_len_tokens.append(new_l_i)
        finish_reason.append(fr_i)
        stop_pos_in_new.append(stoppos_i)

    if debug and B > 0:
        print("\n========== [DEBUG FIRST BATCH] ==========")
        print("[DEBUG] buggy_code[:300]:")
        print((buggy_texts[0] or "")[:300])
        print("\n[DEBUG] prompt[:800]:")
        print(prompts[0][:800])
        print(f"\n[DEBUG] prompt_len_tokens[0]={prompt_len_tokens[0]}  input_len(padded)={input_len}  B={B}  N_SAMPLES={N_SAMPLES}")
        print(f"[DEBUG] pad_id={tokenizer.pad_token_id} eos_id={tokenizer.eos_token_id} stop_ids={stop_ids}")
        print(f"[DEBUG] new_len_tokens[0][0]={new_len_tokens[0][0]}")
        print(f"[DEBUG] finish_reason[0][0]={finish_reason[0][0]}")
        print(f"[DEBUG] stop_pos_in_new[0][0]={stop_pos_in_new[0][0]}")
        print("\n[DEBUG] continuation_text[0][0][:800]:")
        print((continuation_texts[0][0] or "")[:800])
        print("=========================================\n")

    return (
        prompts,
        prompt_len_tokens,
        full_texts,
        continuation_texts,
        full_len_tokens,
        new_len_tokens,
        finish_reason,
        stop_pos_in_new,
    )


# =====================
# Test + Save JSONL
# =====================
@torch.no_grad()
def run_test_and_save(decoder, loader, tokenizer, save_jsonl_path: str, device: str, rank: int, world: int, stop_ids: StopIds):
    os.makedirs(os.path.dirname(save_jsonl_path), exist_ok=True)
    with open(save_jsonl_path, "w", encoding="utf-8") as f_out:
        print(f"[Rank{rank}] Saving JSONL -> {save_jsonl_path}")

        all_preds, all_gts = [], []
        em_cnt = 0
        diff_scores = []
        printed_debug = False

        pbar = tqdm(loader, desc=f"Testing(rank{rank}/{world})", leave=True)
        for (
            fixed_ids, fixed_mask,
            buggy_text, gt_fixed_text, lang,
            gidx, pid, buggy_sid, fixed_sid
        ) in pbar:
            fixed_ids = fixed_ids.to(device, non_blocking=True)
            fixed_mask = fixed_mask.to(device, non_blocking=True)

            debug_now = DEBUG_PRINT_FIRST_BATCH and (not printed_debug) and (rank == 0)

            (
                prompts,
                prompt_len_tokens,
                full_texts,
                continuation_texts,
                full_len_tokens,
                new_len_tokens,
                finish_reason,
                stop_pos_in_new,
            ) = generate_full_logs_batch(
                decoder, tokenizer,
                buggy_texts=list(buggy_text),
                languages=list(lang),
                device=device,
                stop_ids=stop_ids,
                debug=debug_now
            )
            if debug_now:
                printed_debug = True

            gts_texts = tokenizer.batch_decode(fixed_ids, skip_special_tokens=True)

            B = len(prompts)
            for i in range(B):
                rec = {
                    # ids/meta
                    "global_index": int(gidx[i]),
                    "problem_id": str(pid[i]),
                    "buggy_submission_id": int(buggy_sid[i]),
                    "fixed_submission_id": int(fixed_sid[i]),
                    "language": str(lang[i]),
                    "shard_rank": rank,
                    "shard_world_size": world,

                    # inputs / references
                    "buggy_code": str(buggy_text[i]),
                    "gt_fixed_code": str(gt_fixed_text[i]),
                    "prompt": str(prompts[i]),

                    # outputs (per candidate) - padding removed from full_texts
                    "full_texts": full_texts[i],
                    "continuation_texts": continuation_texts[i],

                    # lengths / stopping (trimmed)
                    "prompt_len_tokens": int(prompt_len_tokens[i]),
                    "full_len_tokens": [int(x) for x in full_len_tokens[i]],
                    "new_len_tokens": [int(x) for x in new_len_tokens[i]],
                    "finish_reason": finish_reason[i],
                    "stop_pos_in_new": stop_pos_in_new[i],
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # shard-local metrics (candidate0 only) over continuation_texts
            if DO_TEXT_METRICS:
                preds0 = [
                    (continuation_texts[i][0] if (isinstance(continuation_texts[i], list) and len(continuation_texts[i]) > 0) else "")
                    for i in range(B)
                ]
                for p, g, b in zip(preds0, gts_texts, buggy_text):
                    if p == g:
                        em_cnt += 1
                    diff_scores.append(diff_match_score(str(b), str(g), str(p)))
                all_preds.extend(preds0)
                all_gts.extend(gts_texts)

    # metrics
    if DO_TEXT_METRICS:
        try:
            bleu = corpus_bleu(all_preds, [all_gts]).score
        except Exception:
            bleu = 0.0

        em = em_cnt / max(len(all_preds), 1) * 100.0
        diffm = float(np.mean(diff_scores)) * 100.0 if diff_scores else 0.0

        if rank == 0:
            print("\n====================")
            print("[Rank0] NOTE: metrics below are for Rank0 shard only.")
            print(f"[TEST] BLEU      : {bleu:.2f}")
            print(f"[TEST] EM        : {em:.2f}%")
            print(f"[TEST] DiffMatch : {diffm:.2f}%")
            print("====================\n")


# =====================
# Main
# =====================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", type=str, required=True,
                    help="Path to your PEFT adapter dir (output_dir or checkpoint-XXXX)")
    ap.add_argument("--base_model_id", type=str, default=BASE_MODEL_ID_DEFAULT,
                    help="Base model id (default: bigcode/starcoder2-3b)")
    ap.add_argument("--indices_dir", type=str, default="saved_indices",
                    help="Directory containing saved_indices/*.npy")
    ap.add_argument("--save_dir", type=str, default="results_test_lora",
                    help="Output directory to store rank shards jsonl")
    args = ap.parse_args()

    assert os.path.exists(args.indices_dir), f"indices dir not found: {args.indices_dir}"
    assert os.path.exists(args.adapter_dir), f"adapter_dir not found: {args.adapter_dir}"

    rank, world, local_rank = get_dist_info()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    if rank == 0:
        print(f"[Dist] world_size={world} | using torchrun env vars")
    print(f"[Rank{rank}] local_rank={local_rank} device={device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    (
        buggy_texts, fixed_texts, languages,
        global_test_indices, problem_ids, buggy_sids, fixed_sids
    ) = load_test_subset_from_saved_indices(args.indices_dir)

    buggy_texts = shard_list(buggy_texts, rank, world)
    fixed_texts = shard_list(fixed_texts, rank, world)
    languages = shard_list(languages, rank, world)
    global_test_indices = shard_ndarray(global_test_indices, rank, world)
    problem_ids = shard_list(problem_ids, rank, world)
    buggy_sids = shard_list(buggy_sids, rank, world)
    fixed_sids = shard_list(fixed_sids, rank, world)

    print(f"[Rank{rank}] shard size = {len(buggy_texts)}")

    # -------------------------
    # Tokenizer
    # loading tokenizer from adapter_dir
    # -------------------------
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
        if rank == 0:
            print(f"[Tokenizer] Loaded from adapter_dir: {args.adapter_dir}")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True)
        if rank == 0:
            print(f"[Tokenizer] adapter_dir has no tokenizer; loaded from base: {args.base_model_id}")

    # Ensure special tokens consistent with training
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    if "<END>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})

    tokenizer.padding_side = "left"

    end_id = tokenizer.convert_tokens_to_ids("<END>")
    stop_ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))
    if end_id is not None and end_id != tokenizer.unk_token_id:
        stop_ids.append(int(end_id))

    if rank == 0:
        print(f"[Tokenizer] pad_id={tokenizer.pad_token_id} eos_id={tokenizer.eos_token_id} end_id={end_id} stop_ids={stop_ids}")
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            print("[Warn] pad_id == eos_id (NOT recommended). You may want a real PAD token.")

    # -------------------------
    # Tokenize GT FIXED (for metrics only)
    # -------------------------
    if rank == 0:
        print("[Info] Tokenizing FIXED (for metrics only)...")
    enc_fixed = tokenizer(
        fixed_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN_FIXED,
        return_tensors="pt"
    )

    test_ds = PromptBaselineTestDataset(
        fixed_ids=enc_fixed.input_ids,
        fixed_mask=enc_fixed.attention_mask,
        buggy_texts=buggy_texts,
        gt_fixed_texts=fixed_texts,
        languages=languages,
        global_indices=global_test_indices,
        problem_ids=problem_ids,
        buggy_submission_ids=buggy_sids,
        fixed_submission_ids=fixed_sids,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # -------------------------
    # Load BASE model + attach LoRA adapter
    # -------------------------
    if rank == 0:
        print("[Info] Loading BASE model (bf16) + PEFT adapter (LoRA)...")
        print(f"[Info] base_model_id = {args.base_model_id}")
        print(f"[Info] adapter_dir   = {args.adapter_dir}")

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(device)

    base.resize_token_embeddings(len(tokenizer))

    decoder = PeftModel.from_pretrained(base, args.adapter_dir)
    decoder = decoder.to(device)

    decoder.resize_token_embeddings(len(tokenizer))

    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    os.makedirs(args.save_dir, exist_ok=True)
    save_jsonl = os.path.join(args.save_dir, f"fixed_code_test.rank{rank}.jsonl")

    t0 = time.time()
    run_test_and_save(decoder, test_loader, tokenizer, save_jsonl, device, rank, world, stop_ids)
    print(f"[Rank{rank}] Done. Elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
