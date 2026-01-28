# baseline_softprompt_train_test.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Baseline: Frozen StarCoder2-3B + global learnable soft prompt (token-space)
"""

import os
import sys
import time
import json
import random
import difflib
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sacrebleu import corpus_bleu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Optional CodeBLEU
# -----------------------------
try:
    from codebleu import calc_codebleu  # type: ignore
    HAS_CODEBLEU = True
except Exception:
    HAS_CODEBLEU = False


# =====================
# Config (defaults)
# =====================
DECODER_MODEL_ID = "bigcode/starcoder2-3b"
HF_DATASET_ID = "ASSERT-KTH/RunBugRun-Final"

SAVED_INDICES_DIR = "saved_indices"

LOG_DIR = "logs_token_baseline_b1"
CKPT_DIR = "checkpoints_token_baseline_b1"

MAX_LEN_BUGGY = 256
MAX_LEN_FIXED = 256

PROMPT_LEN = 64
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 16
NUM_WORKERS = 4

NUM_EPOCHS = 2
LR = 1e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
RANDOM_SEED = 42

SAVE_EVERY_EPOCH = 1

VAL_CHECK_LIMIT = 1500

MAX_NEW_TOKENS = 256
EXTRA_TOKENS_OVER_GT = 64

LOG_EVERY_STEPS = 100

DO_CODEBLEU = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NORMALIZE_LANG_MAP = {
    "python": "python",
    "java": "java",
    "javascript": "javascript",
    "js": "javascript",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "php": "php",
    "go": "go",
    "ruby": "ruby",
    "c#": "c_sharp",
}

# Test outputs
SAVE_DIR_TEST = "results_baseline_test"
SAVE_JSONL_TEST = os.path.join(SAVE_DIR_TEST, "fixed_code_test.jsonl")


# =====================
# Try importing shared utilities
# =====================
_SHARED_OK = True
try:
    # Example: adjust these imports to match your existing project structure
    # from utils.logger import Logger  # noqa: F401
    # from data.runbugrun import (
    #     load_saved_indices,
    #     load_subset_by_global_indices,
    #     load_test_subset_from_saved_indices,
    # )  # noqa: F401
    raise ImportError("Shared utilities import placeholders not configured.")
except Exception:
    _SHARED_OK = False


# =====================
# Logger (fallback)
# =====================
if not _SHARED_OK:
    class Logger(object):
        def __init__(self, filename: str = "training_log.txt"):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding="utf-8")

        def write(self, message: str):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()


# =====================
# Data loading (fallback)
# =====================
if not _SHARED_OK:
    def load_saved_indices(saved_dir: str):
        global_target_indices = np.load(os.path.join(saved_dir, "global_target_indices.npy"))
        train_idx = np.load(os.path.join(saved_dir, "train_idx.npy"))
        val_idx = np.load(os.path.join(saved_dir, "val_idx.npy"))
        test_idx = np.load(os.path.join(saved_dir, "test_idx.npy"))
        return global_target_indices, train_idx, val_idx, test_idx

    def load_subset_by_global_indices(global_target_indices: np.ndarray):
        ds = load_dataset(HF_DATASET_ID, split="train")
        subset = ds.select(global_target_indices.tolist())

        buggy_texts = [str(x) if x is not None else "" for x in subset["buggy_code"]]
        fixed_texts = [str(x) if x is not None else "" for x in subset["fixed_code"]]
        languages = subset["language"]
        return buggy_texts, fixed_texts, languages

    def load_test_subset_from_saved_indices(indices_dir: str):
        global_target_indices = np.load(os.path.join(indices_dir, "global_target_indices.npy"))
        test_idx = np.load(os.path.join(indices_dir, "test_idx.npy"))
        global_test_indices = global_target_indices[test_idx]

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
# Helpers: diff-match
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

    match = 0
    total = 0
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
class TokenRepairDataset(Dataset):
    def __init__(self, buggy_ids, buggy_mask, fixed_ids, fixed_mask, languages):
        self.buggy_ids = buggy_ids
        self.buggy_mask = buggy_mask
        self.fixed_ids = fixed_ids
        self.fixed_mask = fixed_mask
        self.languages = languages

    def __len__(self):
        return self.fixed_ids.size(0)

    def __getitem__(self, idx):
        return (
            self.buggy_ids[idx],
            self.buggy_mask[idx],
            self.fixed_ids[idx],
            self.fixed_mask[idx],
            self.languages[idx],
        )


class TokenRepairTestDataset(Dataset):
    def __init__(
        self,
        buggy_ids, buggy_mask,
        fixed_ids, fixed_mask,
        buggy_texts, gt_fixed_texts,
        languages, global_indices,
        problem_ids, buggy_submission_ids, fixed_submission_ids,
    ):
        self.buggy_ids = buggy_ids
        self.buggy_mask = buggy_mask
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
        return self.buggy_ids.size(0)

    def __getitem__(self, idx):
        return (
            self.buggy_ids[idx],
            self.buggy_mask[idx],
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
# Model
# =====================
class SoftPromptTokenRepair(nn.Module):
    def __init__(self, decoder_model, tokenizer, prompt_len: int = 64):
        super().__init__()
        self.decoder = decoder_model
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.hidden_dim = decoder_model.config.hidden_size

        self.soft_prompt = nn.Parameter(torch.zeros(prompt_len, self.hidden_dim))
        nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)

        for p in self.decoder.parameters():
            p.requires_grad = False
        self.decoder.eval()

        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        self.eos_token_id = tokenizer.eos_token_id

    def forward(self, buggy_ids, buggy_mask, fixed_ids, fixed_mask):
        B = buggy_ids.size(0)

        buggy_emb = self.decoder.get_input_embeddings()(buggy_ids).to(self.decoder.dtype)
        fixed_emb = self.decoder.get_input_embeddings()(fixed_ids).to(self.decoder.dtype)

        prompt = self.soft_prompt.unsqueeze(0).expand(B, -1, -1).to(self.decoder.dtype)
        full_emb = torch.cat([prompt, buggy_emb, fixed_emb], dim=1)

        prompt_mask = torch.ones(B, self.prompt_len, device=buggy_mask.device, dtype=buggy_mask.dtype)
        full_mask = torch.cat([prompt_mask, buggy_mask, fixed_mask], dim=1)

        full_labels = torch.full((B, full_emb.size(1)), -100, device=buggy_ids.device, dtype=torch.long)
        fixed_labels = fixed_ids.clone()
        fixed_labels[fixed_mask == 0] = -100

        start = self.prompt_len + buggy_ids.size(1)
        full_labels[:, start:] = fixed_labels

        out = self.decoder(inputs_embeds=full_emb, attention_mask=full_mask, labels=full_labels)
        return out.loss

    @torch.no_grad()
    def generate_fast_ids(self, buggy_ids, buggy_mask, max_new_tokens: int = 256) -> torch.Tensor:
        self.decoder.eval()
        B = buggy_ids.size(0)

        buggy_emb = self.decoder.get_input_embeddings()(buggy_ids).to(self.decoder.dtype)
        prompt = self.soft_prompt.unsqueeze(0).expand(B, -1, -1).to(self.decoder.dtype)
        cond_emb = torch.cat([prompt, buggy_emb], dim=1)

        prompt_mask = torch.ones(B, self.prompt_len, device=buggy_mask.device, dtype=buggy_mask.dtype)
        cond_mask = torch.cat([prompt_mask, buggy_mask], dim=1)

        input_len = cond_emb.size(1)

        gen_ids = self.decoder.generate(
            inputs_embeds=cond_emb,
            attention_mask=cond_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        if gen_ids.size(1) > input_len:
            gen_ids = gen_ids[:, input_len:]

        return gen_ids


def save_tunable_parameters(model: SoftPromptTokenRepair, path: str):
    saved = {"soft_prompt": model.soft_prompt.detach().to("cpu")}
    torch.save(saved, path)
    print(f"  >>> [Saved] Tunable params only -> {path}")


def load_soft_prompt_checkpoint(model: SoftPromptTokenRepair, ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "soft_prompt" in state:
        sp = state["soft_prompt"]
        if sp.shape != model.soft_prompt.shape:
            raise ValueError(f"soft_prompt shape mismatch: ckpt={tuple(sp.shape)} vs model={tuple(model.soft_prompt.shape)}")
        model.soft_prompt.data.copy_(sp)
        print(f"[OK] Loaded soft_prompt from {ckpt_path}")
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[OK] Loaded state_dict (strict=False) from {ckpt_path}")
        if unexpected:
            print(f"[Warn] unexpected keys: {unexpected[:5]} ...")
        if missing:
            print(f"[Info] missing keys: {missing[:5]} ...")


# =====================
# Plotting
# =====================
def save_history_and_plots(log_dir: str, history: dict):
    os.makedirs(log_dir, exist_ok=True)

    np.save(os.path.join(log_dir, "train_step.npy"), np.array(history["train_step"], dtype=np.int64))
    np.save(os.path.join(log_dir, "train_step_loss.npy"), np.array(history["train_step_loss"], dtype=np.float32))
    np.save(os.path.join(log_dir, "train_epoch_loss.npy"), np.array(history["train_epoch_loss"], dtype=np.float32))
    np.save(os.path.join(log_dir, "val_epoch_loss.npy"), np.array(history["val_epoch_loss"], dtype=np.float32))
    np.save(os.path.join(log_dir, "val_epoch_bleu.npy"), np.array(history["val_epoch_bleu"], dtype=np.float32))
    np.save(os.path.join(log_dir, "val_epoch_codebleu.npy"), np.array(history["val_epoch_codebleu"], dtype=np.float32))
    np.save(os.path.join(log_dir, "val_epoch_em.npy"), np.array(history["val_epoch_em"], dtype=np.float32))
    np.save(os.path.join(log_dir, "val_epoch_diff.npy"), np.array(history["val_epoch_diff"], dtype=np.float32))

    if len(history["train_step_loss"]) > 0:
        plt.figure()
        plt.plot(history["train_step"], history["train_step_loss"])
        plt.xlabel("Step")
        plt.ylabel("Train Loss")
        plt.title("Train Loss (per step)")
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "train_loss_steps.png"), dpi=200)
        plt.close()

    if len(history["train_epoch_loss"]) > 0:
        epochs = list(range(1, len(history["train_epoch_loss"]) + 1))
        plt.figure()
        plt.plot(epochs, history["train_epoch_loss"], marker="o", label="train")
        plt.plot(epochs, history["val_epoch_loss"], marker="o", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train/Val Loss (per epoch)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "loss_epochs.png"), dpi=200)
        plt.close()

    if len(history["val_epoch_bleu"]) > 0:
        epochs = list(range(1, len(history["val_epoch_bleu"]) + 1))
        plt.figure()
        plt.plot(epochs, history["val_epoch_bleu"], marker="o", label="BLEU")
        plt.plot(epochs, history["val_epoch_codebleu"], marker="o", label="CodeBLEU")
        plt.plot(epochs, history["val_epoch_em"], marker="o", label="EM (%)")
        plt.plot(epochs, history["val_epoch_diff"], marker="o", label="DiffMatch (%)")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Metrics (capped)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "val_metrics_epochs.png"), dpi=200)
        plt.close()


# =====================
# Eval (fast)
# =====================
def evaluate_fast(
    model: SoftPromptTokenRepair,
    loader,
    tokenizer,
    limit: int = 1000,
    print_k: int = 0,
):
    model.eval()
    total_loss = 0.0
    all_preds, all_gts, all_langs = [], [], []
    exact_matches = 0
    total_eval = 0
    diff_scores = []
    seen = 0

    printed = False
    val_pbar = tqdm(loader, desc="Validating", leave=False)

    with torch.no_grad():
        for buggy_ids, buggy_mask, fixed_ids, fixed_mask, langs in val_pbar:
            buggy_ids = buggy_ids.to(DEVICE, non_blocking=True)
            buggy_mask = buggy_mask.to(DEVICE, non_blocking=True)
            fixed_ids = fixed_ids.to(DEVICE, non_blocking=True)
            fixed_mask = fixed_mask.to(DEVICE, non_blocking=True)

            loss = model(buggy_ids, buggy_mask, fixed_ids, fixed_mask)
            if not torch.isnan(loss):
                total_loss += loss.item()

            gen_ids = model.generate_fast_ids(buggy_ids, buggy_mask, max_new_tokens=MAX_NEW_TOKENS)

            gt_lens = fixed_mask.sum(dim=1).long()
            preds_texts: List[str] = []
            gts_texts = tokenizer.batch_decode(fixed_ids, skip_special_tokens=True)
            buggy_texts = tokenizer.batch_decode(buggy_ids, skip_special_tokens=True)

            for i in range(gen_ids.size(0)):
                max_keep = int(min(gen_ids.size(1), gt_lens[i].item() + EXTRA_TOKENS_OVER_GT))
                ids_i = gen_ids[i, :max_keep]
                pred_i = tokenizer.decode(ids_i, skip_special_tokens=True).strip()
                preds_texts.append(pred_i)

            if (not printed) and print_k > 0:
                k = min(print_k, len(preds_texts))
                print("\n" + "#" * 80)
                print(f"[Sample Preview] Showing {k} samples")
                print("#" * 80)
                for i in range(k):
                    b = (buggy_texts[i] or "").strip().replace("\n", "\\n")
                    g = (gts_texts[i] or "").strip().replace("\n", "\\n")
                    p = (preds_texts[i] or "").strip().replace("\n", "\\n")
                    print(f"\n--- Sample {i+1} ---")
                    print(f"[BUGGY] {b[:300]}")
                    print(f"[GT   ] {g[:300]}")
                    print(f"[PRED ] {p[:300]}")
                print("#" * 80 + "\n")
                printed = True

            all_preds.extend(preds_texts)
            all_gts.extend(gts_texts)
            all_langs.extend(langs)

            for b, p, g in zip(buggy_texts, preds_texts, gts_texts):
                if p == g:
                    exact_matches += 1
                total_eval += 1
                diff_scores.append(diff_match_score(b, g, p))

            seen += buggy_ids.size(0)
            val_pbar.set_postfix(count=f"{seen}/{limit}")
            if seen >= limit:
                break

    avg_loss = total_loss / max(len(all_preds) / max(loader.batch_size, 1), 1)

    try:
        bleu = corpus_bleu(all_preds, [all_gts]).score
    except Exception:
        bleu = 0.0

    codebleu = 0.0
    if DO_CODEBLEU and HAS_CODEBLEU:
        lang_groups: Dict[str, Dict[str, List[str]]] = {}
        for p, g, l in zip(all_preds, all_gts, all_langs):
            ll = str(l).lower()
            lang_groups.setdefault(ll, {"p": [], "r": []})
            lang_groups[ll]["p"].append(p)
            lang_groups[ll]["r"].append(g)

        total_weighted, total_n = 0.0, 0
        for lang, data in lang_groups.items():
            n = len(data["p"])
            target_lang = NORMALIZE_LANG_MAP.get(lang)
            if target_lang:
                try:
                    res = calc_codebleu(data["r"], data["p"], lang=target_lang)
                    score = res["codebleu"] * 100
                except Exception:
                    score = corpus_bleu(data["p"], [data["r"]]).score
            else:
                score = corpus_bleu(data["p"], [data["r"]]).score

            total_weighted += score * n
            total_n += n

        codebleu = (total_weighted / total_n) if total_n > 0 else 0.0

    em_rate = (exact_matches / total_eval * 100.0) if total_eval > 0 else 0.0
    diff_match = (float(np.mean(diff_scores)) * 100.0) if len(diff_scores) > 0 else 0.0

    return avg_loss, bleu, codebleu, em_rate, diff_match


# =====================
# Test + Save JSONL
# =====================
@torch.no_grad()
def run_test_and_save(model: SoftPromptTokenRepair, loader, tokenizer, save_jsonl_path: str):
    os.makedirs(os.path.dirname(save_jsonl_path), exist_ok=True)
    f_out = open(save_jsonl_path, "w", encoding="utf-8")
    print(f"[Info] Saving JSONL -> {save_jsonl_path}")

    all_preds, all_gts, all_langs = [], [], []
    em_cnt = 0
    diff_scores = []

    pbar = tqdm(loader, desc="Testing", leave=True)
    for (
        buggy_ids, buggy_mask, fixed_ids, fixed_mask,
        buggy_text, gt_fixed_text, lang,
        gidx, pid, buggy_sid, fixed_sid
    ) in pbar:
        buggy_ids = buggy_ids.to(DEVICE, non_blocking=True)
        buggy_mask = buggy_mask.to(DEVICE, non_blocking=True)
        fixed_ids = fixed_ids.to(DEVICE, non_blocking=True)
        fixed_mask = fixed_mask.to(DEVICE, non_blocking=True)

        gen_ids = model.generate_fast_ids(buggy_ids, buggy_mask, max_new_tokens=MAX_NEW_TOKENS)
        gt_lens = fixed_mask.sum(dim=1).long()

        preds_texts: List[str] = []
        for i in range(gen_ids.size(0)):
            max_keep = int(min(gen_ids.size(1), gt_lens[i].item() + EXTRA_TOKENS_OVER_GT))
            ids_i = gen_ids[i, :max_keep]
            pred_i = tokenizer.decode(ids_i, skip_special_tokens=True).strip()
            preds_texts.append(pred_i)

        gts_texts = tokenizer.batch_decode(fixed_ids, skip_special_tokens=True)

        for i in range(len(preds_texts)):
            rec = {
                "global_index": int(gidx[i]),
                "problem_id": str(pid[i]),
                "buggy_submission_id": int(buggy_sid[i]),
                "fixed_submission_id": int(fixed_sid[i]),
                "language": str(lang[i]),
                "preds": [preds_texts[i]],
                "gt_fixed_code": str(gt_fixed_text[i]),
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

        for p, g, b in zip(preds_texts, gts_texts, buggy_text):
            if p == g:
                em_cnt += 1
            diff_scores.append(diff_match_score(str(b), str(g), str(p)))

        all_preds.extend(preds_texts)
        all_gts.extend(gts_texts)
        all_langs.extend([str(x) for x in lang])

    f_out.close()

    try:
        bleu = corpus_bleu(all_preds, [all_gts]).score
    except Exception:
        bleu = 0.0

    em = em_cnt / max(len(all_preds), 1) * 100.0
    diffm = float(np.mean(diff_scores)) * 100.0 if diff_scores else 0.0

    print("\n====================")
    print(f"[TEST] BLEU      : {bleu:.2f}")
    print(f"[TEST] EM        : {em:.2f}%")
    print(f"[TEST] DiffMatch : {diffm:.2f}%")

    if DO_CODEBLEU and HAS_CODEBLEU:
        lang_groups: Dict[str, Dict[str, List[str]]] = {}
        for p, g, l in zip(all_preds, all_gts, all_langs):
            ll = (l or "").lower()
            lang_groups.setdefault(ll, {"p": [], "r": []})
            lang_groups[ll]["p"].append(p)
            lang_groups[ll]["r"].append(g)

        total_weighted, total_cnt = 0.0, 0
        print(f"\n{'Language':<12} | {'Count':<6} | {'CodeBLEU':<8}")
        print("-" * 36)
        for lang_name in sorted(lang_groups.keys()):
            data = lang_groups[lang_name]
            cnt = len(data["p"])
            target_lang = NORMALIZE_LANG_MAP.get(lang_name)

            if target_lang:
                try:
                    res = calc_codebleu(data["r"], data["p"], lang=target_lang)
                    score = res["codebleu"] * 100
                except Exception:
                    score = corpus_bleu(data["p"], [data["r"]]).score
            else:
                score = corpus_bleu(data["p"], [data["r"]]).score

            total_weighted += score * cnt
            total_cnt += cnt
            print(f"{lang_name:<12} | {cnt:<6} | {score:>8.2f}")

        avg_codebleu = total_weighted / max(total_cnt, 1)
        print(f"\n[TEST] Avg CodeBLEU (weighted): {avg_codebleu:.2f}")
    elif DO_CODEBLEU and not HAS_CODEBLEU:
        print("[Info] CodeBLEU not available -> skipped.")

    print("====================\n")


# =====================
# Train / Test entrypoints
# =====================
def main_train():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file_path = os.path.join(LOG_DIR, f"train_log_{timestamp}.txt")
    sys.stdout = Logger(log_file_path)

    print(f"=== Token Baseline (Soft Prompt) Training Started at {timestamp} ===")
    print(f"Logging to: {log_file_path}")
    print(
        f"Config: epochs={NUM_EPOCHS}, batch={BATCH_SIZE_TRAIN}, prompt_len={PROMPT_LEN}, "
        f"max_buggy={MAX_LEN_BUGGY}, max_fixed={MAX_LEN_FIXED}, val_limit={VAL_CHECK_LIMIT}"
    )
    print(f"GEN: max_new_tokens={MAX_NEW_TOKENS}, extra_over_gt={EXTRA_TOKENS_OVER_GT}")
    print(f"CodeBLEU available: {HAS_CODEBLEU}")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    global_target_indices, train_idx, val_idx, test_idx = load_saved_indices(SAVED_INDICES_DIR)
    print(
        f"Loaded saved indices. Subset size={len(global_target_indices)} | "
        f"Train={len(train_idx)} Val={len(val_idx)} Test={len(test_idx)}"
    )

    buggy_texts, fixed_texts, languages = load_subset_by_global_indices(global_target_indices)

    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading StarCoder2-3B in bfloat16 (frozen)...")
    decoder = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(DEVICE)

    decoder.gradient_checkpointing_enable()
    decoder.enable_input_require_grads()

    print("Tokenizing BUGGY texts...")
    enc_buggy = tokenizer(
        buggy_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN_BUGGY,
        return_tensors="pt"
    )
    print("Tokenizing FIXED texts...")
    enc_fixed = tokenizer(
        fixed_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN_FIXED,
        return_tensors="pt"
    )

    train_langs = [languages[i] for i in train_idx]
    val_langs = [languages[i] for i in val_idx]

    train_ds = TokenRepairDataset(
        enc_buggy.input_ids[train_idx], enc_buggy.attention_mask[train_idx],
        enc_fixed.input_ids[train_idx], enc_fixed.attention_mask[train_idx],
        train_langs
    )
    val_ds = TokenRepairDataset(
        enc_buggy.input_ids[val_idx], enc_buggy.attention_mask[val_idx],
        enc_fixed.input_ids[val_idx], enc_fixed.attention_mask[val_idx],
        val_langs
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    model = SoftPromptTokenRepair(decoder, tokenizer, prompt_len=PROMPT_LEN).to(DEVICE)
    optimizer = torch.optim.AdamW([model.soft_prompt], lr=LR, weight_decay=WEIGHT_DECAY)

    history = {
        "train_step": [],
        "train_step_loss": [],
        "train_epoch_loss": [],
        "val_epoch_loss": [],
        "val_epoch_bleu": [],
        "val_epoch_codebleu": [],
        "val_epoch_em": [],
        "val_epoch_diff": [],
    }

    best_val_loss = float("inf")
    best_val_diff = -1.0
    global_step = 0

    print("Starting training loop...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        step_count = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=True)
        for buggy_ids, buggy_mask, fixed_ids, fixed_mask, _ in pbar:
            buggy_ids = buggy_ids.to(DEVICE, non_blocking=True)
            buggy_mask = buggy_mask.to(DEVICE, non_blocking=True)
            fixed_ids = fixed_ids.to(DEVICE, non_blocking=True)
            fixed_mask = fixed_mask.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            loss = model(buggy_ids, buggy_mask, fixed_ids, fixed_mask)
            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_([model.soft_prompt], GRAD_CLIP)
            optimizer.step()

            lval = float(loss.item())
            total_train_loss += lval
            step_count += 1
            global_step += 1

            history["train_step"].append(global_step)
            history["train_step_loss"].append(lval)

            pbar.set_postfix(loss=f"{lval:.4f}")

            if step_count % LOG_EVERY_STEPS == 0:
                avg = total_train_loss / max(step_count, 1)
                print(f"[Train] epoch={epoch} step={step_count} global_step={global_step} avg_loss={avg:.4f}")

        avg_train_loss = total_train_loss / max(step_count, 1)
        epoch_time = (time.time() - start_time) / 3600.0
        print(f"Epoch {epoch} Done ({epoch_time:.2f} hrs) | Train Avg Loss: {avg_train_loss:.6f}")

        print_k = 5 if epoch == 1 else 0
        val_loss, val_bleu, val_codebleu, val_em, val_diff = evaluate_fast(
            model, val_loader, tokenizer, limit=VAL_CHECK_LIMIT, print_k=print_k
        )

        print(
            f"\n[Epoch {epoch}] train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | BLEU={val_bleu:.2f} | CodeBLEU={val_codebleu:.2f} | "
            f"EM={val_em:.2f}% | DiffMatch={val_diff:.2f}%\n"
        )

        history["train_epoch_loss"].append(avg_train_loss)
        history["val_epoch_loss"].append(val_loss)
        history["val_epoch_bleu"].append(val_bleu)
        history["val_epoch_codebleu"].append(val_codebleu)
        history["val_epoch_em"].append(val_em)
        history["val_epoch_diff"].append(val_diff)

        save_history_and_plots(LOG_DIR, history)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  >>> [Best Val Loss] Updated (Loss: {val_loss:.4f})")

        if val_diff > best_val_diff:
            best_val_diff = val_diff
            print(f"  >>> [Best DiffMatch] Updated (DiffMatch: {val_diff:.2f}%)")

        if epoch % SAVE_EVERY_EPOCH == 0:
            save_tunable_parameters(model, os.path.join(CKPT_DIR, f"checkpoint_epoch_{epoch}.pt"))

    print("\nTraining Finished.")
    print(f"Best Val Loss: {best_val_loss:.4f} | Best Val DiffMatch: {best_val_diff:.2f}%")
    print(f"Curves saved under: {LOG_DIR}")


def main_test(ckpt_path: str, save_jsonl_path: str):
    assert os.path.exists(SAVED_INDICES_DIR), f"indices dir not found: {SAVED_INDICES_DIR}"
    assert os.path.exists(ckpt_path), f"ckpt not found: {ckpt_path}"

    (
        buggy_texts, fixed_texts, languages,
        global_test_indices, problem_ids, buggy_sids, fixed_sids
    ) = load_test_subset_from_saved_indices(SAVED_INDICES_DIR)

    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[Info] Tokenizing BUGGY...")
    enc_buggy = tokenizer(
        buggy_texts, padding=True, truncation=True, max_length=MAX_LEN_BUGGY, return_tensors="pt"
    )
    print("[Info] Tokenizing FIXED...")
    enc_fixed = tokenizer(
        fixed_texts, padding=True, truncation=True, max_length=MAX_LEN_FIXED, return_tensors="pt"
    )

    test_ds = TokenRepairTestDataset(
        buggy_ids=enc_buggy.input_ids,
        buggy_mask=enc_buggy.attention_mask,
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
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print("[Info] Loading StarCoder2-3B (bfloat16, frozen)...")
    decoder = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    model = SoftPromptTokenRepair(decoder, tokenizer, prompt_len=PROMPT_LEN).to(DEVICE)
    load_soft_prompt_checkpoint(model, ckpt_path)

    t0 = time.time()
    run_test_and_save(model, test_loader, tokenizer, save_jsonl_path)
    print(f"[Done] Elapsed: {(time.time()-t0)/60:.1f} min")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--ckpt", type=str, default=os.path.join(CKPT_DIR, "checkpoint_epoch_2.pt"))
    parser.add_argument("--save_jsonl", type=str, default=SAVE_JSONL_TEST)
    args = parser.parse_args()

    if args.mode == "train":
        main_train()
    else:
        main_test(args.ckpt, args.save_jsonl)


if __name__ == "__main__":
    main()
