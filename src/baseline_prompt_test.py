# baseline_prompt_test.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prompt baseline (no training):
prompt + buggy_code -> generate fixed_code
"""

import os
import json
import time
import difflib
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from sacrebleu import corpus_bleu
import re
from typing import List, Optional, Tuple


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
INDICES_DIR = "saved_indices"

MAX_LEN_FIXED = 1024
MAX_LEN_PROMPT_INPUT = 1280

MAX_NEW_TOKENS = 4096

BATCH_SIZE = 16
NUM_WORKERS = 4

N_SAMPLES = 1
DO_SAMPLE = True
TEMPERATURE = 0.2
TOP_P = 0.95

DO_CODEBLEU = True

SAVE_DIR = "results_prompt_baseline_addtokens"
SAVE_JSONL = os.path.join(SAVE_DIR, "fixed_code_test.jsonl")

DEBUG_PRINT_FIRST_BATCH = True

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


# =====================
# Try importing shared data loaders
# =====================
_SHARED_OK = True
try:
    # Example: adjust these imports to match your existing project structure
    # from data.runbugrun import load_test_subset_from_saved_indices  # noqa: F401
    raise ImportError("Shared utilities import placeholders not configured.")
except Exception:
    _SHARED_OK = False


# =====================
# Data loading (fallback)
# =====================
if not _SHARED_OK:
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
# Stopping criteria: stop when "<END>" appears
# =====================
class StopOnSubsequenceAll(StoppingCriteria):
    def __init__(self, stop_ids: List[int]):
        super().__init__()
        self.stop_ids = stop_ids
        self._seen = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        L = len(self.stop_ids)
        if L == 0:
            return False

        bsz = input_ids.size(0)
        if self._seen is None or self._seen.numel() != bsz:
            self._seen = torch.zeros(bsz, dtype=torch.bool, device=input_ids.device)

        if input_ids.size(1) >= L:
            tail = input_ids[:, -L:]
            stop = torch.tensor(self.stop_ids, device=input_ids.device).unsqueeze(0)
            match = (tail == stop).all(dim=1)
            self._seen |= match

        return bool(self._seen.all().item())


def make_end_stopping(tokenizer) -> StoppingCriteriaList:
    stop_ids = tokenizer.encode("<END>", add_special_tokens=False)
    return StoppingCriteriaList([StopOnSubsequenceAll(stop_ids)])


# =====================
# Diff-match helpers
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
# Prompt baseline
# =====================
def build_prompt(buggy_code: str, language: str) -> str:
    buggy_code = (buggy_code or "").rstrip()
    language = (language or "").strip()
    return (
        f"Your task is to fix the {language} code.\n"
        "Here is the buggy program:\n"
        "```\n"
        f"{buggy_code}\n"
        "```\n\n"
        "Please return the fixed code delimited by a Markdown code block (triple backticks).\n\n"
        "Here is the fixed program:\n"
        "```\n"
    )


def clean_pred_text(pred: str) -> str:
    if pred is None:
        return ""
    s = pred.strip()

    if "<END>" in s:
        s = s.split("<END>", 1)[0].strip()

    if "```" in s:
        if s.lstrip().startswith("```"):
            parts = s.split("```", 2)
            if len(parts) >= 2:
                s2 = parts[1].lstrip()
                first_nl = s2.find("\n")
                if first_nl != -1:
                    first_line = s2[:first_nl].strip()
                    if len(first_line) <= 20 and all((c.isalpha() or c in "#+-_") for c in first_line):
                        s2 = s2[first_nl + 1:]
                s = s2.strip()
        else:
            s = s.split("```", 1)[0].strip()

    for marker in ["<issue_comment>", "<issue_start>", "\nassistant:", "\nuser:"]:
        if marker in s:
            s = s.split(marker, 1)[0].strip()

    return s


SPECIAL_TOKENS_RE = re.compile(r"<\|endoftext\|>|\[PAD\]|<\|pad\|>", re.IGNORECASE)

STOP_MARKERS: List[str] = [
    "<END>", "</END>",
]

def _strip_special(s: str) -> str:
    if s is None:
        return ""
    return SPECIAL_TOKENS_RE.sub("", str(s)).lstrip()

def _find_earliest(hay: str, needles: List[str]) -> Optional[int]:
    best = None
    for n in needles:
        i = hay.find(n)
        if i != -1 and (best is None or i < best):
            best = i
    return best

def _extract_first_fenced_block(s: str) -> str:
    parts = s.split("\n", 1)
    inner = parts[1] if len(parts) == 2 else ""
    end = inner.find("```")
    if end != -1:
        return inner[:end]
    return inner

def extract_code(text: str) -> Tuple[str, str]:
    s = _strip_special(text)
    if not s:
        return "", "empty"

    # 1) continuation starts with ``` -> extract inside first block
    if s.startswith("```"):
        code = _extract_first_fenced_block(s)
        return code.strip(), "fence_wrapped"

    # 2) prompt already opened ```; first ``` is closing fence
    i = s.find("```")
    if i != -1:
        return s[:i].strip(), "fence_close"

    # 3) triple quotes
    j = s.find("'''")
    if j != -1:
        return s[:j].strip(), "triplequote_close"

    # 4) stop markers
    cut = _find_earliest(s, STOP_MARKERS)
    if cut is not None:
        return s[:cut].strip(), "stop_marker"

    # 5) whole
    return s.strip(), "whole"


@torch.no_grad()
def generate_preds_list_batch(decoder, tokenizer, buggy_texts: List[str], languages: List[str], debug: bool = False):
    B = len(buggy_texts)
    prompts = [build_prompt(b, l) for b, l in zip(buggy_texts, languages)]

    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN_PROMPT_INPUT,
        return_tensors="pt"
    ).to(DEVICE)

    prompt_lens = enc.attention_mask.sum(dim=1).long()
    input_len = enc.input_ids.shape[1]


    gen = decoder.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=64,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE if DO_SAMPLE else None,
        top_p=TOP_P if DO_SAMPLE else None,
        num_return_sequences=N_SAMPLES,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if N_SAMPLES > 1:
        gen = gen.view(B, N_SAMPLES, -1)
    else:
        gen = gen.unsqueeze(1)

    if debug:
        print("[DEBUG] new_tokens_len (first 5 samples):")
        for i in range(min(B, 5)):
            cut = int(input_len)
            new_len = int(gen[i, 0, cut:].numel())
            print(f"  sample{i}: {new_len}")

    all_preds: List[List[str]] = []
    all_full_texts: List[List[str]] = []
    all_continuation_texts: List[List[str]] = []

    for i in range(B):
        cut = int(input_len)
        preds_i: List[str] = []
        full_i: List[str] = []
        cont_i: List[str] = []

        for k in range(N_SAMPLES):
            new_ids = gen[i, k, cut:]
            cont_txt = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            full_txt = prompts[i] + cont_txt
            pred = extract_code(cont_txt)
            preds_i.append(pred.strip())
            full_i.append(full_txt)
            cont_i.append(cont_txt)
        all_preds.append(preds_i)
        all_full_texts.append(full_i)
        all_continuation_texts.append(cont_i)

    if debug and B > 0:
        print("\n========== [DEBUG FIRST BATCH] ==========")
        print("[DEBUG] buggy_code[:300]:")
        print((buggy_texts[0] or "")[:300])
        print("\n[DEBUG] prompt[:800]:")
        print(prompts[0][:800])
        print(f"\n[DEBUG] prompt_len[0]={int(prompt_lens[0].item())}  input_len(padded)={int(input_len)}  B={B}  N_SAMPLES={N_SAMPLES}")
        raw0 = tokenizer.decode(gen[0, 0, int(input_len):], skip_special_tokens=True)
        print("\n[DEBUG] raw_pred0 (candidate0)[:800]:")
        print((raw0 or "")[:800])
        print("\n[DEBUG] cleaned_pred0[:800]:")
        print((all_preds[0][0] or "")[:800])
        print("=========================================\n")

    return all_preds, all_full_texts, all_continuation_texts


@torch.no_grad()
def run_test_and_save(decoder, loader, tokenizer, save_jsonl_path: str):
    os.makedirs(os.path.dirname(save_jsonl_path), exist_ok=True)
    f_out = open(save_jsonl_path, "w", encoding="utf-8")
    print(f"[Info] Saving JSONL -> {save_jsonl_path}")

    all_preds, all_gts, all_langs = [], [], []
    em_cnt = 0
    diff_scores = []
    printed_debug = False

    pbar = tqdm(loader, desc="Testing(prompt baseline)", leave=True)
    for (
        fixed_ids, fixed_mask,
        buggy_text, gt_fixed_text, lang,
        gidx, pid, buggy_sid, fixed_sid
    ) in pbar:
        fixed_ids = fixed_ids.to(DEVICE, non_blocking=True)
        fixed_mask = fixed_mask.to(DEVICE, non_blocking=True)

        debug_now = DEBUG_PRINT_FIRST_BATCH and (not printed_debug)
        preds_list, full_texts, continuation_texts = generate_preds_list_batch(
            decoder, tokenizer,
            buggy_texts=list(buggy_text),
            languages=list(lang),
            debug=debug_now
        )
        if debug_now:
            printed_debug = True

        preds_texts = [
            (cands[0].strip() if (isinstance(cands, list) and len(cands) > 0 and isinstance(cands[0], str)) else "")
            for cands in preds_list
        ]

        gts_texts = tokenizer.batch_decode(fixed_ids, skip_special_tokens=True)

        B = len(preds_texts)
        for i in range(B):
            cands = preds_list[i] if i < len(preds_list) else []
            if not isinstance(cands, list):
                cands = [str(cands)]
            cands = [c for c in cands if isinstance(c, str)]

            rec = {
                "global_index": int(gidx[i]),
                "problem_id": str(pid[i]),
                "buggy_submission_id": int(buggy_sid[i]),
                "fixed_submission_id": int(fixed_sid[i]),
                "language": str(lang[i]),
                "preds": cands,
                "full_texts": full_texts[i],
                "continuation_texts": continuation_texts[i],
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


def main():
    assert os.path.exists(INDICES_DIR), f"indices dir not found: {INDICES_DIR}"

    (
        buggy_texts, fixed_texts, languages,
        global_test_indices, problem_ids, buggy_sids, fixed_sids
    ) = load_test_subset_from_saved_indices(INDICES_DIR)

    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL_ID)
    tokenizer.padding_side = "left"
    if (tokenizer.pad_token is None) or (tokenizer.pad_token_id == tokenizer.eos_token_id):
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    print(f"[Tokenizer] pad_id={tokenizer.pad_token_id} eos_id={tokenizer.eos_token_id} (pad!=eos expected)")
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

    print("[Info] Loading StarCoder2-3B (bfloat16, frozen)...")
    decoder = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    decoder.resize_token_embeddings(len(tokenizer))
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    os.makedirs(SAVE_DIR, exist_ok=True)
    t0 = time.time()
    run_test_and_save(decoder, test_loader, tokenizer, SAVE_JSONL)
    print(f"[Done] Elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
