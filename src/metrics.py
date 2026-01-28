# metrics.py
# -*- coding: utf-8 -*-

import os
import json
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from sacrebleu import corpus_bleu

try:
    from codebleu import calc_codebleu
    HAS_CODEBLEU = True
except Exception:
    HAS_CODEBLEU = False


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


def evaluate_performance_fast(model, loader, tokenizer, epoch: int, limit: int = 1000, device: str = "cuda"):
    model.eval()
    total_val_loss = 0
    all_preds: List[str] = []
    all_gts: List[str] = []
    all_langs: List[str] = []

    count = 0
    print(f"\nRunning Validation (Capped at {limit} samples)...")

    val_pbar = tqdm(loader, desc="Validating", leave=False)

    with torch.no_grad():
        for b_emb, b_ids, b_mask, b_langs in val_pbar:
            b_emb = b_emb.to(device)
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            loss = model(b_emb, b_ids, b_mask)
            if not torch.isnan(loss):
                total_val_loss += loss.item()

            preds = model.generate_fast(b_emb)
            gts = tokenizer.batch_decode(b_ids, skip_special_tokens=True)

            all_preds.extend(preds)
            all_gts.extend(gts)
            all_langs.extend(list(b_langs))

            count += len(b_emb)
            val_pbar.set_postfix(count=f"{count}/{limit}")

            if count >= limit:
                break

    avg_val_loss = total_val_loss / max(len(loader) * (count / len(loader.dataset)), 1)

    try:
        g_bleu = corpus_bleu(all_preds, [all_gts]).score
    except Exception:
        g_bleu = 0.0

    print("-" * 70)
    print(f"[Validation] Epoch {epoch} | Est. Loss: {avg_val_loss:.4f} | Text BLEU: {g_bleu:.2f}")

    lang_groups: Dict[str, Dict[str, List[str]]] = {}
    for p, g, l in zip(all_preds, all_gts, all_langs):
        l_lower = str(l).lower()
        if l_lower not in lang_groups:
            lang_groups[l_lower] = {"p": [], "r": []}
        lang_groups[l_lower]["p"].append(p)
        lang_groups[l_lower]["r"].append(g)

    total_weighted_codebleu = 0.0
    total_valid_samples = 0

    print(f"{'Language':<12} | {'Count':<5} | {'CodeBLEU':<8}")

    for lang in sorted(lang_groups.keys()):
        data = lang_groups[lang]
        count_lang = len(data["p"])
        target_lang = NORMALIZE_LANG_MAP.get(lang)
        score = 0.0

        if target_lang:
            try:
                res = calc_codebleu(data["r"], data["p"], lang=target_lang)
                score = res["codebleu"] * 100
            except Exception:
                score = corpus_bleu(data["p"], [data["r"]]).score
        else:
            score = corpus_bleu(data["p"], [data["r"]]).score

        if score > 0:
            total_weighted_codebleu += score * count_lang
            total_valid_samples += count_lang

        print(f"{lang:<12} | {count_lang:<5} | {score:>8.2f}")

    avg_codebleu = 0.0
    if total_valid_samples > 0:
        avg_codebleu = total_weighted_codebleu / total_valid_samples

    print(f"AVERAGE      | {total_valid_samples:<5} | {avg_codebleu:>8.2f}")
    print("=" * 70 + "\n")

    return avg_val_loss, avg_codebleu


@torch.no_grad()
def evaluate_and_save(
    model,
    loader,
    tokenizer,
    device: str,
    max_new_tokens: int = 128,
    do_codebleu: bool = True,
    save_jsonl_path: str | None = None,
):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds: List[str] = []
    all_gts: List[str] = []
    all_langs: List[str] = []

    f_out = None
    if save_jsonl_path is not None:
        os.makedirs(os.path.dirname(save_jsonl_path), exist_ok=True)
        f_out = open(save_jsonl_path, "w", encoding="utf-8")
        print(f"[Info] Saving generated code JSONL -> {save_jsonl_path}")

    pbar = tqdm(loader, desc="Decoding(test)", leave=True)
    for (
        b_emb, b_ids, b_mask, b_langs,
        b_gidx, b_pid, b_buggy_sid, b_fixed_sid, b_gt_fixed_code
    ) in pbar:
        b_emb = b_emb.to(device)
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        loss = model(b_emb, b_ids, b_mask)
        if not torch.isnan(loss):
            total_loss += loss.item()
            n_batches += 1

        preds = model.generate_fast(b_emb, max_new_tokens=max_new_tokens)
        gts = tokenizer.batch_decode(b_ids, skip_special_tokens=True)

        if f_out is not None:
            if torch.is_tensor(b_gidx):
                b_gidx = b_gidx.tolist()
            if torch.is_tensor(b_pid):
                b_pid = b_pid.tolist()
            if torch.is_tensor(b_buggy_sid):
                b_buggy_sid = b_buggy_sid.tolist()
            if torch.is_tensor(b_fixed_sid):
                b_fixed_sid = b_fixed_sid.tolist()

            for i in range(len(preds)):
                record = {
                    "global_index": int(b_gidx[i]),
                    "problem_id": str(b_pid[i]),
                    "buggy_submission_id": int(b_buggy_sid[i]),
                    "fixed_submission_id": int(b_fixed_sid[i]),
                    "language": str(b_langs[i]),
                    "preds": [preds[i]],
                    "gt_fixed_code": str(b_gt_fixed_code[i]),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        all_preds.extend(preds)
        all_gts.extend(gts)
        all_langs.extend([str(x) for x in b_langs])

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    if f_out is not None:
        f_out.close()

    avg_loss = total_loss / max(n_batches, 1)
    try:
        text_bleu = corpus_bleu(all_preds, [all_gts]).score
    except Exception:
        text_bleu = 0.0

    print("\n====================")
    print(f"[TEST] Avg Loss : {avg_loss:.4f}")
    print(f"[TEST] Text BLEU: {text_bleu:.2f}")

    if do_codebleu:
        if not HAS_CODEBLEU:
            print("[Info] codebleu not available in env -> skip CodeBLEU.")
        else:
            lang_groups: Dict[str, Dict[str, List[str]]] = {}
            for p, g, l in zip(all_preds, all_gts, all_langs):
                ll = l.lower()
                lang_groups.setdefault(ll, {"p": [], "r": []})
                lang_groups[ll]["p"].append(p)
                lang_groups[ll]["r"].append(g)

            total_weighted = 0.0
            total_cnt = 0

            print(f"\n{'Language':<12} | {'Count':<6} | {'CodeBLEU':<8}")
            print("-" * 36)

            for lang in sorted(lang_groups.keys()):
                data = lang_groups[lang]
                cnt = len(data["p"])
                target_lang = NORMALIZE_LANG_MAP.get(lang)

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
                print(f"{lang:<12} | {cnt:<6} | {score:>8.2f}")

            avg_codebleu = total_weighted / max(total_cnt, 1)
            print(f"\n[TEST] Avg CodeBLEU (weighted): {avg_codebleu:.2f}")

    print("====================\n")
    return avg_loss
