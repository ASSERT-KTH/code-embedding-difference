#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse each candidate from continuation_texts
"""

import argparse
import json
import re
from typing import Dict, List, Optional, Tuple


SPECIAL_TOKENS_RE = re.compile(r"<\|endoftext\|>|\[PAD\]|<\|pad\|>", re.IGNORECASE)

STOP_MARKERS: List[str] = [
    # thread-ish / meta (if you ever used issue-style elsewhere)
    "<issue_comment>", "Upvotes:", "<issue_start>",
    "username_", "OP:", "Maintainer:", "Question:", "Answer:", "User:", "Assistant:",
    # explicit ends / explanations
    "<END>", "</END>",
    "Explanation:", "Reasoning:", "Analysis:",
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
    """
    s is assumed to start with ``` (maybe ```lang).
    Returns the content inside the FIRST fenced block.
    """
    # Drop opening fence line (``` or ```lang)
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

    # 1) If it starts with ``` then it's a self-contained fenced block
    if s.startswith("```"):
        code = _extract_first_fenced_block(s)
        return code.strip(), "fence_wrapped"

    # 2) Otherwise, prompt already opened ```; first ``` is usually the closing fence.
    i = s.find("```")
    if i != -1:
        return s[:i].strip(), "fence_close"

    # 3) Triple quotes fallback
    j = s.find("'''")
    if j != -1:
        return s[:j].strip(), "triplequote_close"

    # 4) Stop markers
    cut = _find_earliest(s, STOP_MARKERS)
    if cut is not None:
        return s[:cut].strip(), "stop_marker"

    # 5) Whole fallback
    return s.strip(), "whole"


def parse_record(rec: Dict, source_field: str) -> Dict:
    outs = rec.get(source_field)

    # Fallback to the other field if missing
    if not (isinstance(outs, list) and len(outs) > 0):
        other = "full_texts" if source_field == "continuation_texts" else "continuation_texts"
        outs = rec.get(other)
        source_field = other

    preds: List[str] = []
    methods: List[str] = []

    if isinstance(outs, list) and len(outs) > 0:
        for t in outs:
            code, m = extract_code(t)
            preds.append(code)
            methods.append(f"{m}@{source_field}")
    else:
        preds = [""]
        methods = ["missing"]

    rec["preds"] = preds
    rec["pred_parse_method"] = methods
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument(
        "--source_field",
        choices=["continuation_texts", "full_texts"],
        default="continuation_texts",
        help="Which field to parse from (default: continuation_texts).",
    )
    ap.add_argument("--max_lines", type=int, default=-1, help="Process at most N lines (default: all).")
    args = ap.parse_args()

    n = 0
    empty_first = 0
    method_counts: Dict[str, int] = {}

    with open(args.in_jsonl, "r", encoding="utf-8") as f_in, open(args.out_jsonl, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if args.max_lines > 0 and n >= args.max_lines:
                break
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            rec = parse_record(rec, args.source_field)

            for m in rec.get("pred_parse_method", []):
                method_counts[m] = method_counts.get(m, 0) + 1

            preds = rec.get("preds", [])
            if isinstance(preds, list) and preds and preds[0].strip() == "":
                empty_first += 1

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"[Done] wrote {n} lines -> {args.out_jsonl}")
    print(f"[Stats] empty first pred: {empty_first}/{n}")
    print("[Stats] parse method counts:")
    for k in sorted(method_counts.keys()):
        print(f"  {k}: {method_counts[k]}")


if __name__ == "__main__":
    main()
