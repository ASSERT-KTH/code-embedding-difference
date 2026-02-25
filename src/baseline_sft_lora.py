#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import numpy as np
from typing import Dict, Any, List
import yaml
import time
from transformers import DataCollatorForSeq2Seq
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel
from torch.utils.data import Dataset as TorchDataset, DataLoader
from logger import Logger
import sys
import json
from tqdm import tqdm
from typing import Union
StopIds = Union[int, List[int]]


HF_DATASET_ID = "ASSERT-KTH/RunBugRun-Final"
DEFAULT_MODEL_ID = "bigcode/starcoder2-3b"


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def build_run_name(cfg: dict) -> str:
    wandb_cfg = (cfg.get("wandb") or {})
    base_name = str(wandb_cfg.get("name", "run"))

    trial_id = cfg.get("_trial_id", None)
    config_name = cfg.get("_config_name", None)

    suffix_parts = []
    if config_name:
        suffix_parts.append(str(config_name))
    if trial_id is not None:
        suffix_parts.append(f"t{trial_id}")

    return base_name if not suffix_parts else base_name + "_" + "_".join(suffix_parts)

def setup_wandb(cfg: dict, outdir: str):
    wandb_cfg = cfg.get("wandb", {}) or {}
    if not bool(wandb_cfg.get("enabled", False)):
        return None

    if not is_rank0():
        os.environ["WANDB_DISABLED"] = "true"
        return None

    import wandb

    os.makedirs(outdir, exist_ok=True)
    run_id_path = os.path.join(outdir, "wandb_run_id.txt")

    run_name = build_run_name(cfg)

    group = str(wandb_cfg.get("group", "")).strip() or None
    tags = wandb_cfg.get("tags", None)

    # use same run id
    if os.path.exists(run_id_path):
        run_id = open(run_id_path, "r", encoding="utf-8").read().strip()
        run = wandb.init(
            entity=wandb_cfg.get("entity"),
            project=wandb_cfg.get("project"),
            name=run_name,
            id=run_id,
            resume="must",
            config=cfg,
            group=group,
            tags=tags,
        )
        return run

    # First run: create a new run and write down its id
    run = wandb.init(
        entity=wandb_cfg.get("entity"),
        project=wandb_cfg.get("project"),
        name=run_name,
        config=cfg,
        group=group,
        tags=tags,
    )
    with open(run_id_path, "w", encoding="utf-8") as f:
        f.write(run.id)
    return run




# -------------------------
# Prompt (baseline #1 + stronger output constraint)
# -------------------------
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


def build_target(gt_fixed_code: str) -> str:
    fixed = (gt_fixed_code or "").rstrip()
    return fixed + "\n```\n<END>"


# -------------------------
# Indices loading
# -------------------------
def load_indices(indices_dir: str, split: str) -> np.ndarray:
    return np.load(os.path.join(indices_dir, f"{split}_idx.npy"))


def load_global_target_indices(indices_dir: str) -> np.ndarray:
    return np.load(os.path.join(indices_dir, "global_target_indices.npy"))


def load_hf_subset(global_target_indices: np.ndarray) -> Dict[str, List[Any]]:
    ds = load_dataset(HF_DATASET_ID, split="train")
    subset = ds.select(global_target_indices.tolist())

    buggy = [str(x) if x is not None else "" for x in subset["buggy_code"]]
    fixed = [str(x) if x is not None else "" for x in subset["fixed_code"]]

    return {
        "buggy_code": buggy,
        "fixed_code": fixed,
        "language": list(subset["language"]),
        "problem_id": list(subset["problem_id"]),
        "buggy_submission_id": list(subset["buggy_submission_id"]),
        "fixed_submission_id": list(subset["fixed_submission_id"]),
    }


def make_split_dataset(all_subset: Dict[str, List[Any]], split_idx: np.ndarray) -> Dataset:
    idxs = split_idx.tolist()

    def take(arr): return [arr[i] for i in idxs]

    data = {k: take(v) for k, v in all_subset.items()}
    return Dataset.from_dict(data)

def take_by_indices(all_subset: Dict[str, List[Any]], idx: np.ndarray) -> Dict[str, List[Any]]:
    idxs = idx.tolist()
    def take(arr): return [arr[i] for i in idxs]
    return {k: take(v) for k, v in all_subset.items()}

# -------------------------
# Tokenization + label masking (completion SFT)
# -------------------------
def preprocess_batch(examples: Dict[str, List[Any]], tokenizer: AutoTokenizer, max_seq_len: int) -> Dict[str, Any]:
    prompts = [build_prompt(b, l) for b, l in zip(examples["buggy_code"], examples["language"])]
    targets = [build_target(g) for g in examples["fixed_code"]]
    full_texts = [p + t for p, t in zip(prompts, targets)]

    enc_full = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        return_attention_mask=True,
        add_special_tokens=True,
    )

    enc_prompt = tokenizer(
        prompts,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        add_special_tokens=True,
    )

    input_ids = enc_full["input_ids"]
    attention_mask = enc_full["attention_mask"]

    labels = []
    for ids, p_ids in zip(input_ids, enc_prompt["input_ids"]):
        p_len = len(p_ids)
        lab = [-100] * p_len + ids[p_len:]
        lab = lab[: len(ids)]
        labels.append(lab)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# -------------------------
# Callback: export loss curve (train + eval) after training
# -------------------------
class LossCurveCallback(TrainerCallback):
    def __init__(self, out_csv_path: str, smooth_window: int = 200):
        self.out_csv_path = out_csv_path
        self.smooth_window = smooth_window

    @staticmethod
    def _moving_average(xs: List[float], w: int) -> List[float]:
        if w <= 1:
            return xs
        out = []
        s = 0.0
        q = []
        for x in xs:
            q.append(x)
            s += x
            if len(q) > w:
                s -= q.pop(0)
            out.append(s / len(q))
        return out

    def on_train_end(self, args, state, control, **kwargs):
        train_steps, train_losses = [], []
        eval_steps, eval_losses = [], []

        for rec in state.log_history:
            if "loss" in rec and "step" in rec and "eval_loss" not in rec:
                train_steps.append(int(rec["step"]))
                train_losses.append(float(rec["loss"]))
            if "eval_loss" in rec and "step" in rec:
                eval_steps.append(int(rec["step"]))
                eval_losses.append(float(rec["eval_loss"]))

        train_losses_smooth = self._moving_average(train_losses, self.smooth_window)

        os.makedirs(os.path.dirname(self.out_csv_path), exist_ok=True)
        with open(self.out_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["type", "step", "loss", "loss_smooth"])
            for s, l, ls in zip(train_steps, train_losses, train_losses_smooth):
                w.writerow(["train", s, l, ls])
            for s, l in zip(eval_steps, eval_losses):
                w.writerow(["eval", s, l, ""])

        print(f"[LossCurveCallback] Wrote loss curve CSV -> {self.out_csv_path}")


# -------------------------
# Main
# -------------------------
def run_train(args_cli):
    # ---- Load YAML config ----
    with open(args_cli.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # ---- Apply CLI overrides ----
    if args_cli.model_id is not None:
        cfg["model_id"] = args_cli.model_id
    if args_cli.indices_dir is not None:
        cfg["indices_dir"] = args_cli.indices_dir
    if args_cli.output_dir is not None:
        cfg["output_dir"] = args_cli.output_dir

    # ---- Required keys check ----
    cfg.setdefault("model_id", DEFAULT_MODEL_ID)
    required = ["indices_dir", "output_dir"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}. Please set them in YAML or pass via CLI.")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    if is_rank0():
        log_path = os.path.join(cfg["output_dir"], "training_log.txt")
        sys.stdout = Logger(log_path)
        sys.stderr = sys.stdout
    wandb_cfg = cfg.get("wandb", {}) or {}
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    run_name = build_run_name(cfg)

    run = setup_wandb(cfg, cfg["output_dir"])


    # ---- Load indices + data ----
    global_target_indices = load_global_target_indices(cfg["indices_dir"])
    subset_all = load_hf_subset(global_target_indices)

    train_idx = load_indices(cfg["indices_dir"], "train")
    val_idx = load_indices(cfg["indices_dir"], "val")

    train_ds = make_split_dataset(subset_all, train_idx)

    # ---- val_small ----
    val_small_seed = int(cfg.get("val_small_seed", 123))
    val_small_size = int(cfg.get("val_small_size", 2000))
    rng = np.random.default_rng(val_small_seed)
    if len(val_idx) <= val_small_size:
        val_small_idx = val_idx
    else:
        val_small_idx = rng.choice(val_idx, size=val_small_size, replace=False)

    val_small_path = os.path.join(cfg["output_dir"], "val_small_idx.npy")
    np.save(val_small_path, val_small_idx)
    print(f"[Info] val_small_size={len(val_small_idx)} saved -> {val_small_path}")

    val_small_ds = make_split_dataset(subset_all, val_small_idx)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    if "<END>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})
    tokenizer.padding_side = "right"

    # ---- Model ----
    use_qlora = bool(cfg.get("use_qlora", False))
    if use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.bfloat16,
        )

    model.resize_token_embeddings(len(tokenizer))

    # ---- LoRA ----
    lora_target_modules = cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    if not isinstance(lora_target_modules, list) or not all(isinstance(x, str) for x in lora_target_modules):
        raise ValueError("Config field 'lora_target_modules' must be a list of strings.")

    # module name hits
    if bool(cfg.get("debug_lora_modules", True)):
        cands = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "Wqkv", "query", "key", "value"]
        hits = [n for n, _m in model.named_modules() if any(k in n for k in cands)]
        print(f"[Debug] module name hits (first 50): {hits[:50]}")
        print(f"[Debug] hits_count={len(hits)}")

    lora_cfg = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- Training args ----
    training_args = TrainingArguments(
        group_by_length=True,
        output_dir=cfg["output_dir"],
        num_train_epochs=float(cfg.get("epochs", 5)),

        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 16)),

        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "cosine")),

        logging_steps=int(cfg.get("logging_steps", 50)),

        eval_strategy="steps",
        eval_steps=int(cfg.get("eval_steps", 2000)),

        save_strategy="steps",
        save_steps=int(cfg.get("save_steps", 2000)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        bf16=bool(cfg.get("bf16", True)),
        fp16=bool(cfg.get("fp16", False)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),

        report_to = ["wandb"] if (wandb_enabled and is_rank0()) else ["none"],
        run_name=run_name,

        seed=int(cfg.get("seed", 42)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 2)),
        remove_unused_columns=False,
    )

    # LoRA + gradient checkpointing:
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    model.train()

    # ---- Tokenize datasets ----
    max_seq_len = int(cfg.get("max_seq_len", 4096))
    train_tok = train_ds.map(
        lambda ex: preprocess_batch(ex, tokenizer, max_seq_len),
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train (prompt-masked labels)",
    )
    val_tok = val_small_ds.map(
        lambda ex: preprocess_batch(ex, tokenizer, max_seq_len),
        batched=True,
        remove_columns=val_small_ds.column_names,
        desc="Tokenizing val_small (prompt-masked labels)",
    )

    # collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
    )


    # loss curve CSV
    loss_csv = os.path.join(cfg["output_dir"], "loss_curve.csv")
    smooth_window = int(cfg.get("loss_curve_smooth_window", 200))
    callbacks = [LossCurveCallback(loss_csv, smooth_window=smooth_window)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=True)

    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    print(f"\n[Done] Saved to: {cfg['output_dir']}")
    print(f"[Done] Loss curve CSV: {loss_csv}\n")



# =====================
# RunBugRun TEST generation using a finetuned LoRA checkpoint.


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

# =====================
# Multi-process helpers
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
    """
    Returns:
      prompts: List[str]                         (B)
      prompt_len_tokens: List[int]               (B) true prompt length (no pad) from attention_mask
      full_texts: List[List[str]]                (B, N_SAMPLES) prompt(no pad) + continuation_trim
      continuation_texts: List[List[str]]        (B, N_SAMPLES) continuation_trim only
      full_len_tokens: List[List[int]]           (B, N_SAMPLES) lengths of (prompt_no_pad + cont_trim)
      new_len_tokens: List[List[int]]            (B, N_SAMPLES) lengths of cont_trim
      finish_reason: List[List[str]]             (B, N_SAMPLES) stop/length/unknown
      stop_pos_in_new: List[List[int]]           (B, N_SAMPLES) first stop position in untrimmed new_ids, -1 if none
    """
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
        eos_token_id=stop_ids,  # IMPORTANT: includes <END>
        return_dict_in_generate=True,
        output_scores=False,
    )

    sequences = gen_out.sequences
    if N_SAMPLES > 1:
        sequences = sequences.view(B, N_SAMPLES, -1)
    else:
        sequences = sequences.unsqueeze(1)

    input_len = enc.input_ids.shape[1]  # padded width; new tokens start after this

    full_texts: List[List[str]] = []
    continuation_texts: List[List[str]] = []
    full_len_tokens: List[List[int]] = []
    new_len_tokens: List[List[int]] = []
    finish_reason: List[List[str]] = []
    stop_pos_in_new: List[List[int]] = []

    for i in range(B):
        # prompt ids without padding (works with left/right padding)
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


@torch.no_grad()
def run_test_and_save(decoder, loader, tokenizer, save_jsonl_path: str, device: str, rank: int, world: int, stop_ids):
    os.makedirs(os.path.dirname(save_jsonl_path), exist_ok=True)
    with open(save_jsonl_path, "w", encoding="utf-8") as f_out:
        print(f"[Rank{rank}] Saving JSONL -> {save_jsonl_path}")

        printed_debug = False
        pbar = tqdm(loader, desc=f"Testing(rank{rank}/{world})", leave=True)

        for (
            buggy_text, gt_fixed_text, lang,
            gidx, pid, buggy_sid, fixed_sid
        ) in pbar:
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

                    # outputs (per candidate)
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

# =====================
class PromptBaselineTestDataset(TorchDataset):
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


def run_test(args):
    # ---- checks ----
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

    # -------------------------
    # Load indices + HF subset (REUSE TRAIN LOADERS)
    # -------------------------
    global_target_indices = load_global_target_indices(args.indices_dir)   # (subset_size,)
    subset_all = load_hf_subset(global_target_indices)                     # dict-of-lists, len=subset_size
    test_idx = load_indices(args.indices_dir, "test")                      # (test_size,), indices into subset_all

    split = take_by_indices(subset_all, test_idx)                          # dict-of-lists, len=test_size

    buggy_texts = split["buggy_code"]
    fixed_texts = split["fixed_code"]
    languages = split["language"]
    problem_ids = split["problem_id"]
    buggy_sids = split["buggy_submission_id"]
    fixed_sids = split["fixed_submission_id"]

    # global indices for logging (same semantics as before)
    global_test_indices = global_target_indices[test_idx]

    if rank == 0:
        print(f"[Info] Subset size={len(global_target_indices)} | test size={len(test_idx)}")

    # shard by rank
    buggy_texts = shard_list(buggy_texts, rank, world)
    fixed_texts = shard_list(fixed_texts, rank, world)
    languages = shard_list(languages, rank, world)
    global_test_indices = shard_ndarray(global_test_indices, rank, world)
    problem_ids = shard_list(problem_ids, rank, world)
    buggy_sids = shard_list(buggy_sids, rank, world)
    fixed_sids = shard_list(fixed_sids, rank, world)

    print(f"[Rank{rank}] shard size = {len(buggy_texts)}")

    # -------------------------
    # Tokenizer (prefer adapter_dir)
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

    tokenizer.padding_side = "left"  # good for generation

    end_id = tokenizer.convert_tokens_to_ids("<END>")
    stop_ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))
    if end_id is not None and end_id != tokenizer.unk_token_id:
        stop_ids.append(int(end_id))

    if rank == 0:
        print(
            f"[Tokenizer] pad_id={tokenizer.pad_token_id} eos_id={tokenizer.eos_token_id} "
            f"end_id={end_id} stop_ids={stop_ids}"
        )
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
        return_tensors="pt",
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
        pin_memory=True,
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
        low_cpu_mem_usage=True,
    ).to(device)

    # IMPORTANT: resize base BEFORE loading adapter
    base.resize_token_embeddings(len(tokenizer))

    decoder = PeftModel.from_pretrained(base, args.adapter_dir).to(device)

    # If tokenizer added tokens, resize embeddings
    decoder.resize_token_embeddings(len(tokenizer))

    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    os.makedirs(args.save_dir, exist_ok=True)
    save_jsonl = os.path.join(args.save_dir, f"fixed_code_test.rank{rank}.jsonl")

    t0 = time.time()
    run_test_and_save(decoder, test_loader, tokenizer, save_jsonl, device, rank, world, stop_ids)
    print(f"[Rank{rank}] Done. Elapsed: {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_tr = sub.add_parser("train")
    ap_tr.add_argument("--config", required=True)
    ap_tr.add_argument("--model_id", default=None)
    ap_tr.add_argument("--indices_dir", default=None)
    ap_tr.add_argument("--output_dir", default=None)

    ap_te = sub.add_parser("test")
    ap_te.add_argument("--adapter_dir", required=True)
    ap_te.add_argument("--base_model_id", default=DEFAULT_MODEL_ID)
    ap_te.add_argument("--indices_dir", required=True)
    ap_te.add_argument("--save_dir", required=True)

    args = ap.parse_args()
    if args.cmd == "train":
        run_train(args)
    else:
        run_test(args)


if __name__ == "__main__":
    main()
