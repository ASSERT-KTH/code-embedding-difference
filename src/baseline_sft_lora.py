#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import numpy as np
from typing import Dict, Any, List
import yaml
from transformers import DataCollatorForSeq2Seq
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

HF_DATASET_ID = "ASSERT-KTH/RunBugRun-Final"
DEFAULT_MODEL_ID = "bigcode/starcoder2-3b"


import os

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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    ap.add_argument("--model_id", type=str, default=None)
    ap.add_argument("--indices_dir", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args_cli = ap.parse_args()

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
        length_column_name="length",
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


if __name__ == "__main__":
    main()