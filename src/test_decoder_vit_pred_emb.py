#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Dict, Any
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from data import load_embeddings_for_global_indices, DecoderTestDataset
from decoder import SoftPromptStarCoderDecoder, load_tunable_parameters
from metrics import evaluate_and_save


try:
    from codebleu import calc_codebleu  # noqa: F401
    HAS_CODEBLEU = True
except Exception:
    HAS_CODEBLEU = False


INDICES_DIR = "saved_indices"

BUGGY_FIXED_EMB_DIR = "/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/buggy_fixed_embeddings"
VIT_CKPT_PATH = "/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/vit1d_buggy_to_fixed.pth"

DECODER_TUNABLE_CKPT = "checkpoints/checkpoint_epoch_2.pt"

SAVE_PRED_EMB_PATH = "pred_fixed_emb_test.pt"

DECODER_MODEL_ID = "bigcode/starcoder2-3b"
MAX_LEN = 512
PROMPT_LEN = 128

BATCH_VIT = 512
BATCH_DECODER = 20

MAX_NEW_TOKENS = 128

DO_CODEBLEU = True

CHUNK_PATTERN = "buggy_fixed_embeddings_chunk_{:04d}.pkl"
BUGGY_PREFER_KEYS = ["buggy_embeddings", "buggy_emb", "buggy", "embeddings"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_PRED_CODE_DIR = "results/pred_fixed_code"
SAVE_PRED_CODE_JSONL = os.path.join(SAVE_PRED_CODE_DIR, "pred_fixed_code_test.jsonl")


class ViT1D(nn.Module):
    def __init__(self, input_size=1024, patch_size=16, emb_dim=256, depth=4, heads=8, mlp_ratio=4):
        super().__init__()
        assert input_size % patch_size == 0, "input_size must be divisible by patch_size"
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = input_size // patch_size
        self.emb_dim = emb_dim

        self.patch_embed = nn.Linear(patch_size, emb_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=heads,
            dim_feedforward=int(emb_dim * mlp_ratio),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.output_layer = nn.Linear(emb_dim * self.num_patches, input_size)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.num_patches, self.patch_size)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.reshape(B, -1)
        x = self.output_layer(x)
        return x


def _strip_module_prefix_if_needed(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(sd.keys())
    if len(keys) > 0 and all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def infer_vit1d_config_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    pe_w = sd.get("patch_embed.weight", None)
    pos = sd.get("pos_embed", None)
    out_w = sd.get("output_layer.weight", None)

    if pe_w is None or pos is None or out_w is None:
        raise ValueError(
            "State dict does not look like ViT1D. Missing patch_embed.weight / pos_embed / output_layer.weight."
        )

    emb_dim, patch_size = pe_w.shape
    num_patches = pos.shape[1]
    input_size = out_w.shape[0]

    layer_ids = []
    for k in sd.keys():
        m = re.match(r"transformer\.layers\.(\d+)\.", k)
        if m:
            layer_ids.append(int(m.group(1)))
    depth = (max(layer_ids) + 1) if layer_ids else 4

    dim_ff = None
    for k, v in sd.items():
        if k == "transformer.layers.0.linear1.weight":
            dim_ff = v.shape[0]
            break
    mlp_ratio = (dim_ff / emb_dim) if dim_ff is not None else 4.0

    heads = 8 if emb_dim % 8 == 0 else (4 if emb_dim % 4 == 0 else 1)

    if input_size % patch_size != 0:
        raise ValueError(f"Invalid config inferred: input_size({input_size}) not divisible by patch_size({patch_size}).")
    if input_size // patch_size != num_patches:
        print(
            f"[Warn] inferred num_patches mismatch: input_size/patch_size={input_size // patch_size} vs pos_embed={num_patches}"
        )

    return dict(
        input_size=int(input_size),
        patch_size=int(patch_size),
        emb_dim=int(emb_dim),
        depth=int(depth),
        heads=int(heads),
        mlp_ratio=float(mlp_ratio),
    )


def load_vit1d_from_pth(vit_pth: str) -> nn.Module:
    raw = torch.load(vit_pth, map_location="cpu")

    if isinstance(raw, dict):
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            sd = raw["state_dict"]
        elif "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
            sd = raw["model_state_dict"]
        else:
            sd = raw
    else:
        raise ValueError("vit_pth is not a state_dict/checkpoint dict.")

    sd = _strip_module_prefix_if_needed(sd)
    cfg = infer_vit1d_config_from_state_dict(sd)
    print(f"  >>> [ViT1D config inferred] {cfg}")

    model = ViT1D(**cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  >>> [Loaded ViT1D weights] <- {vit_pth}")
    if len(unexpected) > 0:
        print(f"  [Warn] unexpected keys: {unexpected[:5]} ...")
    if len(missing) > 0:
        print(f"  [Warn] missing keys: {missing[:5]} ...")

    model.eval()
    return model


def main():
    for p in [INDICES_DIR, BUGGY_FIXED_EMB_DIR]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Not found: {p}")
    for p in [VIT_CKPT_PATH, DECODER_TUNABLE_CKPT]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Not found: {p}")

    global_target_indices = np.load(os.path.join(INDICES_DIR, "global_target_indices.npy"))
    test_idx = np.load(os.path.join(INDICES_DIR, "test_idx.npy"))
    global_test_indices = global_target_indices[test_idx]
    print(f"[Info] Test samples: {len(global_test_indices)}")

    print("Loading HuggingFace dataset (test subset)...")
    ds = load_dataset("ASSERT-KTH/RunBugRun-Final", split="train")
    subset_test = ds.select(global_test_indices.tolist())

    fixed_texts = [str(x) if x is not None else "" for x in subset_test["fixed_code"]]
    languages = subset_test["language"]
    problem_ids = subset_test["problem_id"]
    buggy_submission_ids = subset_test["buggy_submission_id"]
    fixed_submission_ids = subset_test["fixed_submission_id"]

    buggy_emb_test = load_embeddings_for_global_indices(
        emb_dir=BUGGY_FIXED_EMB_DIR,
        global_indices=global_test_indices,
        chunk_pattern=CHUNK_PATTERN,
        prefer_keys=BUGGY_PREFER_KEYS,
    )
    print(f"[Info] buggy_emb_test shape: {tuple(buggy_emb_test.shape)}")

    print("Loading ViT1D mapper...")
    vit = load_vit1d_from_pth(VIT_CKPT_PATH).to(DEVICE)
    vit.eval()

    pred_chunks: List[torch.Tensor] = []
    vit_loader = DataLoader(buggy_emb_test, batch_size=BATCH_VIT, shuffle=False, num_workers=0)

    print("Precomputing pred_fixed_emb_test ...")
    for xb in tqdm(vit_loader, desc="ViT predict", leave=True):
        xb = xb.to(DEVICE)
        with torch.no_grad():
            yb = vit(xb).detach().cpu()
        pred_chunks.append(yb)

    pred_fixed_emb_test = torch.cat(pred_chunks, dim=0).contiguous()
    print(f"[Info] pred_fixed_emb_test shape: {tuple(pred_fixed_emb_test.shape)}")

    torch.save(
        {
            "global_test_indices": torch.tensor(global_test_indices, dtype=torch.long),
            "pred_fixed_emb_test": pred_fixed_emb_test,
        },
        SAVE_PRED_EMB_PATH,
    )
    print(f"  >>> [Saved] pred fixed emb -> {SAVE_PRED_EMB_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing fixed_code (GT)...")
    enc = tokenizer(
        fixed_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    print("Loading StarCoder decoder (bfloat16)...")
    decoder = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(DEVICE)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    cond_dim = int(pred_fixed_emb_test.shape[1])
    model = SoftPromptStarCoderDecoder(cond_dim, decoder, tokenizer, prompt_len=PROMPT_LEN, device=DEVICE).to(DEVICE)

    try:
        model.prompt_proj.to(torch.bfloat16)
        model.ln.to(torch.bfloat16)
    except Exception as e:
        print(f"[Warn] Failed to cast prompt modules to bfloat16: {e}")

    load_tunable_parameters(model, DECODER_TUNABLE_CKPT)

    test_ds = DecoderTestDataset(
        pred_fixed_emb_test,
        enc.input_ids,
        enc.attention_mask,
        languages,
        global_test_indices,
        problem_ids,
        buggy_submission_ids,
        fixed_submission_ids,
        fixed_texts,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_DECODER,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    evaluate_and_save(
        model,
        test_loader,
        tokenizer,
        device=DEVICE,
        max_new_tokens=MAX_NEW_TOKENS,
        do_codebleu=DO_CODEBLEU,
        save_jsonl_path=SAVE_PRED_CODE_JSONL,
    )


if __name__ == "__main__":
    main()
