#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from data import load_test_data_from_saved_indices_fixed, DecoderTestDataset
from decoder import SoftPromptStarCoderDecoder, load_tunable_parameters
from metrics import evaluate_and_save


DECODER_MODEL_ID = "bigcode/starcoder2-3b"
FIXED_EMB_DIR = "/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/buggy_fixed_embeddings"

MAX_LEN = 512
BATCH_SIZE = 20
PROMPT_LEN = 128
MAX_NEW_TOKENS = 128

INDICES_DIR = "saved_indices"
CKPT_PATH = "checkpoints/checkpoint_epoch_2.pt"
DO_CODEBLEU = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_PRED_CODE_DIR = "results/pred_fixed_code"
SAVE_PRED_CODE_JSONL = os.path.join(SAVE_PRED_CODE_DIR, "fixed_code_test.jsonl")


def main():
    assert os.path.exists(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}"
    assert os.path.exists(INDICES_DIR), f"Indices dir not found: {INDICES_DIR}"
    assert os.path.exists(FIXED_EMB_DIR), f"Embedding dir not found: {FIXED_EMB_DIR}"

    (
        fixed_texts,
        languages,
        problem_ids,
        buggy_submission_ids,
        fixed_submission_ids,
        fixed_emb_all,
        test_idx,
        global_test_indices,
    ) = load_test_data_from_saved_indices_fixed(INDICES_DIR, FIXED_EMB_DIR)

    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading decoder (bfloat16)...")
    decoder = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(DEVICE)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    print(f"Tokenizing (Max Len={MAX_LEN})...")
    enc = tokenizer(
        fixed_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    test_ds = DecoderTestDataset(
        cond_emb=fixed_emb_all[test_idx],
        input_ids=enc.input_ids[test_idx],
        attention_mask=enc.attention_mask[test_idx],
        languages=[languages[i] for i in test_idx],
        global_indices=global_test_indices,
        problem_ids=[problem_ids[i] for i in test_idx],
        buggy_submission_ids=[buggy_submission_ids[i] for i in test_idx],
        fixed_submission_ids=[fixed_submission_ids[i] for i in test_idx],
        gt_fixed_codes=[fixed_texts[i] for i in test_idx],
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = SoftPromptStarCoderDecoder(
        cond_dim=int(fixed_emb_all.shape[1]),
        decoder_model=decoder,
        tokenizer=tokenizer,
        prompt_len=PROMPT_LEN,
        device=DEVICE,
    ).to(DEVICE)

    try:
        model.prompt_proj.to(torch.bfloat16)
        model.ln.to(torch.bfloat16)
    except Exception as e:
        print(f"[Warn] Failed to cast prompt modules to bfloat16: {e}")

    load_tunable_parameters(model, CKPT_PATH)

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
