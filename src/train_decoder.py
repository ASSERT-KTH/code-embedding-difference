#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from logger import Logger
from data import load_data_stratified, split_indices_stratified_local, DecoderTrainDataset
from decoder import SoftPromptStarCoderDecoder, save_tunable_parameters
from metrics import evaluate_performance_fast


DECODER_MODEL_ID = "bigcode/starcoder2-3b"
FIXED_EMB_DIR = "/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/buggy_fixed_embeddings"

N_SAMPLES = 456749
MAX_LEN = 512
RANDOM_SEED = 42

BATCH_SIZE = 20
NUM_EPOCHS = 5
PATIENCE = 2
LR = 5e-5
PROMPT_LEN = 128
TEST_SIZE = 0.05
VAL_SIZE = 0.05

SAVE_EVERY_EPOCH = 1
VAL_CHECK_LIMIT = 1500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("saved_indices", exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file_path = f"logs/train_log_{timestamp}.txt"
    sys.stdout = Logger(log_file_path)

    print(f"=== LARGE SCALE Training Started at {timestamp} ===")
    print(f"Logging to: {log_file_path}")
    print(f"Config: Epochs={NUM_EPOCHS}, Batch={BATCH_SIZE}, MaxLen={MAX_LEN}, LimitVal={VAL_CHECK_LIMIT}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    fixed_texts, languages, fixed_emb_data, global_target_indices = load_data_stratified(
        N_SAMPLES, FIXED_EMB_DIR, seed=RANDOM_SEED
    )

    tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model in bfloat16...")
    decoder = AutoModelForCausalLM.from_pretrained(
        DECODER_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(DEVICE)

    decoder.gradient_checkpointing_enable()
    decoder.enable_input_require_grads()

    print(f"Tokenizing data (Max Len = {MAX_LEN})...")
    enc = tokenizer(
        fixed_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    print("Splitting dataset...")
    train_idx, val_idx, test_idx = split_indices_stratified_local(
        len(fixed_texts),
        languages,
        test_ratio=TEST_SIZE,
        val_ratio=VAL_SIZE,
        seed=RANDOM_SEED,
    )
    print(f"Stats -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    print("Saving indices...")
    np.save("saved_indices/global_target_indices.npy", global_target_indices)
    np.save("saved_indices/train_idx.npy", train_idx)
    np.save("saved_indices/val_idx.npy", val_idx)
    np.save("saved_indices/test_idx.npy", test_idx)

    train_ds = DecoderTrainDataset(
        fixed_emb_data[train_idx],
        enc.input_ids[train_idx],
        enc.attention_mask[train_idx],
        [languages[i] for i in train_idx],
    )
    val_ds = DecoderTrainDataset(
        fixed_emb_data[val_idx],
        enc.input_ids[val_idx],
        enc.attention_mask[val_idx],
        [languages[i] for i in val_idx],
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    model = SoftPromptStarCoderDecoder(
        fixed_emb_data.shape[1], decoder, tokenizer, PROMPT_LEN, device=DEVICE
    ).to(DEVICE)

    model.prompt_proj.to(torch.bfloat16)
    model.ln.to(torch.bfloat16)

    optimizer = torch.optim.AdamW(model.prompt_proj.parameters(), lr=LR, weight_decay=0.01)

    best_val_loss = float("inf")
    best_val_codebleu = -1.0
    patience_counter = 0

    print("Starting training loop...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0
        step_count = 0
        start_time = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=True)

        for step, (b_emb, b_ids, b_mask, _) in enumerate(progress_bar):
            optimizer.zero_grad()
            loss = model(b_emb.to(DEVICE), b_ids.to(DEVICE), b_mask.to(DEVICE))

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.prompt_proj.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            step_count += 1
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / max(step_count, 1)
        epoch_time = (time.time() - start_time) / 3600
        print(f"Epoch {epoch} Done ({epoch_time:.2f} hrs) | Train Avg Loss: {avg_train_loss:.6f}")

        val_loss, val_codebleu = evaluate_performance_fast(
            model, val_loader, tokenizer, epoch, limit=VAL_CHECK_LIMIT, device=DEVICE
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  >>> [Best Loss] Saved (Loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        if val_codebleu > best_val_codebleu:
            best_val_codebleu = val_codebleu
            print(f"  >>> [Best CodeBLEU] Saved (Score: {val_codebleu:.2f})")

        if epoch % SAVE_EVERY_EPOCH == 0:
            save_tunable_parameters(model, f"checkpoints/checkpoint_epoch_{epoch}.pt")

        if patience_counter >= PATIENCE:
            print("\n[Early Stopping] Triggered.")
            break

    print("\nTraining Finished.")


if __name__ == "__main__":
    main()
