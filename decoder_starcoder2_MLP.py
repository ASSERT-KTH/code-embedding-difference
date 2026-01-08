#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from codebleu import calc_codebleu
from sacrebleu import corpus_bleu
from sklearn.model_selection import train_test_split

# =====================
# Configuration
# =====================
DECODER_MODEL_ID = "bigcode/starcoder2-3b"
FIXED_EMB_DIR = "/mimer/NOBACKUP/groups/naiss2025-5-243/Embeddings_RBR/buggy_fixed_embeddings"

# [Full-data setting]
N_SAMPLES = 456749    # full ~450k
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

# max number of validation samples used each epoch
VAL_CHECK_LIMIT = 1500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# Utilities
# =====================
class Logger(object):
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# =====================
# Data loading
# =====================
def get_stratified_indices(ds, n_samples, seed=42):
    print("Scanning dataset language distribution...")
    all_langs = ds["language"]
    total_count = len(all_langs)
    all_indices = np.arange(total_count)

    # If requested samples exceed total, return all indices
    if n_samples >= total_count:
        print(f"[Info] Using full dataset ({total_count} samples).")
        return all_indices

    subset_ratio = n_samples / total_count
    try:
        _, selected_indices = train_test_split(
            all_indices, test_size=subset_ratio, stratify=all_langs, random_state=seed
        )
    except ValueError:
        print("[Warning] Stratified sampling failed. Falling back to random sampling.")
        _, selected_indices = train_test_split(all_indices, test_size=subset_ratio, random_state=seed)

    selected_indices = np.sort(selected_indices)
    return selected_indices


def load_data_stratified(n_samples: int):
    print("Loading HuggingFace dataset object...")
    ds = load_dataset("ASSERT-KTH/RunBugRun-Final", split="train")

    target_indices = get_stratified_indices(ds, n_samples, seed=RANDOM_SEED)
    print(f"Selected {len(target_indices)} indices.")

    print("Fetching text data...")
    subset = ds.select(target_indices)
    fixed_texts = [str(x) if x is not None else "" for x in subset["fixed_code"]]
    languages = subset["language"]

    print(f"Fetching embedding data from: {FIXED_EMB_DIR}")
    target_idx_set = set(target_indices)
    collected_embeddings = []
    global_counter = 0
    chunk_num = 0
    max_target_idx = target_indices[-1]

    while True:
        file_name = f"buggy_fixed_embeddings_chunk_{chunk_num:04d}.pkl"
        file_path = os.path.join(FIXED_EMB_DIR, file_name)

        if not os.path.exists(file_path):
            break
        if global_counter > max_target_idx:
            break

        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            chunk = data["fixed_embeddings"] if isinstance(data, dict) else data
            if isinstance(chunk, np.ndarray):
                chunk = chunk.tolist()

            chunk_size = len(chunk)

            # only iterate if the current chunk overlaps the target range
            chunk_end = global_counter + chunk_size
            if global_counter <= max_target_idx and chunk_end > target_indices[0]:
                for local_i in range(chunk_size):
                    current_global_id = global_counter + local_i
                    if current_global_id in target_idx_set:
                        collected_embeddings.append(chunk[local_i])

            global_counter += chunk_size
            chunk_num += 1
            if chunk_num % 10 == 0:
                print(f"  Processed chunk {chunk_num}...")

        except Exception as e:
            print(f"[Error] Failed to read chunk {file_name}: {e}")
            break

    fixed_emb_tensor = torch.tensor(collected_embeddings, dtype=torch.float32)
    assert len(fixed_texts) == len(fixed_emb_tensor), "Fatal: Alignment failed!"

    return fixed_texts, languages, fixed_emb_tensor, target_indices


def split_indices_stratified_local(n_total, languages, test_ratio=0.1, val_ratio=0.1, seed=42):
    all_indices = np.arange(n_total)
    try:
        train_val_idx, test_idx = train_test_split(
            all_indices, test_size=test_ratio, stratify=languages, random_state=seed
        )
    except ValueError:
        train_val_idx, test_idx = train_test_split(all_indices, test_size=test_ratio, random_state=seed)

    train_val_langs = [languages[i] for i in train_val_idx]

    # val_ratio is relative to the full set, so convert it here
    relative_val_size = val_ratio / (1 - test_ratio)

    try:
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=relative_val_size, stratify=train_val_langs, random_state=seed
        )
    except ValueError:
        train_idx, val_idx = train_test_split(train_val_idx, test_size=relative_val_size, random_state=seed)

    return train_idx, val_idx, test_idx


class DecoderDataset(Dataset):
    def __init__(self, cond_emb, input_ids, attention_mask, languages):
        self.cond_emb = cond_emb
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.languages = languages
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.cond_emb[idx], self.input_ids[idx], self.attention_mask[idx], self.languages[idx]

# =====================
# Model (MLP)
# =====================
class SoftPromptStarCoderDecoder(nn.Module):
    def __init__(self, cond_dim: int, decoder_model, tokenizer, prompt_len: int = 32):
        super().__init__()
        self.decoder = decoder_model
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.hidden_dim = decoder_model.config.hidden_size

        inter_dim = cond_dim * 4
        self.prompt_proj = nn.Sequential(
            nn.Linear(cond_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(inter_dim, prompt_len * self.hidden_dim)
        )
        self.ln = nn.LayerNorm(self.hidden_dim)

        for m in self.prompt_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for p in self.decoder.parameters():
            p.requires_grad = False
        self.decoder.eval()
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        self.eos_token_id = tokenizer.eos_token_id

    def forward(self, cond_emb, input_ids, attention_mask):
        B, T = input_ids.shape
        prompt = self.prompt_proj(cond_emb.to(self.prompt_proj[0].weight.dtype))
        prompt = prompt.view(B, self.prompt_len, self.hidden_dim)
        prompt = self.ln(prompt)

        tok_emb = self.decoder.get_input_embeddings()(input_ids)
        full_emb = torch.cat([prompt.to(self.decoder.dtype), tok_emb.to(self.decoder.dtype)], dim=1)

        prompt_mask = torch.ones(B, self.prompt_len, device=DEVICE, dtype=attention_mask.dtype)
        full_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        full_labels = torch.full((B, self.prompt_len + T), -100, device=DEVICE, dtype=torch.long)
        code_labels = input_ids.clone()
        code_labels[attention_mask == 0] = -100
        full_labels[:, self.prompt_len:] = code_labels

        return self.decoder(inputs_embeds=full_emb, attention_mask=full_mask, labels=full_labels).loss

    @torch.no_grad()
    def generate_fast(self, cond_emb: torch.Tensor, max_new_tokens: int = 128) -> List[str]:
        self.decoder.eval()
        B = cond_emb.shape[0]
        prompt = self.prompt_proj(cond_emb.to(self.prompt_proj[0].weight.dtype))
        prompt = prompt.view(B, self.prompt_len, self.hidden_dim)
        prompt = self.ln(prompt).to(self.decoder.dtype)
        p_mask = torch.ones(B, self.prompt_len, device=DEVICE, dtype=torch.long)

        generated_ids = self.decoder.generate(
            inputs_embeds=prompt,
            attention_mask=p_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

# =====================
# Main
# =====================

def save_tunable_parameters(model, path):
    """
    Save only parameters with requires_grad=True.
    """
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)
    print(f"  >>> [Saved] Tunable params only -> {path} ({len(saved_params)} keys)")


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

    # 1) Load data
    fixed_texts, languages, fixed_emb_data, global_target_indices = load_data_stratified(N_SAMPLES)

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
    enc = tokenizer(fixed_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")

    print("Splitting dataset...")
    train_idx, val_idx, test_idx = split_indices_stratified_local(
        len(fixed_texts), languages, test_ratio=TEST_SIZE, val_ratio=VAL_SIZE, seed=RANDOM_SEED
    )

    print(f"Stats -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Save indices
    print("Saving indices...")
    np.save("saved_indices/global_target_indices.npy", global_target_indices)
    np.save("saved_indices/train_idx.npy", train_idx)
    np.save("saved_indices/val_idx.npy", val_idx)
    np.save("saved_indices/test_idx.npy", test_idx)

    # Build Dataset
    train_ds = DecoderDataset(
        fixed_emb_data[train_idx],
        enc.input_ids[train_idx],
        enc.attention_mask[train_idx],
        [languages[i] for i in train_idx],
    )
    val_ds = DecoderDataset(
        fixed_emb_data[val_idx],
        enc.input_ids[val_idx],
        enc.attention_mask[val_idx],
        [languages[i] for i in val_idx],
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # in evaluation，decode GT from batch input_ids

    model = SoftPromptStarCoderDecoder(fixed_emb_data.shape[1], decoder, tokenizer, PROMPT_LEN).to(DEVICE)
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

            # Update postfix with current loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / max(step_count, 1)
        epoch_time = (time.time() - start_time) / 3600
        print(f"Epoch {epoch} Done ({epoch_time:.2f} hrs) | Train Avg Loss: {avg_train_loss:.6f}")

        val_loss, val_codebleu = evaluate_performance_fast(
            model, val_loader, tokenizer, epoch, limit=VAL_CHECK_LIMIT
        )

        # Save best loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_tunable_parameters(model, "checkpoints/best_model_loss.pt")
            print(f"  >>> [Best Loss] Saved (Loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Save best CodeBLEU
        if val_codebleu > best_val_codebleu:
            best_val_codebleu = val_codebleu
            save_tunable_parameters(model, "checkpoints/best_model_codebleu.pt")
            print(f"  >>> [Best CodeBLEU] Saved (Score: {val_codebleu:.2f})")

        # Save every epoch
        if epoch % SAVE_EVERY_EPOCH == 0:
            save_tunable_parameters(model, f"checkpoints/checkpoint_epoch_{epoch}.pt")

        if patience_counter >= PATIENCE:
            print("\n[Early Stopping] Triggered.")
            break

    print("\nTraining Finished.")

# =====================
# Evaluation
# =====================
NORMALIZE_LANG_MAP = {
    "python": "python", "java": "java", "javascript": "javascript", "js": "javascript",
    "c": "c", "cpp": "cpp", "c++": "cpp", "php": "php", "go": "go", "ruby": "ruby", "c#": "c_sharp"
}

def evaluate_performance_fast(model, loader, tokenizer, epoch, limit=1000):
    """
    Run only `limit` samples for fast evaluation, avoiding slow full validation.
    """
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_gts = []
    all_langs = []

    count = 0
    print(f"\nRunning Validation (Capped at {limit} samples)...")

    val_pbar = tqdm(loader, desc="Validating", leave=False)

    with torch.no_grad():
        for b_emb, b_ids, b_mask, b_langs in val_pbar:
            b_emb = b_emb.to(DEVICE)
            b_ids = b_ids.to(DEVICE)
            b_mask = b_mask.to(DEVICE)

            # Loss
            loss = model(b_emb, b_ids, b_mask)
            if not torch.isnan(loss):
                total_val_loss += loss.item()

            # Generate
            preds = model.generate_fast(b_emb)

            # Decode GT
            gts = tokenizer.batch_decode(b_ids, skip_special_tokens=True)

            all_preds.extend(preds)
            all_gts.extend(gts)
            all_langs.extend(b_langs)

            count += len(b_emb)
            val_pbar.set_postfix(count=f"{count}/{limit}")

            if count >= limit:
                break

    # Metrics
    avg_val_loss = total_val_loss / max(len(loader) * (count / len(loader.dataset)), 1)

    try:
        g_bleu = corpus_bleu(all_preds, [all_gts]).score
    except Exception:
        g_bleu = 0.0

    print("-" * 70)
    print(f"[Validation] Epoch {epoch} | Est. Loss: {avg_val_loss:.4f} | Text BLEU: {g_bleu:.2f}")

    # Weighted CodeBLEU
    lang_groups = {}
    for p, g, l in zip(all_preds, all_gts, all_langs):
        l_lower = l.lower()
        if l_lower not in lang_groups:
            lang_groups[l_lower] = {"p": [], "r": []}
        lang_groups[l_lower]["p"].append(p)
        lang_groups[l_lower]["r"].append(g)

    total_weighted_codebleu = 0.0
    total_valid_samples = 0

    print(f"{'Language':<12} | {'Count':<5} | {'CodeBLEU':<8}")

    for lang in sorted(lang_groups.keys()):
        data = lang_groups[lang]
        count = len(data["p"])
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
            total_weighted_codebleu += score * count
            total_valid_samples += count

        print(f"{lang:<12} | {count:<5} | {score:>8.2f}")

    avg_codebleu = 0.0
    if total_valid_samples > 0:
        avg_codebleu = total_weighted_codebleu / total_valid_samples

    print(f"AVERAGE      | {total_valid_samples:<5} | {avg_codebleu:>8.2f}")
    print("=" * 70 + "\n")

    return avg_val_loss, avg_codebleu


if __name__ == "__main__":
    main()
