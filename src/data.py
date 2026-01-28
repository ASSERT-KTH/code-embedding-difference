# data.py
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Optional
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def get_stratified_indices(ds, n_samples: int, seed: int = 42) -> np.ndarray:
    print("Scanning dataset language distribution...")
    all_langs = ds["language"]
    total_count = len(all_langs)
    all_indices = np.arange(total_count)

    if n_samples >= total_count:
        print(f"[Info] Using full dataset ({total_count} samples).")
        return all_indices

    subset_ratio = n_samples / total_count
    try:
        _, selected_indices = train_test_split(
            all_indices, test_size=subset_ratio, stratify=all_langs, random_state=seed
        )
    except ValueError:
        print("[Warning] Stratified sampling failed. Fallback to random.")
        _, selected_indices = train_test_split(all_indices, test_size=subset_ratio, random_state=seed)

    selected_indices = np.sort(selected_indices)
    return selected_indices


def load_data_stratified(
    n_samples: int,
    fixed_emb_dir: str,
    seed: int = 42,
) -> Tuple[List[str], List[str], torch.Tensor, np.ndarray]:
    print("Loading HuggingFace dataset object...")
    ds = load_dataset("ASSERT-KTH/RunBugRun-Final", split="train")

    target_indices = get_stratified_indices(ds, n_samples, seed=seed)
    print(f"Selected {len(target_indices)} indices.")

    print("Fetching text data...")
    subset = ds.select(target_indices)
    fixed_texts = [str(x) if x is not None else "" for x in subset["fixed_code"]]
    languages = list(subset["language"])

    print(f"Fetching embedding data from: {fixed_emb_dir}")
    target_idx_set = set(target_indices.tolist())
    collected_embeddings = []
    global_counter = 0
    chunk_num = 0
    max_target_idx = int(target_indices[-1])

    while True:
        file_name = f"buggy_fixed_embeddings_chunk_{chunk_num:04d}.pkl"
        file_path = os.path.join(fixed_emb_dir, file_name)

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
            chunk_end = global_counter + chunk_size

            if global_counter <= max_target_idx and chunk_end > int(target_indices[0]):
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


def split_indices_stratified_local(
    n_total: int,
    languages: List[str],
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    all_indices = np.arange(n_total)
    try:
        train_val_idx, test_idx = train_test_split(
            all_indices, test_size=test_ratio, stratify=languages, random_state=seed
        )
    except ValueError:
        train_val_idx, test_idx = train_test_split(all_indices, test_size=test_ratio, random_state=seed)

    train_val_langs = [languages[i] for i in train_val_idx]
    relative_val_size = val_ratio / (1 - test_ratio)

    try:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val_size,
            stratify=train_val_langs,
            random_state=seed,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(train_val_idx, test_size=relative_val_size, random_state=seed)

    return train_idx, val_idx, test_idx


class DecoderTrainDataset(Dataset):
    def __init__(self, cond_emb, input_ids, attention_mask, languages):
        self.cond_emb = cond_emb
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.languages = languages

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.cond_emb[idx], self.input_ids[idx], self.attention_mask[idx], self.languages[idx]


class DecoderTestDataset(Dataset):
    def __init__(
        self,
        cond_emb,
        input_ids,
        attention_mask,
        languages,
        global_indices,
        problem_ids,
        buggy_submission_ids,
        fixed_submission_ids,
        gt_fixed_codes,
    ):
        self.cond_emb = cond_emb
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.languages = languages

        self.global_indices = global_indices
        self.problem_ids = problem_ids
        self.buggy_submission_ids = buggy_submission_ids
        self.fixed_submission_ids = fixed_submission_ids
        self.gt_fixed_codes = gt_fixed_codes

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            self.cond_emb[idx],
            self.input_ids[idx],
            self.attention_mask[idx],
            self.languages[idx],
            int(self.global_indices[idx]),
            str(self.problem_ids[idx]),
            int(self.buggy_submission_ids[idx]),
            int(self.fixed_submission_ids[idx]),
            self.gt_fixed_codes[idx],
        )


def load_embeddings_for_global_indices(
    emb_dir: str,
    global_indices: np.ndarray,
    chunk_pattern: str = "buggy_fixed_embeddings_chunk_{:04d}.pkl",
    prefer_key: Optional[str] = None,
    prefer_keys: Optional[List[str]] = None,
) -> torch.Tensor:
    if len(global_indices) == 0:
        raise ValueError("global_indices is empty.")

    idx_list = global_indices.tolist()
    idx_set = set(idx_list)
    idx_max = int(max(idx_list))
    need = len(idx_list)

    found: Dict[int, Any] = {}
    global_counter = 0
    chunk_num = 0

    print(f"Loading embeddings from: {emb_dir}")

    while True:
        file_name = chunk_pattern.format(chunk_num)
        file_path = os.path.join(emb_dir, file_name)

        if not os.path.exists(file_path):
            break
        if global_counter > idx_max:
            break

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            if prefer_key is not None:
                if prefer_key not in data:
                    raise KeyError(
                        f"Key '{prefer_key}' not found in {file_path}. Available keys={list(data.keys())[:30]}"
                    )
                chunk = data[prefer_key]
            elif prefer_keys is not None:
                chunk = None
                for k in prefer_keys:
                    if k in data:
                        chunk = data[k]
                        break
                if chunk is None:
                    for _, v in data.items():
                        if isinstance(v, (list, np.ndarray)):
                            chunk = v
                            break
                if chunk is None:
                    raise KeyError(f"Cannot find embedding array in dict keys={list(data.keys())[:30]}")
            else:
                raise ValueError("Either prefer_key or prefer_keys must be provided for dict chunks.")
        else:
            chunk = data

        if isinstance(chunk, np.ndarray):
            chunk = chunk.tolist()

        chunk_size = len(chunk)

        for local_i in range(chunk_size):
            gid = global_counter + local_i
            if gid in idx_set and gid not in found:
                found[gid] = chunk[local_i]
                if len(found) >= need:
                    break

        global_counter += chunk_size
        chunk_num += 1

        if len(found) >= need:
            break

    if len(found) != need:
        missing = need - len(found)
        raise RuntimeError(
            f"Alignment failed: missing {missing} embeddings (found {len(found)}/{need}). "
            f"Check emb_dir/chunk_pattern/key name and whether chunks cover all indices."
        )

    arr = [found[i] for i in idx_list]
    emb = torch.tensor(arr, dtype=torch.float32)
    return emb


def load_test_data_from_saved_indices_fixed(
    indices_dir: str,
    fixed_emb_dir: str,
) -> Tuple[
    List[str],
    Any,
    Any,
    Any,
    Any,
    torch.Tensor,
    np.ndarray,
    np.ndarray,
]:
    global_target_indices = np.load(os.path.join(indices_dir, "global_target_indices.npy"))
    test_idx = np.load(os.path.join(indices_dir, "test_idx.npy"))

    global_test_indices = global_target_indices[test_idx]

    print("Loading HF dataset subset according to saved indices...")
    ds = load_dataset("ASSERT-KTH/RunBugRun-Final", split="train")
    subset = ds.select(global_target_indices.tolist())

    fixed_texts = [str(x) if x is not None else "" for x in subset["fixed_code"]]
    languages = subset["language"]
    problem_ids = subset["problem_id"]
    buggy_submission_ids = subset["buggy_submission_id"]
    fixed_submission_ids = subset["fixed_submission_id"]

    print("Loading embeddings aligned to global_target_indices...")
    fixed_emb_all = load_embeddings_for_global_indices(
        emb_dir=fixed_emb_dir,
        global_indices=global_target_indices,
        chunk_pattern="buggy_fixed_embeddings_chunk_{:04d}.pkl",
        prefer_key="fixed_embeddings",
    )

    assert len(fixed_texts) == len(fixed_emb_all), "Fatal: alignment failed (texts vs embeddings)"

    return (
        fixed_texts,
        languages,
        problem_ids,
        buggy_submission_ids,
        fixed_submission_ids,
        fixed_emb_all,
        test_idx,
        global_test_indices,
    )
