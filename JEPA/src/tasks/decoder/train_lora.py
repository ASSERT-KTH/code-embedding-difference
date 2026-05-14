#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_decoder_softprompt.py

Train decoder soft prompt (prompt_proj) with StarCoder2-3B frozen or LoRA-tuned.
Condition embeddings: z_pred from JEPA (train + val).

Input embedding pt format (from embed_jepa.py):
  train_emb_pt: {"z_pred": Tensor[N,D], "global_indices": LongTensor[N]}
  val_emb_pt:   {"z_pred": Tensor[M,D], "global_indices": LongTensor[M]}

Supervision:
  fixed_code from HF dataset (ASSERT-KTH/RunBugRun-Final, split=train)
  tokenize with StarCoder2 tokenizer
  teacher forcing loss (CrossEntropy)

DDP:
  torchrun --standalone --nproc_per_node=4 train_decoder_softprompt.py ...

Saves:
  out_dir/
    checkpoints/
      ckpt_step{global_step}.pt        # lightweight but true-resume checkpoint (trainable params + optimizer/scaler/meta)
      ckpt_best_val.pt                 # tunable params only
      ckpt_epoch{e}.pt                 # tunable params only
    train_log.jsonl
    resolved_args.json
"""

from __future__ import annotations

import os
import time
import argparse
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, ProfilerActivity

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from jepa.utils import (
    JSONLLogger,
    ddp_all_reduce_sum,
    ddp_barrier,
    ddp_cleanup,
    ddp_enabled,
    ddp_local_rank,
    ddp_rank,
    ddp_setup,
    ddp_world,
    is_main,
    load_embeddings_pt,
    unwrap_ddp,
)

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def save_tunable_parameters(model: nn.Module, path: str) -> None:
    """
    Save only tunable parameters for inference / lightweight eval use.
    """
    base_model = unwrap_ddp(model)

    saved = {
        "projector": {
            name: p.detach().to("cpu")
            for name, p in base_model.prompt_proj.named_parameters()
        },
        "ln": {
            name: p.detach().to("cpu")
            for name, p in base_model.ln.named_parameters()
        },
        "lora": {
            name: p.detach().to("cpu")
            for name, p in base_model.decoder.named_parameters()
            if p.requires_grad
        }
    }
    torch.save(saved, path)


def save_resume_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    path: str,
    epoch: int,
    global_step: int,
    best_val: float,
    args,
) -> None:
    """
    Lightweight but true-resume checkpoint:
      - trainable params only
      - optimizer state
      - scaler state
      - training metadata
    Does NOT save frozen decoder backbone.
    """
    base_model = unwrap_ddp(model)

    ckpt = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val": float(best_val),
        "args": vars(args),

        "projector": {
            name: p.detach().to("cpu")
            for name, p in base_model.prompt_proj.named_parameters()
        },
        "ln": {
            name: p.detach().to("cpu")
            for name, p in base_model.ln.named_parameters()
        },
        "lora": {
            name: p.detach().to("cpu")
            for name, p in base_model.decoder.named_parameters()
            if p.requires_grad
        },

        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(ckpt, path)

def load_resume_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    path: str,
) -> Tuple[int, int, float]:
    """
    Restore trainable params + optimizer/scaler + metadata.
    Returns:
      epoch, global_step, best_val
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    base_model = unwrap_ddp(model)

    if "projector" in ckpt:
        base_model.prompt_proj.load_state_dict(ckpt["projector"], strict=True)

    if "ln" in ckpt:
        base_model.ln.load_state_dict(ckpt["ln"], strict=True)

    if "lora" in ckpt:
        current_lora_state = {
            name: p
            for name, p in base_model.decoder.named_parameters()
            if p.requires_grad
        }

        missing = []
        for name, tensor in ckpt["lora"].items():
            if name in current_lora_state:
                current_lora_state[name].data.copy_(tensor)
            else:
                missing.append(name)

        if missing and is_main():
            print(f"[Warn] Some LoRA keys from checkpoint were not found: {missing[:10]}")

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    best_val = float(ckpt.get("best_val", float("inf")))

    return epoch, global_step, best_val


def get_total_flops_from_prof(prof) -> float:
    """
    Robust FLOPs aggregation over profiler events.
    """
    total_flops = 0.0
    try:
        for evt in prof.key_averages():
            fl = getattr(evt, "flops", None)
            if fl is not None:
                total_flops += float(fl)
    except Exception:
        return 0.0
    return float(total_flops)


# -------------------------
# Model: SoftPrompt Decoder
# -------------------------
class SoftPromptStarCoderDecoder(nn.Module):
    def __init__(self, cond_dim: int, decoder_model, tokenizer, prompt_len: int = 128):
        super().__init__()
        self.decoder = decoder_model
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.hidden_dim = int(decoder_model.config.hidden_size)

        inter_dim = cond_dim * 4
        self.prompt_proj = nn.Sequential(
            nn.Linear(cond_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(inter_dim, self.prompt_len * self.hidden_dim),
        )
        self.ln = nn.LayerNorm(self.hidden_dim)

        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        self.eos_token_id = tokenizer.eos_token_id

    def forward(self, cond_emb: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forcing loss on fixed_code tokens.
        """
        B, T = input_ids.shape

        prompt = self.prompt_proj(cond_emb.to(self.prompt_proj[0].weight.dtype))
        prompt = prompt.view(B, self.prompt_len, self.hidden_dim)
        prompt = self.ln(prompt)

        tok_emb = self.decoder.get_input_embeddings()(input_ids)
        full_emb = torch.cat([prompt.to(self.decoder.dtype), tok_emb.to(self.decoder.dtype)], dim=1)

        prompt_mask = torch.ones(B, self.prompt_len, device=input_ids.device, dtype=attention_mask.dtype)
        full_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        full_labels = torch.full((B, self.prompt_len + T), -100, device=input_ids.device, dtype=torch.long)
        code_labels = input_ids.clone()
        code_labels[attention_mask == 0] = -100
        full_labels[:, self.prompt_len:] = code_labels

        out = self.decoder(inputs_embeds=full_emb, attention_mask=full_mask, labels=full_labels)
        return out.loss


# -------------------------
# Dataset
# -------------------------
class CondCodeDataset(Dataset):
    def __init__(
        self,
        emb_cpu: torch.Tensor,            # [N,D] CPU float
        gidx_cpu: torch.Tensor,           # [N] CPU long
        fixed_codes: List[str],
        tokenizer,
        max_len: int,
    ):
        self.emb = emb_cpu
        self.gidx = gidx_cpu
        self.fixed_codes = fixed_codes
        self.tokenizer = tokenizer
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return int(self.emb.shape[0])

    def __getitem__(self, i: int):
        e = self.emb[i]
        gi = int(self.gidx[i])
        code = self.fixed_codes[i]
        enc = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        return e, enc["input_ids"], enc["attention_mask"], gi


def collate_pad(batch, pad_token_id: int):
    """
    batch: list of (emb, input_ids(list[int]), attention_mask(list[int]), global_idx)
    Pads to max length in batch.
    """
    embs, ids_list, mask_list, gidx = zip(*batch)

    embs = torch.stack([x.float() for x in embs], dim=0)
    gidx = torch.tensor(gidx, dtype=torch.long)

    max_t = max(len(x) for x in ids_list)
    B = len(ids_list)
    input_ids = torch.full((B, max_t), pad_token_id, dtype=torch.long)
    attn = torch.zeros((B, max_t), dtype=torch.long)

    for i in range(B):
        ids = torch.tensor(ids_list[i], dtype=torch.long)
        m = torch.tensor(mask_list[i], dtype=torch.long)
        input_ids[i, : ids.numel()] = ids
        attn[i, : m.numel()] = m

    return embs, input_ids, attn, gidx


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: torch.device, use_bf16: bool) -> float:
    model.eval()
    tot_loss = 0.0
    tot_n = 0

    for b_emb, b_ids, b_mask, _ in dl:
        bsz = b_emb.size(0)
        b_emb = b_emb.to(device, non_blocking=True)
        b_ids = b_ids.to(device, non_blocking=True)
        b_mask = b_mask.to(device, non_blocking=True)

        autocast_enabled = (device.type == "cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled and use_bf16):
            loss = model(b_emb, b_ids, b_mask)

        if not torch.isnan(loss):
            tot_loss += float(loss.item()) * bsz
            tot_n += bsz

    t = torch.tensor([tot_loss, float(tot_n)], device=device, dtype=torch.float64)
    ddp_all_reduce_sum(t)
    return (t[0] / t[1]).item() if t[1].item() > 0 else 0.0


def train(args) -> None:
    if ddp_enabled():
        ddp_setup("nccl" if torch.cuda.is_available() else "gloo")

    device = torch.device(f"cuda:{ddp_local_rank()}") if torch.cuda.is_available() else torch.device("cpu")
    r, w = ddp_rank(), ddp_world()

    os.makedirs(args.out_dir, exist_ok=True)
    if is_main():
        os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
        with open(os.path.join(args.out_dir, "resolved_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    logger = JSONLLogger(os.path.join(args.out_dir, "train_log.jsonl")) if is_main() else None

    # -------------------------
    # W&B init (rank0 only)
    # -------------------------
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb enabled but wandb is not installed in this environment.")
        if is_main():
            wandb.init(
                entity=args.wandb_entity or None,
                project=args.wandb_project or None,
                group=args.wandb_group or None,
                name=args.wandb_run_name or None,
                id=args.wandb_id or None,
                resume="never",
                config=vars(args),
                settings=wandb.Settings(init_timeout=180),
            )

    # Load embeddings
    train_emb, train_gidx = load_embeddings_pt(args.train_emb_pt, key=args.emb_key)
    val_emb, val_gidx = load_embeddings_pt(args.val_emb_pt, key=args.emb_key)

    if is_main():
        print(f"[Data] train_emb={tuple(train_emb.shape)} val_emb={tuple(val_emb.shape)} emb_key={args.emb_key}")

    # Load HF dataset fixed_code by global indices
    ds = load_dataset(args.hf_dataset_id, split=args.hf_split)

    train_fixed = [str(x) if x is not None else "" for x in ds.select(train_gidx.tolist())[args.hf_fixed_field]]
    val_fixed = [str(x) if x is not None else "" for x in ds.select(val_gidx.tolist())[args.hf_fixed_field]]

    # Tokenizer/model (StarCoder2)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    decoder = AutoModelForCausalLM.from_pretrained(
        args.decoder_model_id,
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    # Add LoRA on decoder
    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        decoder = get_peft_model(decoder, lora_config)
        if is_main():
            decoder.print_trainable_parameters()
    else:
        for p in decoder.parameters():
            p.requires_grad = False
        if is_main():
            print("[LoRA] disabled, decoder frozen")

    # Build soft prompt model
    cond_dim = int(train_emb.shape[1])
    model = SoftPromptStarCoderDecoder(cond_dim, decoder, tokenizer, prompt_len=args.prompt_len).to(device)

    # Cast tunable modules to bf16 for speed
    if device.type == "cuda" and args.use_bf16:
        try:
            model.prompt_proj.to(torch.bfloat16)
            model.ln.to(torch.bfloat16)
        except Exception as e:
            if is_main():
                print(f"[Warn] Failed to cast prompt modules to bf16: {e}")

    # Wrap DDP
    if ddp_enabled():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[ddp_local_rank()],
            find_unused_parameters=False
        )

    # Datasets / loaders
    train_ds = CondCodeDataset(train_emb, train_gidx, train_fixed, tokenizer, max_len=args.max_len)
    val_ds = CondCodeDataset(val_emb, val_gidx, val_fixed, tokenizer, max_len=args.max_len)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp_enabled() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp_enabled() else None

    collate = lambda b: collate_pad(b, pad_token_id=tokenizer.pad_token_id)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size or args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )

    base_model = model.module if hasattr(model, "module") else model

    projector_params = [p for p in base_model.prompt_proj.parameters() if p.requires_grad]
    ln_params = [p for p in base_model.ln.parameters() if p.requires_grad]
    lora_params = [p for p in base_model.decoder.parameters() if p.requires_grad]
    tunable_params = projector_params + ln_params + lora_params

    optimizer = torch.optim.AdamW(tunable_params, lr=args.lr, weight_decay=args.weight_decay)

    use_amp = (device.type == "cuda") and args.use_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # -------------------------
    # Resume
    # -------------------------
    start_epoch = 1
    global_step = 0
    best_val = float("inf")

    if args.resume_ckpt:
        resumed_epoch, resumed_global_step, resumed_best_val = load_resume_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            path=args.resume_ckpt,
        )
        start_epoch = resumed_epoch + 1
        global_step = resumed_global_step
        best_val = resumed_best_val

        if is_main():
            print(f"[Resume] loaded from {args.resume_ckpt}")
            print(f"[Resume] start_epoch={start_epoch} global_step={global_step} best_val={best_val:.4f}")

    # -------------------------
    # FLOPs profiler state
    # -------------------------
    prof = None
    profiling_active = bool(args.profile_flops and args.profile_steps > 0)
    profiled_update_steps = 0
    estimated_flops_per_update = None  # global total FLOPs/update across all ranks

    if profiling_active:
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        prof = profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=True,
        )
        prof.__enter__()

        if is_main():
            print(f"[FLOPs] profiling first {args.profile_steps} optimizer-update steps ...")

    if is_main():
        n_proj = sum(p.numel() for p in projector_params if p.requires_grad)
        n_ln = sum(p.numel() for p in ln_params if p.requires_grad)
        n_lora = sum(p.numel() for p in lora_params if p.requires_grad)
        print(f"[Trainable] projector={n_proj:,} ln={n_ln:,} lora={n_lora:,} total={n_proj + n_ln + n_lora:,}")
        print(f"[Train] world={w} batch={args.batch_size} grad_accum={args.grad_accum} lr={args.lr}")
        print(f"[Train] max_len={args.max_len} prompt_len={args.prompt_len} use_bf16={args.use_bf16} use_amp={use_amp}")

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        if ddp_enabled() and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        (model.module if hasattr(model, "module") else model).train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        running_n = 0
        t0 = time.time()

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True) if is_main() else train_dl

        for step, (b_emb, b_ids, b_mask, _) in enumerate(pbar):
            b_emb = b_emb.to(device, non_blocking=True)
            b_ids = b_ids.to(device, non_blocking=True)
            b_mask = b_mask.to(device, non_blocking=True)

            autocast_enabled = (device.type == "cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled and args.use_bf16):
                loss = (model(b_emb, b_ids, b_mask) / args.grad_accum)

            # backward
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # optimizer update
            if (step + 1) % args.grad_accum == 0:
                if args.clip_grad > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(tunable_params, args.clip_grad)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                # profiler step
                if profiling_active and prof is not None and estimated_flops_per_update is None:
                    prof.step()
                    profiled_update_steps += 1

                    if profiled_update_steps >= args.profile_steps:
                        local_total_flops = get_total_flops_from_prof(prof)

                        if is_main() and args.profile_print_tables:
                            try:
                                print("\n[FLOPs Debug] profiler table sorted by self_cuda_time_total")
                                print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
                            except Exception as e:
                                print(f"[Warn] Failed to print CUDA-time profiler table: {e}")

                            try:
                                print("\n[FLOPs Debug] profiler table sorted by flops")
                                print(prof.key_averages().table(sort_by="flops", row_limit=20))
                            except Exception as e:
                                print(f"[Warn] Failed to print FLOPs profiler table: {e}")

                        prof.__exit__(None, None, None)
                        prof = None
                        profiling_active = False

                        local_avg_flops = local_total_flops / float(profiled_update_steps)

                        t_flops = torch.tensor([local_avg_flops], device=device, dtype=torch.float64)
                        ddp_all_reduce_sum(t_flops)
                        estimated_flops_per_update = t_flops[0].item()

                        ddp_barrier()
                        if is_main():
                            if estimated_flops_per_update > 0:
                                print(f"[FLOPs] estimated_global_flops_per_update = {estimated_flops_per_update:.3e}")
                            else:
                                print("[Warn] profiler finished but FLOPs are still 0. "
                                      "This usually means PyTorch profiler did not attribute FLOPs "
                                      "for the fused/transformer kernels in this run.")

                cumulative_flops = None
                if estimated_flops_per_update is not None and estimated_flops_per_update > 0:
                    cumulative_flops = global_step * estimated_flops_per_update

                # track train loss (unscaled)
                loss_item = float(loss.item()) * args.grad_accum
                running_loss += loss_item * b_emb.size(0)
                running_n += b_emb.size(0)

                if is_main() and (global_step % args.log_every == 0):
                    avg = running_loss / max(1, running_n)
                    pbar.set_postfix(loss=f"{avg:.4f}")

                    if args.wandb:
                        log_dict = {
                            "train/loss": float(avg),
                            "train/epoch": epoch,
                            "train/step": global_step,
                            "lr": float(optimizer.param_groups[0]["lr"]),
                        }
                        if cumulative_flops is not None:
                            log_dict["train/cumulative_flops"] = float(cumulative_flops)
                            log_dict["train/flops_per_update"] = float(estimated_flops_per_update)
                        wandb.log(log_dict, step=global_step)

                # step-based resume checkpoint
                if args.save_every_steps > 0 and (global_step % args.save_every_steps == 0):
                    if is_main():
                        step_ckpt_path = os.path.join(
                            args.out_dir, "checkpoints", f"ckpt_step{global_step}.pt"
                        )
                        save_resume_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            path=step_ckpt_path,
                            epoch=epoch,
                            global_step=global_step,
                            best_val=best_val,
                            args=args,
                        )
                        print(f"[Save] step resume ckpt -> {step_ckpt_path}")

                # periodic eval
                if args.eval_every_steps > 0 and (global_step % args.eval_every_steps == 0):
                    if ddp_enabled() and val_sampler is not None:
                        val_sampler.set_epoch(epoch)

                    val_loss = evaluate(model, val_dl, device=device, use_bf16=args.use_bf16)

                    if is_main():
                        rec = {
                            "type": "eval",
                            "epoch": epoch,
                            "step": global_step,
                            "val_loss": float(val_loss),
                            "time": time.time(),
                        }
                        if estimated_flops_per_update is not None and estimated_flops_per_update > 0:
                            rec["flops_per_update"] = float(estimated_flops_per_update)
                            rec["cumulative_flops"] = float(global_step * estimated_flops_per_update)

                        logger.log(rec)
                        print(f"[Eval] epoch={epoch} step={global_step} val_loss={val_loss:.4f}")

                        if args.wandb:
                            val_log = {
                                "val/loss": float(val_loss),
                                "val/epoch": epoch,
                                "val/step": global_step,
                            }
                            if estimated_flops_per_update is not None and estimated_flops_per_update > 0:
                                val_log["val/cumulative_flops"] = float(global_step * estimated_flops_per_update)
                            wandb.log(val_log, step=global_step)

                        # save best tunable-only checkpoint
                        if val_loss < best_val:
                            best_val = val_loss
                            best_path = os.path.join(args.out_dir, "checkpoints", "ckpt_best_val.pt")
                            save_tunable_parameters(model, best_path)
                            print(f"[Save] new best -> {best_path} (val_loss={best_val:.4f})")

                    (model.module if hasattr(model, "module") else model).train()

        # epoch end train-loss reduce
        t = torch.tensor([running_loss, float(running_n)], device=device, dtype=torch.float64)
        ddp_all_reduce_sum(t)
        tr_loss = (t[0] / t[1]).item() if t[1].item() > 0 else 0.0

        # epoch end eval
        if ddp_enabled() and val_sampler is not None:
            val_sampler.set_epoch(epoch)
        val_loss = evaluate(model, val_dl, device=device, use_bf16=args.use_bf16)

        if is_main():
            dt = time.time() - t0
            rec = {
                "type": "epoch",
                "epoch": epoch,
                "step": global_step,
                "train_loss": float(tr_loss),
                "val_loss": float(val_loss),
                "seconds": float(dt),
                "time": time.time(),
            }
            if estimated_flops_per_update is not None and estimated_flops_per_update > 0:
                rec["flops_per_update"] = float(estimated_flops_per_update)
                rec["cumulative_flops"] = float(global_step * estimated_flops_per_update)
            logger.log(rec)

            print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f} time={dt/60:.1f} min")

            if args.wandb:
                epoch_log = {
                    "epoch/train_loss": float(tr_loss),
                    "epoch/val_loss": float(val_loss),
                    "epoch": epoch,
                }
                if estimated_flops_per_update is not None and estimated_flops_per_update > 0:
                    epoch_log["epoch/cumulative_flops"] = float(global_step * estimated_flops_per_update)
                wandb.log(epoch_log, step=global_step)

            # save epoch tunable-only checkpoint
            ep_path = os.path.join(args.out_dir, "checkpoints", f"ckpt_epoch{epoch}.pt")
            save_tunable_parameters(model, ep_path)
            print(f"[Save] epoch tunable ckpt -> {ep_path}")

            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(args.out_dir, "checkpoints", "ckpt_best_val.pt")
                save_tunable_parameters(model, best_path)
                print(f"[Save] new best -> {best_path} (val_loss={best_val:.4f})")

        ddp_barrier()

    # Safety: close profiler if training ended early
    if prof is not None:
        local_total_flops = get_total_flops_from_prof(prof)

        if is_main() and args.profile_print_tables:
            try:
                print("\n[FLOPs Debug] profiler table sorted by self_cuda_time_total")
                print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
            except Exception as e:
                print(f"[Warn] Failed to print CUDA-time profiler table: {e}")

            try:
                print("\n[FLOPs Debug] profiler table sorted by flops")
                print(prof.key_averages().table(sort_by="flops", row_limit=20))
            except Exception as e:
                print(f"[Warn] Failed to print FLOPs profiler table: {e}")

        prof.__exit__(None, None, None)
        prof = None

        if profiled_update_steps > 0 and estimated_flops_per_update is None:
            local_avg_flops = local_total_flops / float(profiled_update_steps)
            t_flops = torch.tensor([local_avg_flops], device=device, dtype=torch.float64)
            ddp_all_reduce_sum(t_flops)
            estimated_flops_per_update = t_flops[0].item()

            ddp_barrier()
            if is_main():
                if estimated_flops_per_update > 0:
                    print(f"[FLOPs] fallback estimated_global_flops_per_update = {estimated_flops_per_update:.3e}")
                else:
                    print("[Warn] profiler fallback also produced FLOPs=0. "
                          "Likely cause: FLOPs attribution is unavailable for the actual fused kernels used here.")

    if is_main():
        print("Training done.")

    if args.wandb and is_main():
        wandb.finish()

    if ddp_enabled():
        ddp_cleanup()


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    # Embeddings
    ap.add_argument("--train_emb_pt", type=str, required=True, help="train z_pred.pt (dict with z_pred + global_indices)")
    ap.add_argument("--val_emb_pt", type=str, required=True, help="val z_pred.pt (dict with z_pred + global_indices)")
    ap.add_argument("--emb_key", type=str, default="z_pred", help="key name of embedding tensor inside pt (default: z_pred)")

    # HF dataset
    ap.add_argument("--hf_dataset_id", type=str, default="ASSERT-KTH/RunBugRun-Final")
    ap.add_argument("--hf_split", type=str, default="train")
    ap.add_argument("--hf_fixed_field", type=str, default="fixed_code")

    # LoRA
    ap.add_argument("--use_lora", action="store_true", help="Enable LoRA on decoder")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Target modules for LoRA"
    )

    # Decoder
    ap.add_argument("--decoder_model_id", type=str, default="bigcode/starcoder2-3b")
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=512)

    # Wandb
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (rank0 only)")
    ap.add_argument("--wandb_entity", type=str, default="", help="W&B entity (optional)")
    ap.add_argument("--wandb_project", type=str, default="CodeRepair_JEPA", help="W&B project")
    ap.add_argument("--wandb_group", type=str, default="decoder_zpred", help="W&B group")
    ap.add_argument("--wandb_run_name", type=str, default="", help="W&B run name (optional)")
    ap.add_argument("--wandb_id", type=str, default="", help="W&B run id")
    ap.add_argument("--wandb_resume", type=str, default="auto", help="W&B resume mode: auto|must|never")

    # Train hyperparams
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--eval_batch_size", type=int, default=0, help="0 means use batch_size")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    # Resume / save
    ap.add_argument("--resume_ckpt", type=str, default="", help="Path to step resume checkpoint")
    ap.add_argument("--save_every_steps", type=int, default=0, help="Save true-resume checkpoint every N optimizer steps; 0 disables")

    # Runtime
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every_steps", type=int, default=0, help="0 disables step eval; epoch eval always runs")
    ap.add_argument("--use_bf16", action="store_true", help="use bf16 autocast on CUDA")
    ap.add_argument("--use_amp", action="store_true", help="use fp16 GradScaler (if you want fp16)")

    # Flops
    ap.add_argument("--profile_flops", action="store_true", help="Estimate FLOPs with torch.profiler")
    ap.add_argument("--profile_steps", type=int, default=5, help="Number of optimizer update steps used for FLOPs profiling")
    ap.add_argument("--profile_print_tables", action="store_true", help="Print profiler tables for FLOPs debugging")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
