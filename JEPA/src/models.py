# models.py
# -*- coding: utf-8 -*-
"""
Models module for JEPA-style training on code pairs.

Goals:
- Flexible encoder: ModernBERT (or any HF AutoModel) with modes:
  - frozen: no trainable params
  - full: full fine-tuning
  - lora: LoRA fine-tuning via PEFT (optional dependency)
- Flexible predictor:
  - vit1d: your current "ViT on 1D embedding" predictor
  - mlp: simple MLP predictor (swap-in baseline)
- Provide small "factory" functions:
  - build_encoder(cfg, device)
  - build_predictor(cfg, emb_dim, device)
  - infer_emb_dim(model_name)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


# -------------------------
# Helpers
# -------------------------
def infer_emb_dim(model_name: str) -> int:
    """Infer hidden size from HF config (no weights needed)."""
    m = AutoModel.from_pretrained(model_name)
    return int(getattr(m.config, "hidden_size"))


def _freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def _unfreeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = True


# -------------------------
# Encoder
# -------------------------
class HFMeanPoolEncoder(nn.Module):
    """
    Encoder wrapper:
    - backbone: HF AutoModel
    - output: [B, D] mean-pooled embedding (mask-aware)
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    @property
    def emb_dim(self) -> int:
        return int(getattr(self.backbone.config, "hidden_size"))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B, L, D]
        mask = attention_mask.unsqueeze(-1).to(h.dtype)  # [B, L, 1]
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled  # [B, D]


def _apply_lora(backbone: nn.Module, lora_cfg: Dict[str, Any]) -> nn.Module:
    """
    Apply PEFT LoRA to a HF model.
    lora_cfg expected keys (examples):
      enabled: bool
      r: int
      alpha: int
      dropout: float
      target_modules: list[str]
      bias: str (optional, default "none")
      task_type: str (optional; for encoders we can omit)
    """
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PEFT (peft) is required for LoRA mode but is not available in this environment. "
            "Install it via `pip install peft`."
        ) from e

    if not lora_cfg.get("target_modules"):
        raise ValueError(
            "LoRA requires `encoder.lora.target_modules` to be a non-empty list. "
            "Please set it in your config."
        )

    bias = lora_cfg.get("bias", "none")
    # For encoder-style models, PEFT task_type is not strictly required for plain LoRA.
    # If you want, you can pass task_type=lora_cfg.get("task_type", None)
    lcfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg["target_modules"]),
        bias=bias,
    )
    return get_peft_model(backbone, lcfg)


def build_encoder(cfg: Dict[str, Any], device: torch.device) -> Tuple[Optional[nn.Module], int]:
    """
    Build encoder according to cfg["encoder"].

    Expected cfg structure (example):
      cfg["encoder"] = {
        "name": "answerdotai/ModernBERT-large",
        "train_mode": "frozen" | "full" | "lora",
        "lora": {"enabled": true, "r":16, "alpha":32, "dropout":0.05, "target_modules":[...]}
      }

    Returns:
      (encoder_module_or_None, emb_dim)

    Note:
      If you later want "cached embeddings" mode, you can set cfg["encoder"]["name"]=None
      and handle it upstream; here we return (None, emb_dim) only if emb_dim is provided.
    """
    enc_cfg = cfg.get("encoder", {})
    model_name = enc_cfg.get("name", None)

    # If no encoder name, assume upstream will provide embeddings.
    if not model_name:
        emb_dim = enc_cfg.get("emb_dim")
        if emb_dim is None:
            raise ValueError("encoder.name is None but encoder.emb_dim is not set.")
        return None, int(emb_dim)

    encoder = HFMeanPoolEncoder(model_name=model_name).to(device)
    emb_dim = encoder.emb_dim

    train_mode = (enc_cfg.get("train_mode") or "frozen").lower()
    if train_mode not in ("frozen", "full", "lora"):
        raise ValueError(f"Unknown encoder.train_mode: {train_mode}")

    if train_mode == "frozen":
        _freeze_module(encoder)
        encoder.eval()  # usually frozen encoder in eval mode
    elif train_mode == "full":
        _unfreeze_module(encoder)
        encoder.train()
    else:  # lora
        lora_cfg = enc_cfg.get("lora", {})
        if not lora_cfg.get("enabled", True):
            raise ValueError("encoder.train_mode is 'lora' but encoder.lora.enabled is false.")
        # Apply LoRA to the backbone only (keeps wrapper intact)
        encoder.backbone = _apply_lora(encoder.backbone, lora_cfg)
        _unfreeze_module(encoder)  # PEFT will set only LoRA params trainable; others frozen.
        encoder.train()

    return encoder, emb_dim


# -------------------------
# Predictors
# -------------------------
class ViTPredictor1D(nn.Module):
    """
    1D "ViT-like" predictor operating on pooled embeddings [B, D].

    It reshapes embedding into tokens: [B, T, patch], projects to d_model,
    runs TransformerEncoder, projects back to patch and flattens.
    """

    def __init__(
        self,
        dim: int,
        patch: int = 32,
        layers: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % patch != 0:
            raise ValueError(f"emb_dim={dim} must be divisible by patch={patch}")
        self.dim = dim
        self.patch = patch
        self.num_tokens = dim // patch

        self.token_proj = nn.Linear(patch, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.out = nn.Linear(dim, patch)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D]
        b, d = z.shape
        x = z.view(b, self.num_tokens, self.patch)  # [B, T, patch]
        x = self.token_proj(x)                      # [B, T, D]
        x = self.encoder(x)                         # [B, T, D]
        x = self.out(x)                             # [B, T, patch]
        return x.reshape(b, d)                      # [B, D]


class MLPredictor(nn.Module):
    """
    MLP predictor: [B, D] -> [B, D]

    Two styles supported:
    1) hidden_sizes list (your reference): e.g. [4096, 2048, 1024]
       => Linear(D->4096)->Act->Linear(4096->2048)->Act->Linear(2048->1024)->Act->Linear(1024->D)
    2) hidden + layers (compact): layers>=1
       => (layers-1) blocks of Linear->Act(+LN/Dropout) then final Linear->D
    """

    def __init__(
        self,
        dim: int,
        hidden_sizes: Optional[list[int]] = None,
        hidden: int = 2048,
        layers: int = 3,
        activation: str = "relu",     # "relu" | "gelu"
        dropout: float = 0.0,
        use_layernorm: bool = False,
        residual: bool = False,
        out_layernorm: bool = False,
    ):
        super().__init__()

        act = nn.ReLU() if activation.lower() == "relu" else nn.GELU()

        sizes = None
        if hidden_sizes is not None and len(hidden_sizes) > 0:
            sizes = [int(x) for x in hidden_sizes]
        else:
            if layers < 1:
                raise ValueError("MLPredictor layers must be >= 1")
            sizes = [hidden] * max(0, layers - 1)

        mods = []
        in_f = dim
        for h in sizes:
            mods.append(nn.Linear(in_f, h))
            if use_layernorm:
                mods.append(nn.LayerNorm(h))
            mods.append(act)
            if dropout and dropout > 0:
                mods.append(nn.Dropout(float(dropout)))
            in_f = h

        mods.append(nn.Linear(in_f, dim))
        self.net = nn.Sequential(*mods)

        self.residual = bool(residual)
        self.out_ln = nn.LayerNorm(dim) if out_layernorm else nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.net(z)
        if self.residual:
            y = y + z
        return self.out_ln(y)


def build_predictor(cfg: Dict[str, Any], emb_dim: int, device: torch.device) -> nn.Module:
    p_cfg = cfg.get("predictor", {})
    name = (p_cfg.get("name") or "vit1d").lower()

    if name == "vit1d":
        v = p_cfg.get("vit", {})
        m = ViTPredictor1D(
            dim=emb_dim,
            patch=int(v.get("patch", 32)),
            layers=int(v.get("layers", 4)),
            heads=int(v.get("heads", 8)),
            mlp_ratio=float(v.get("mlp_ratio", 4.0)),
            dropout=float(v.get("dropout", 0.0)),
        ).to(device)
        return m

    if name == "mlp":
        mcfg = p_cfg.get("mlp", {})
        # preferred: hidden_sizes list (matches your reference implementation)
        hidden_sizes = mcfg.get("hidden_sizes", None)
        if hidden_sizes is not None:
            # YAML might load it as list already; if string, try parse later upstream
            hidden_sizes = list(hidden_sizes)

        m = MLPredictor(
            dim=emb_dim,
            hidden_sizes=hidden_sizes,
            hidden=int(mcfg.get("hidden", 2048)),
            layers=int(mcfg.get("layers", 3)),
            activation=str(mcfg.get("activation", "relu")),
            dropout=float(mcfg.get("dropout", 0.0)),
            use_layernorm=bool(mcfg.get("use_layernorm", False)),
            residual=bool(mcfg.get("residual", False)),
            out_layernorm=bool(mcfg.get("out_layernorm", False)),
        ).to(device)
        return m

    raise ValueError(f"Unknown predictor.name: {name}")