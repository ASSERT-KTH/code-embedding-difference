# losses.py
# -*- coding: utf-8 -*-
"""
Losses for JEPA-style training.

Exp2 minimal set:
- cosine alignment: align(z_pred, z_tgt)
- variance regularizer (VICReg variance term): avoid collapse without negatives

Also provide simple training-time metrics:
- retrieval_top1_acc: batch retrieval acc z_pred -> z_tgt
- emb_std_mean: mean std across embedding dimensions (collapse indicator)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Core losses
# -------------------------
class CosineAlignLoss(nn.Module):
    """1 - cosine similarity (mean over batch)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, z_pred: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
        z_pred = F.normalize(z_pred, dim=-1, eps=self.eps)
        z_tgt = F.normalize(z_tgt, dim=-1, eps=self.eps)
        return 1.0 - (z_pred * z_tgt).sum(dim=-1).mean()


class VarianceLoss(nn.Module):
    """
    VICReg variance term:
      std = sqrt(var + eps)
      loss = mean( relu(target_std - std) )
    """

    def __init__(self, target_std: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.target_std = float(target_std)
        self.eps = float(eps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D]
        std = torch.sqrt(z.var(dim=0, unbiased=False) + self.eps)
        return F.relu(self.target_std - std).mean()


class EmaPredictiveLoss(nn.Module):
    """
    Minimal JEPA-style loss for exp2 (EMA target + predictor):

      L = w_align * align(z_pred, z_tgt) + w_var * (var(z_ctx) + var(z_tgt))
    """
    def __init__(
        self,
        w_align: float = 1.0,
        w_var: float = 1.0,
        align_eps: float = 1e-8,
        var_target_std: float = 1.0,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        self.w_align = float(w_align)
        self.w_var = float(w_var)
        self.align = CosineAlignLoss(eps=align_eps)
        self.var = VarianceLoss(target_std=var_target_std, eps=var_eps)

    def forward(self, z_ctx, z_pred, z_tgt):
        l_align = self.align(z_pred, z_tgt)
        l_var = self.var(z_ctx) + self.var(z_tgt)
        loss = self.w_align * l_align + self.w_var * l_var
        return {"loss": loss, "align": l_align, "var": l_var}


def build_loss(cfg: Dict[str, Any]) -> EmaPredictiveLoss:
    lcfg = cfg.get("loss", {})
    return EmaPredictiveLoss(
        w_align=float(lcfg.get("w_align", 1.0)),
        w_var=float(lcfg.get("w_var", 1.0)),
        align_eps=float(lcfg.get("align_eps", 1e-8)),
        var_target_std=float(lcfg.get("var_target_std", 1.0)),
        var_eps=float(lcfg.get("var_eps", 1e-4)),
    )


# -------------------------
# Metrics (no grad)
# -------------------------
@torch.no_grad()
def retrieval_top1_acc(z_pred: torch.Tensor, z_tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Batch retrieval accuracy:
      For each i, find argmax_j cos(z_pred[i], z_tgt[j])
      count how many i match j=i
    AMP-safe: cast to float32 for stable metric computation.
    """
    z_pred = z_pred.float()
    z_tgt = z_tgt.float()

    z_pred = F.normalize(z_pred, dim=-1, eps=eps)
    z_tgt = F.normalize(z_tgt, dim=-1, eps=eps)

    sim = z_pred @ z_tgt.t()  # [B, B]
    pred_idx = sim.argmax(dim=1)
    gt_idx = torch.arange(sim.size(0), device=sim.device)
    return (pred_idx == gt_idx).float().mean()


@torch.no_grad()
def emb_std_mean(z: torch.Tensor) -> torch.Tensor:
    z = z.float()
    return z.std(dim=0, unbiased=False).mean()