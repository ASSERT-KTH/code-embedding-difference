from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineAlignLoss(nn.Module):
    """Mean cosine alignment loss."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.normalize(pred, dim=-1, eps=self.eps)
        target = F.normalize(target, dim=-1, eps=self.eps)
        return 1.0 - (pred * target).sum(dim=-1).mean()


class VarianceLoss(nn.Module):
    """Variance regularizer on the predictor-side pooled region embeddings."""

    def __init__(self, target_std: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.target_std = float(target_std)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(x.var(dim=0, unbiased=False) + self.eps)
        return F.relu(self.target_std - std).mean()


class CovarianceLoss(nn.Module):
    """Covariance regularizer on pooled region embeddings."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"CovarianceLoss expects [B, D], got {tuple(x.shape)}")
        batch_size, dim = x.shape
        if batch_size <= 1:
            return x.new_zeros(())

        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / float(batch_size - 1)
        off_diag = cov.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].flatten()
        return off_diag.pow(2).sum() / float(dim)


class RegionPoolLoss(nn.Module):
    """
    Region-pooled JEPA loss.

    Inputs:
      pred_region: [B, D]
      tgt_region:  [B, D]
    """

    def __init__(
        self,
        w_align: float = 1.0,
        w_var: float = 1.0,
        w_cov: float = 0.0,
        align_eps: float = 1e-8,
        var_target_std: float = 1.0,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        self.w_align = float(w_align)
        self.w_var = float(w_var)
        self.w_cov = float(w_cov)

        self.align = CosineAlignLoss(eps=align_eps)
        self.var = VarianceLoss(target_std=var_target_std, eps=var_eps)
        self.cov = CovarianceLoss()

    def forward(self, pred_region: torch.Tensor, tgt_region: torch.Tensor) -> Dict[str, torch.Tensor]:
        l_align = self.align(pred_region, tgt_region)
        l_var = self.var(pred_region)
        l_cov = self.cov(pred_region)
        loss = self.w_align * l_align + self.w_var * l_var + self.w_cov * l_cov
        return {
            "loss": loss,
            "align": l_align,
            "var": l_var,
            "cov": l_cov,
        }


def build_loss(cfg: Dict[str, Any]) -> RegionPoolLoss:
    lcfg = cfg.get("loss", {})
    return RegionPoolLoss(
        w_align=float(lcfg.get("w_align", 1.0)),
        w_var=float(lcfg.get("w_var", 1.0)),
        w_cov=float(lcfg.get("w_cov", 0.0)),
        align_eps=float(lcfg.get("align_eps", 1e-8)),
        var_target_std=float(lcfg.get("var_target_std", 1.0)),
        var_eps=float(lcfg.get("var_eps", 1e-4)),
    )
