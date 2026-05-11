from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from transformers import AutoModel


def infer_hidden_dim(model_name: str) -> int:
    model = AutoModel.from_pretrained(model_name)
    return int(getattr(model.config, "hidden_size"))


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def _named_trainable_params(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().to("cpu")
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def _copy_named_params_into_module(
    module: nn.Module,
    saved: Dict[str, torch.Tensor],
    strict: bool = False,
) -> None:
    current = dict(module.named_parameters())
    missing = []
    for name, tensor in saved.items():
        if name in current:
            current[name].data.copy_(tensor)
        elif strict:
            missing.append(name)
    if strict and missing:
        raise KeyError(f"Missing parameter names while loading: {missing[:20]}")


def _apply_lora(backbone: nn.Module, lora_cfg: Dict[str, Any]) -> nn.Module:
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PEFT is required for LoRA encoder mode. Install it with `pip install peft`."
        ) from e

    target_modules = list(lora_cfg.get("target_modules", []))
    if not target_modules:
        raise ValueError("LoRA mode requires a non-empty encoder.lora.target_modules list.")

    lora = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=target_modules,
        bias=str(lora_cfg.get("bias", "none")),
    )
    return get_peft_model(backbone, lora)


class HFSequenceEncoder(nn.Module):
    """
    Hugging Face encoder wrapper.

    Output:
      hidden: [B, L, D]
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    @property
    def hidden_dim(self) -> int:
        return int(getattr(self.backbone.config, "hidden_size"))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state


class LightweightTransformerBlock(nn.Module):
    """Pre-norm transformer block for sequence latent prediction."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(float(dropout))

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.dropout2 = nn.Dropout(float(dropout))

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        x = self.norm1(hidden)
        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = hidden + self.dropout1(attn_out)

        x = self.norm2(hidden)
        mlp_out = self.mlp(x)
        hidden = hidden + self.dropout2(mlp_out)
        return hidden


class SequencePredictor(nn.Module):
    """
    Lightweight sequence predictor.

    Modes:
      - direct:   z_pred = predictor(z_ctx)
      - residual: z_pred = z_ctx + residual_scale * delta
    """

    def __init__(
        self,
        hidden_dim: int,
        layers: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_residual: bool = True,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.use_residual = bool(use_residual)
        self.residual_scale = float(residual_scale)

        self.blocks = nn.ModuleList(
            [
                LightweightTransformerBlock(
                    hidden_dim=hidden_dim,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = hidden
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.final_norm(x)
        x = self.out_proj(x)

        if self.use_residual:
            return hidden + self.residual_scale * x
        return x


def build_encoder(cfg: Dict[str, Any], device: torch.device) -> Tuple[nn.Module, int]:
    enc_cfg = cfg.get("encoder", {})
    model_name = enc_cfg.get("name")
    if not model_name:
        raise ValueError("encoder.name must be set.")

    encoder = HFSequenceEncoder(model_name=model_name).to(device)
    hidden_dim = encoder.hidden_dim

    train_mode = str(enc_cfg.get("train_mode", "frozen")).lower()
    if train_mode not in {"frozen", "full", "lora"}:
        raise ValueError(f"Unknown encoder.train_mode: {train_mode}")

    if train_mode == "frozen":
        _freeze_module(encoder)
        encoder.eval()
    elif train_mode == "full":
        _unfreeze_module(encoder)
        encoder.train()
    else:
        encoder.backbone = _apply_lora(encoder.backbone, enc_cfg.get("lora", {}))
        _unfreeze_module(encoder)
        encoder.train()

    return encoder, hidden_dim


def build_predictor(cfg: Dict[str, Any], hidden_dim: int, device: torch.device) -> nn.Module:
    pred_cfg = cfg.get("predictor", {})
    predictor = SequencePredictor(
        hidden_dim=hidden_dim,
        layers=int(pred_cfg.get("layers", 4)),
        heads=int(pred_cfg.get("heads", 8)),
        mlp_ratio=float(pred_cfg.get("mlp_ratio", 4.0)),
        dropout=float(pred_cfg.get("dropout", 0.1)),
        use_residual=bool(pred_cfg.get("use_residual", True)),
        residual_scale=float(pred_cfg.get("residual_scale", 1.0)),
    ).to(device)
    return predictor


def export_encoder_tunable_state(module: nn.Module) -> Dict[str, torch.Tensor]:
    return _named_trainable_params(module)


def load_encoder_tunable_state(
    module: nn.Module,
    state: Dict[str, torch.Tensor],
    strict: bool = False,
) -> None:
    _copy_named_params_into_module(module, state, strict=strict)
