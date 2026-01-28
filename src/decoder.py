# decoder.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from typing import List


class SoftPromptStarCoderDecoder(nn.Module):
    def __init__(self, cond_dim: int, decoder_model, tokenizer, prompt_len: int = 32, device: str = "cuda"):
        super().__init__()
        self.decoder = decoder_model
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.hidden_dim = decoder_model.config.hidden_size
        self.device = device

        inter_dim = cond_dim * 4
        self.prompt_proj = nn.Sequential(
            nn.Linear(cond_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(inter_dim, prompt_len * self.hidden_dim),
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

        prompt_mask = torch.ones(B, self.prompt_len, device=self.device, dtype=attention_mask.dtype)
        full_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        full_labels = torch.full((B, self.prompt_len + T), -100, device=self.device, dtype=torch.long)
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

        p_mask = torch.ones(B, self.prompt_len, device=cond_emb.device, dtype=torch.long)

        generated_ids = self.decoder.generate(
            inputs_embeds=prompt,
            attention_mask=p_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def save_tunable_parameters(model: nn.Module, path: str):
    saved_params = {k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad}
    torch.save(saved_params, path)
    print(f"  >>> [Saved] Tunable params only -> {path} ({len(saved_params)} keys)")


def load_tunable_parameters(model: nn.Module, path: str):
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  >>> [Loaded decoder tunable params] <- {path}")
    if len(unexpected) > 0:
        print(f"  [Warn] unexpected keys: {unexpected[:5]} ...")
    if len(missing) > 0:
        print(f"  [Info] missing keys (expected): {missing[:5]} ...")
