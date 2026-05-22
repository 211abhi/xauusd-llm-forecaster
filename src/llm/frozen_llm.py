"""Frozen LLM wrapper — exposes hidden state extraction only. No weight updates ever."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional


class FrozenLLM(nn.Module):
    """
    Loads a causal LLM, freezes all parameters, and exposes hidden state extraction.

    No text decoding. No vocabulary projection. Hidden states only.
    """

    def __init__(self, model_name: str = "gpt2-medium", cache_dir: str = ".cache/llm",
                 hidden_dim: int = 768, device: str = "cpu") -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device_str = device
        _device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            output_hidden_states=True,
        )
        self.model.to(_device)

        # Freeze everything — verified below
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        assert not any(p.requires_grad for p in self.model.parameters()), \
            "LLM has unfrozen parameters — abort."

    @classmethod
    def from_config(cls, cfg: dict) -> "FrozenLLM":
        return cls(
            model_name=cfg["llm"]["model_name"],
            cache_dir=cfg["llm"]["cache_dir"],
            hidden_dim=cfg["llm"]["hidden_dim"],
            device=cfg["project"]["device"],
        )

    @torch.no_grad()
    def get_hidden_state(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-computed embeddings; return last-layer CLS hidden state.

        Args:
            inputs_embeds: (B, seq_len, hidden_dim) — soft_prompt + ts_embed concat
        Returns:
            (B, hidden_dim) — last hidden state at position 0 (CLS)
        """
        outputs = self.model(inputs_embeds=inputs_embeds, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]   # (B, seq_len, D)
        return last_hidden[:, 0, :]               # (B, D) — CLS position

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the model's token embedding layer (for soft-prompt init reference)."""
        return self.model.get_input_embeddings()
