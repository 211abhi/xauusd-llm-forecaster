"""Wraps a frozen LLM to extract text embeddings for regime descriptions."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict


class TextEncoder(nn.Module):
    """
    Extracts frozen text embeddings from a pre-trained LLM.

    Used only during Phase 2 (encoder contrastive training). Not used at inference.
    """

    def __init__(self, model_name: str = "gpt2-medium", cache_dir: str = ".cache/llm",
                 device: str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], max_length: int = 64) -> torch.Tensor:
        """
        Encode a list of strings into mean-pooled embeddings.

        Returns: (N, hidden_dim) tensor on self.device
        """
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.model(**enc)
        # Mean pool over token dimension (masked)
        hidden = outputs.last_hidden_state         # (B, seq_len, D)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled

    def precompute_regime_embeddings(self, templates: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Precompute and cache embeddings for all regime text templates."""
        return {regime: self.encode([text]).squeeze(0)
                for regime, text in templates.items()}
