"""CMA-ES optimizer for soft prompt using BBTv2 subspace trick."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Callable, Optional
import cma

from src.soft_prompt.soft_prompt import SoftPrompt
from src.encoder.ts_encoder import TSEncoder
from src.llm.frozen_llm import FrozenLLM
from src.prediction.pred_head import PredictionHead
from src.encoder.encoder_trainer import ProjectionHead


class CMAESOptimizer:
    """
    Optimizes soft prompt via CMA-ES in a low-dimensional subspace (BBTv2 trick).

    The full soft_prompt is (32 × 768) = 24,576 dims.
    We optimize in `subspace_dim` dims and project back via a fixed random matrix.
    """

    def __init__(self, soft_prompt: SoftPrompt, encoder: TSEncoder, llm: FrozenLLM,
                 pred_head: PredictionHead, proj_head: ProjectionHead,
                 cfg: dict, device: torch.device) -> None:
        self.soft_prompt = soft_prompt.to(device)
        self.encoder = encoder.to(device)
        self.llm = llm
        self.pred_head = pred_head.to(device)
        self.proj_head = proj_head.to(device)
        self.cfg = cfg
        self.device = device

        cc = cfg["cmaes"]
        self.subspace_dim = cc["subspace_dim"]
        self.full_dim = soft_prompt.n_tokens * soft_prompt.token_dim
        self.popsize = cc["popsize"]
        self.sigma0 = cc["sigma0"]
        self.maxiter = cc["maxiter"]
        self.patience = cc["early_stop_patience"]
        self.eval_batch_size = cc["eval_batch_size"]

        # Fixed random projection matrix: subspace → full_dim
        rng = np.random.RandomState(42)
        self.proj_matrix = rng.randn(self.subspace_dim, self.full_dim).astype(np.float32)
        # Normalize columns
        norms = np.linalg.norm(self.proj_matrix, axis=0, keepdims=True).clip(min=1e-9)
        self.proj_matrix /= norms

        for m in [self.encoder, self.pred_head, self.proj_head]:
            for p in m.parameters():
                p.requires_grad = False
        self.encoder.eval()
        self.pred_head.eval()
        self.proj_head.eval()

    def _subspace_to_prompt(self, z: np.ndarray) -> np.ndarray:
        """Map subspace vector (subspace_dim,) → prompt array (n_tokens, token_dim)."""
        full = z @ self.proj_matrix                        # (full_dim,)
        return full.reshape(self.soft_prompt.n_tokens, self.soft_prompt.token_dim)

    @torch.no_grad()
    def _evaluate(self, z: np.ndarray, val_loader: DataLoader) -> float:
        """Evaluate a candidate subspace vector; return val MAE."""
        prompt_arr = self._subspace_to_prompt(z)
        self.soft_prompt.set_from_numpy(prompt_arr.astype(np.float32))

        total_mae = 0.0
        n_batches = 0
        for patches, targets in val_loader:
            patches = patches.to(self.device)
            targets = targets.to(self.device)
            B = patches.size(0)

            ts_embed = self.encoder(patches)           # (B, 256)
            ts_proj = self.proj_head(ts_embed)         # (B, 768)
            ts_proj = ts_proj.unsqueeze(1)             # (B, 1, 768)

            soft = self.soft_prompt(B)                 # (B, 32, 768)
            inputs = torch.cat([soft, ts_proj], dim=1) # (B, 33, 768)

            hidden = self.llm.get_hidden_state(inputs) # (B, 768)
            preds = self.pred_head(hidden)              # (B, N)

            mae = (preds - targets).abs().mean().item()
            total_mae += mae
            n_batches += 1

        return total_mae / max(1, n_batches)

    def optimize(self, val_loader: DataLoader, checkpoint_path: str) -> np.ndarray:
        """Run CMA-ES optimization. Returns best prompt array."""
        from pathlib import Path

        x0 = np.zeros(self.subspace_dim, dtype=np.float32)
        opts = {
            "popsize": self.popsize,
            "maxiter": self.maxiter,
            "tolx": 1e-6,
            "tolfun": 1e-6,
            "verbose": -9,
        }
        es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)

        best_loss = float("inf")
        best_z = x0.copy()
        patience_counter = 0

        generation = 0
        while not es.stop():
            solutions = es.ask()
            fitnesses = [self._evaluate(z, val_loader) for z in solutions]
            es.tell(solutions, fitnesses)

            gen_best = min(fitnesses)
            if gen_best < best_loss:
                best_loss = gen_best
                best_z = solutions[np.argmin(fitnesses)].copy()
                patience_counter = 0
                # Save checkpoint
                best_arr = self._subspace_to_prompt(best_z)
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(checkpoint_path, best_arr)
            else:
                patience_counter += 1

            if generation % 10 == 0:
                print(f"Gen {generation:4d} | best_mae={best_loss:.6f} | sigma={es.sigma:.6f}")

            if patience_counter >= self.patience:
                print(f"CMA-ES early stop at generation {generation}")
                break

            generation += 1

        return self._subspace_to_prompt(best_z)
