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
        self.proj_head = proj_head.to(device)
        pred_head = pred_head.to(device)
        if torch.cuda.device_count() > 1:
            print(f"CMA-ES: using {torch.cuda.device_count()} GPUs for pred_head")
            self.pred_head = torch.nn.DataParallel(pred_head)
        else:
            self.pred_head = pred_head
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
    def _cache_encoder(self, val_loader: DataLoader):
        """Pre-compute encoder projections for all val batches — done once."""
        cached_proj, cached_targets = [], []
        for patches, targets in val_loader:
            ts_proj = self.proj_head(self.encoder(patches.to(self.device))).unsqueeze(1)
            cached_proj.append(ts_proj.cpu())
            cached_targets.append(targets)
        return cached_proj, cached_targets

    @torch.no_grad()
    def _evaluate(self, z: np.ndarray, cached_proj: list, cached_targets: list) -> float:
        """Evaluate a candidate using pre-cached encoder projections."""
        self.soft_prompt.set_from_numpy(self._subspace_to_prompt(z).astype(np.float32))

        total_mae = 0.0
        for ts_proj, targets in zip(cached_proj, cached_targets):
            ts_proj = ts_proj.to(self.device)
            targets = targets.to(self.device)
            B = ts_proj.size(0)
            inputs = torch.cat([self.soft_prompt(B), ts_proj], dim=1)
            preds = self.pred_head(self.llm.get_hidden_state(inputs))
            total_mae += (preds - targets).abs().mean().item()

        return total_mae / max(1, len(cached_proj))

    def optimize(self, val_loader: DataLoader, checkpoint_path: str) -> np.ndarray:
        """Run CMA-ES optimization. Returns best prompt array."""
        from pathlib import Path

        print("Pre-caching encoder outputs (once)...")
        cached_proj, cached_targets = self._cache_encoder(val_loader)
        print(f"Cached {len(cached_proj)} batches.")

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
            fitnesses = [self._evaluate(z, cached_proj, cached_targets) for z in solutions]
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
                np.save(checkpoint_path.replace(".npy", "_latest.npy"),
                        self._subspace_to_prompt(best_z))

            if patience_counter >= self.patience:
                print(f"CMA-ES early stop at generation {generation}")
                break

            generation += 1

        return self._subspace_to_prompt(best_z)
