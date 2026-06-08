"""Deep models behind the BaseModel interface.

A single ``TorchRegressor`` implements the training loop (Adam + MSE, minibatch,
early stopping on val, grad clipping, per-run seeding, device selection) once;
each concrete model only supplies a ``build_module`` that returns the nn.Module.
Training hyperparameters and per-model architecture knobs both come from the
``cfg`` dict passed to fit().
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn

from .base import BaseModel, register
from . import nets


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(cfg: dict) -> torch.device:
    want = str(cfg.get("device", "auto")).lower()
    if want in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TorchRegressor(BaseModel):
    """Shared trainer. Subclasses implement build_module(n_features, window_len, cfg)."""

    def __init__(self):
        self.module: nn.Module | None = None
        self.device: torch.device | None = None
        self._n_params = 0

    # --- to be provided by each concrete model ---
    def build_module(self, n_features: int, window_len: int, cfg: dict) -> nn.Module:
        raise NotImplementedError

    # --- shared training loop ---
    def fit(self, X_tr, y_tr, X_val, y_val, cfg):
        seed_everything(int(cfg.get("seed", 0)))
        self.device = resolve_device(cfg)
        _, window_len, n_features = X_tr.shape
        self.module = self.build_module(n_features, window_len, cfg).to(self.device)
        self._n_params = int(sum(p.numel() for p in self.module.parameters()))
        if cfg.get("verbose", False):
            print(f"    [{self.name}] device={self.device.type} params={self._n_params:,}")

        lr = float(cfg.get("lr", 1e-3))
        batch = int(cfg.get("batch_size", 256))
        max_epochs = int(cfg.get("max_epochs", 50))
        patience = int(cfg.get("patience", 8))
        clip = float(cfg.get("grad_clip", 1.0))

        Xtr = torch.as_tensor(X_tr, dtype=torch.float32)
        ytr = torch.as_tensor(y_tr, dtype=torch.float32)
        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            Xva = torch.as_tensor(X_val, dtype=torch.float32).to(self.device)
            yva = torch.as_tensor(y_val, dtype=torch.float32).to(self.device)

        opt = torch.optim.Adam(self.module.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)))

        best_val, best_state, bad = float("inf"), None, 0
        n = len(Xtr)
        for _ in range(max_epochs):
            self.module.train()
            perm = torch.randperm(n, generator=gen)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                xb = Xtr[idx].to(self.device)
                yb = ytr[idx].to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.module(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.module.parameters(), clip)
                opt.step()

            if not has_val:
                continue
            self.module.eval()
            with torch.no_grad():
                vloss = float(loss_fn(self._forward_batched(Xva), yva))
            if vloss < best_val - 1e-6:
                best_val, best_state, bad = vloss, copy.deepcopy(self.module.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            self.module.load_state_dict(best_state)
        return self

    def _forward_batched(self, X: torch.Tensor, batch: int = 1024) -> torch.Tensor:
        outs = [self.module(X[s:s + batch]) for s in range(0, len(X), batch)]
        return torch.cat(outs) if outs else torch.empty(0, device=self.device)

    def predict(self, X):
        self.module.eval()
        Xt = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self._forward_batched(Xt)
        return out.detach().cpu().numpy().ravel()

    @property
    def n_params(self) -> int:
        return self._n_params


# --------------------------------------------------------------------------
# Concrete registered models. Architecture knobs read from cfg with defaults.
# --------------------------------------------------------------------------
@register("mlp")
class MLP(TorchRegressor):
    def build_module(self, n_features, window_len, cfg):
        return nets.MLPNet(n_features, window_len,
                           hidden=tuple(cfg.get("hidden", (256, 128))),
                           dropout=float(cfg.get("dropout", 0.1)))


@register("lstm")
class LSTM(TorchRegressor):
    def build_module(self, n_features, window_len, cfg):
        return nets.RNNNet(n_features, window_len, kind="lstm",
                           hidden=int(cfg.get("hidden", 96)),
                           num_layers=int(cfg.get("num_layers", 1)),
                           bidirectional=False, dropout=float(cfg.get("dropout", 0.0)))


@register("gru")
class GRU(TorchRegressor):
    def build_module(self, n_features, window_len, cfg):
        return nets.RNNNet(n_features, window_len, kind="gru",
                           hidden=int(cfg.get("hidden", 96)),
                           num_layers=int(cfg.get("num_layers", 1)),
                           bidirectional=False, dropout=float(cfg.get("dropout", 0.0)))


@register("bilstm")
class BiLSTM(TorchRegressor):
    def build_module(self, n_features, window_len, cfg):
        return nets.RNNNet(n_features, window_len, kind="lstm",
                           hidden=int(cfg.get("hidden", 96)),
                           num_layers=int(cfg.get("num_layers", 1)),
                           bidirectional=True, dropout=float(cfg.get("dropout", 0.0)))


@register("cnn_bilstm_attn")
class CNNBiLSTMAttn(TorchRegressor):
    def build_module(self, n_features, window_len, cfg):
        return nets.CNNBiLSTMAttn(n_features, window_len,
                                  filters=int(cfg.get("filters", 16)),
                                  hidden=int(cfg.get("hidden", 64)),
                                  dropout=float(cfg.get("dropout", 0.1)))


@register("tcn")
class TCN(TorchRegressor):
    def build_module(self, n_features, window_len, cfg):
        return nets.TCNNet(n_features, window_len,
                           channels=int(cfg.get("channels", 32)),
                           levels=int(cfg.get("levels", 3)),
                           kernel=int(cfg.get("kernel", 3)),
                           dropout=float(cfg.get("dropout", 0.1)))


@register("transformer")
class Transformer(TorchRegressor):
    def build_module(self, n_features, window_len, cfg):
        return nets.TransformerNet(n_features, window_len,
                                   d_model=int(cfg.get("d_model", 64)),
                                   nhead=int(cfg.get("nhead", 4)),
                                   num_layers=int(cfg.get("num_layers", 2)),
                                   dim_feedforward=int(cfg.get("dim_feedforward", 128)),
                                   dropout=float(cfg.get("dropout", 0.1)))


@register("patchtst")
class PatchTST(TorchRegressor):
    def build_module(self, n_features, window_len, cfg):
        return nets.PatchTSTNet(n_features, window_len,
                                patch_len=int(cfg.get("patch_len", 10)),
                                patch_stride=int(cfg.get("patch_stride", 10)),
                                d_model=int(cfg.get("d_model", 64)),
                                nhead=int(cfg.get("nhead", 4)),
                                num_layers=int(cfg.get("num_layers", 2)),
                                dim_feedforward=int(cfg.get("dim_feedforward", 128)),
                                dropout=float(cfg.get("dropout", 0.1)))
