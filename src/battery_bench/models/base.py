"""Model interface + registry for the SOC method comparison.

Every model — classical (sklearn) or deep (torch) — implements the SAME small
interface so the runner is model-agnostic:

    model.fit(X_tr, y_tr, X_val, y_val, cfg) -> self
    model.predict(X) -> y_hat               # 1-D np.ndarray
    model.name : str                        # set by @register
    model.n_params : int                    # parameter count

Input windows are float32 arrays of shape (N, window_len, n_features) with
features ordered (voltage_V, current_A, temperature_C); the target is SOC at the
window's LAST timestep (many-to-one regression). Models are instantiated by
string through the registry, configured at fit() time via the ``cfg`` dict.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

MODEL_REGISTRY: dict[str, type["BaseModel"]] = {}


def register(name: str):
    """Class decorator: register a model class under ``name`` and stamp .name."""
    def deco(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"model '{name}' already registered")
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls
    return deco


def build_model(name: str, **kwargs) -> "BaseModel":
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'; known: {list_models()}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


class BaseModel(ABC):
    name: str = "OVERRIDE_ME"

    @abstractmethod
    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray, cfg: dict) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def n_params(self) -> int:
        ...
