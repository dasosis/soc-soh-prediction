"""Classical (sklearn) baselines behind the BaseModel interface."""
from __future__ import annotations

import numpy as np

from .base import BaseModel, register


@register("linear")
class LinearRidge(BaseModel):
    """Ridge regression on the flattened window — the performance floor.

    Flattens (N, window_len, n_features) -> (N, window_len*n_features) and fits a
    single linear map to SOC. ``cfg['alpha']`` sets the L2 strength (default 1.0).
    """

    def __init__(self):
        self._model = None

    def fit(self, X_tr, y_tr, X_val, y_val, cfg):
        from sklearn.linear_model import Ridge
        self._model = Ridge(alpha=float(cfg.get("alpha", 1.0)))
        self._model.fit(X_tr.reshape(len(X_tr), -1), y_tr)
        return self

    def predict(self, X):
        return np.asarray(self._model.predict(X.reshape(len(X), -1)), dtype=float).ravel()

    @property
    def n_params(self) -> int:
        if self._model is None:
            return 0
        return int(np.asarray(self._model.coef_).size + 1)  # +1 intercept
