"""Feature standardization with FIT-ON-TRAIN-ONLY discipline.

Fit the Standardizer on the training rows, then transform train AND test with
the same stats. Persisting the stats means evaluation is reproducible and you
can prove no test statistics leaked into normalization.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class Standardizer:
    def __init__(self, columns: list[str]):
        self.columns = list(columns)
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> "Standardizer":
        for c in self.columns:
            col = train_df[c].astype(float)
            self.mean_[c] = float(col.mean())
            s = float(col.std(ddof=0))
            self.std_[c] = s if s > 1e-12 else 1.0
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("call fit() on TRAIN data before transform()")
        out = df.copy()
        for c in self.columns:
            out[c] = (out[c].astype(float) - self.mean_[c]) / self.std_[c]
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {"columns": self.columns, "mean": self.mean_, "std": self.std_},
                f, indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Standardizer":
        with open(path) as f:
            d = json.load(f)
        obj = cls(d["columns"])
        obj.mean_, obj.std_, obj._fitted = d["mean"], d["std"], True
        return obj
