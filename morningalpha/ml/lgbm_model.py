"""LightGBM model wrapper — no PyTorch dependency.

Kept in a separate module so it can be imported (and pickled/unpickled)
in environments where torch is not installed (e.g. GitHub Actions CI).
"""
import pickle

import lightgbm as lgb
import numpy as np
import shap as shap_lib
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge


class LightGBMModel:
    """LightGBM regressor wrapper with SHAP support."""

    DEFAULT_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "n_estimators": 1000,
        "verbose": -1,
    }

    def __init__(self, params=None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: lgb.LGBMRegressor = None
        self._shap_explainer = None

    def fit(self, X_train, y_train, X_val, y_val):
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )
        self._shap_explainer = None  # reset on refit
        return self

    def predict(self, X):
        return self.model.predict(X)

    def shap_values(self, X):
        if self._shap_explainer is None:
            self._shap_explainer = shap_lib.TreeExplainer(self.model)
        return self._shap_explainer.shap_values(X)

    def rank_ic(self, X, y_true):
        preds = self.predict(X)
        return float(spearmanr(preds, y_true).correlation)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class RidgeModel:
    """Ridge regression wrapper for feature importance analysis."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.feature_names_ = None

    def fit(self, X_train, y_train, feature_names=None):
        self.model.fit(X_train, y_train)
        self.feature_names_ = feature_names
        return self

    def predict(self, X):
        return self.model.predict(X)

    def rank_ic(self, X, y_true):
        preds = self.predict(X)
        return float(spearmanr(preds, y_true).correlation)

    def feature_importance_series(self):
        import pandas as pd
        names = self.feature_names_ or [f"f{i}" for i in range(len(self.model.coef_))]
        return (
            pd.Series(self.model.coef_, index=names)
            .reindex(pd.Series(self.model.coef_, index=names).abs().sort_values(ascending=False).index)
        )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
