"""Phase 2 validation gate tests.

All 4 tests must pass before Phase 3 (LSTM/Set Transformer).

Run with:
    pytest morningalpha/ml/tests/test_baselines.py -v

Skipped if model checkpoint not found.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PARQUET_PATH = Path("data/training/dataset.parquet")
MODEL_DIR = Path.home() / ".morningalpha" / "models"
FEAT_CONFIG_PATH = MODEL_DIR / "feature_config.json"

pytestmark = pytest.mark.slow

requires_dataset = pytest.mark.skipif(
    not PARQUET_PATH.exists(),
    reason="data/training/dataset.parquet not found — run `alpha ml dataset` first",
)


def _find_lgbm_checkpoint():
    if not MODEL_DIR.exists():
        return None
    pkls = sorted(MODEL_DIR.glob("lgbm_*.pkl"))
    return pkls[-1] if pkls else None


requires_model = pytest.mark.skipif(
    _find_lgbm_checkpoint() is None,
    reason="No lgbm_*.pkl checkpoint found — run `alpha ml train` first",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df():
    return pd.read_parquet(PARQUET_PATH)


@pytest.fixture(scope="module")
def feat_cols():
    if FEAT_CONFIG_PATH.exists():
        with open(FEAT_CONFIG_PATH) as f:
            return json.load(f)["feature_columns"]
    from morningalpha.ml.features import FEATURE_COLUMNS
    df = pd.read_parquet(PARQUET_PATH)
    return [c for c in FEATURE_COLUMNS if c in df.columns]


@pytest.fixture(scope="module")
def model():
    from morningalpha.ml.baselines import LightGBMModel
    ckpt = _find_lgbm_checkpoint()
    return LightGBMModel.load(str(ckpt))


@pytest.fixture(scope="module")
def X_test(df, feat_cols):
    test = df[df["split"] == "test"][feat_cols].fillna(0)
    return test.values.astype(np.float32)


@pytest.fixture(scope="module")
def y_test(df):
    return df[df["split"] == "test"]["forward_10d"].values.astype(np.float32)


@pytest.fixture(scope="module")
def X_train(df, feat_cols):
    tr = df[df["split"] == "train"][feat_cols].fillna(0)
    return tr.values.astype(np.float32)


@pytest.fixture(scope="module")
def y_train(df):
    return df[df["split"] == "train"]["forward_10d"].values.astype(np.float32)


# ---------------------------------------------------------------------------
# Gate 1 — LightGBM beats persistence baseline
# ---------------------------------------------------------------------------

@requires_dataset
@requires_model
def test_lgbm_beats_persistence(model, df, feat_cols, X_test, y_test):
    """LightGBM rank IC must exceed persistence baseline (return_pct → forward_10d) on test set."""
    from morningalpha.ml.train import rank_ic

    lgbm_ic = model.rank_ic(X_test, y_test)

    test_df = df[df["split"] == "test"]
    if "return_pct" in test_df.columns:
        from scipy.stats import spearmanr
        persist_ic = float(spearmanr(test_df["return_pct"].fillna(0), test_df["forward_10d"]).correlation)
    else:
        persist_ic = 0.02  # conservative floor

    assert lgbm_ic > persist_ic, (
        f"LightGBM IC={lgbm_ic:.4f} <= persistence IC={persist_ic:.4f}. "
        "Model is not beating the persistence baseline."
    )


# ---------------------------------------------------------------------------
# Gate 2 — Output shape
# ---------------------------------------------------------------------------

@requires_dataset
@requires_model
def test_output_shape(model, X_test):
    """model.predict(X) must return shape (n_samples,)."""
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),), (
        f"Expected shape ({len(X_test)},), got {preds.shape}"
    )


# ---------------------------------------------------------------------------
# Gate 3 — SHAP values
# ---------------------------------------------------------------------------

@requires_dataset
@requires_model
def test_shap_values(model, X_test, feat_cols):
    """SHAP values shape must match X, and sum + expected_value ≈ prediction."""
    sample = X_test[:min(100, len(X_test))]
    sv = model.shap_values(sample)

    assert sv.shape == sample.shape, (
        f"SHAP shape {sv.shape} != X shape {sample.shape}"
    )

    # Additive property: sum(shap_values[i]) + expected_value ≈ prediction[i]
    # (within floating point tolerance)
    import shap as shap_lib
    explainer = shap_lib.TreeExplainer(model.model)
    expected_val = float(explainer.expected_value)
    preds = model.predict(sample)
    shap_sums = sv.sum(axis=1) + expected_val

    max_err = float(np.abs(shap_sums - preds).max())
    assert max_err < 1e-3, f"SHAP additive property violated: max error = {max_err:.6f}"


# ---------------------------------------------------------------------------
# Gate 4 — Save / load round-trip
# ---------------------------------------------------------------------------

@requires_dataset
@requires_model
def test_model_save_load(model, X_test, tmp_path):
    """Saved and reloaded model must produce identical predictions."""
    from morningalpha.ml.baselines import LightGBMModel

    ckpt_path = tmp_path / "lgbm_test.pkl"
    model.save(str(ckpt_path))

    reloaded = LightGBMModel.load(str(ckpt_path))
    original_preds = model.predict(X_test[:50])
    reloaded_preds = reloaded.predict(X_test[:50])

    np.testing.assert_array_almost_equal(
        original_preds, reloaded_preds, decimal=6,
        err_msg="Reloaded model predictions differ from original."
    )
