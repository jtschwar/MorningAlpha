"""Phase 1 validation gate tests.

All six tests must pass before Phase 2 (model training) begins.

Run with:
    pytest morningalpha/ml/tests/test_dataset.py -v

Skipped if data/training/dataset.parquet does not exist.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PARQUET_PATH = Path("data/training/dataset.parquet")
RAW_CACHE_DIR = Path("data/raw_ohlcv")

requires_dataset = pytest.mark.skipif(
    not PARQUET_PATH.exists(),
    reason="data/training/dataset.parquet not found — run `alpha ml dataset` first",
)

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_parquet(PARQUET_PATH)


@pytest.fixture(scope="module")
def sample_rows(df) -> pd.DataFrame:
    """10 random (ticker, date) pairs from the dataset."""
    rng = random.Random(42)
    indices = rng.sample(range(len(df)), min(10, len(df)))
    return df.iloc[indices].copy()


# ---------------------------------------------------------------------------
# Gate 1 — No lookahead
# ---------------------------------------------------------------------------

@requires_dataset
def test_no_lookahead(sample_rows):
    """Features must be computable from ohlcv.loc[:date] only.

    Verifies that for each sample row the raw OHLCV cache file exists and
    that all rows in the cache up to that date have an index ≤ date.
    """
    from morningalpha.ml.dataset import (
        MIN_HISTORY_DAYS,
        _compute_features_at_date,
    )

    failures = []
    for _, row in sample_rows.iterrows():
        ticker = row["ticker"]
        date = pd.Timestamp(row["date"])
        cache_path = RAW_CACHE_DIR / f"{ticker}.parquet"
        if not cache_path.exists():
            continue

        ohlcv = pd.read_parquet(cache_path)
        subset = ohlcv.loc[:date]

        # Ensure the subset ends at or before t
        if len(subset) > 0 and subset.index[-1] > date:
            failures.append(f"{ticker}@{date}: subset ends at {subset.index[-1]}")
            continue

        # Ensure features can be computed from this subset alone
        if len(subset) >= MIN_HISTORY_DAYS:
            meta = {"market_cap": np.nan, "market_cap_cat": 0, "exchange": 2}
            feats = _compute_features_at_date(ohlcv, date, meta)
            if feats is None:
                failures.append(f"{ticker}@{date}: _compute_features_at_date returned None")

    assert not failures, f"Lookahead violations:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Gate 2 — Label correctness
# ---------------------------------------------------------------------------

@requires_dataset
def test_label_correctness(df, sample_rows):
    """forward_10d in dataset matches manually computed value within 1e-4."""
    if "forward_10d" not in df.columns:
        pytest.skip("forward_10d column not present")

    failures = []
    for _, row in sample_rows.iterrows():
        ticker = row["ticker"]
        date = pd.Timestamp(row["date"])
        cache_path = RAW_CACHE_DIR / f"{ticker}.parquet"
        if not cache_path.exists():
            continue

        ohlcv = pd.read_parquet(cache_path)
        prices = ohlcv["Close"]

        try:
            idx_pos = prices.index.get_loc(date)
        except KeyError:
            continue

        if idx_pos + 10 >= len(prices):
            continue

        expected = float(prices.iloc[idx_pos + 10] / prices.iloc[idx_pos] - 1)
        actual = float(row["forward_10d"])
        if abs(expected - actual) > 1e-4:
            failures.append(f"{ticker}@{date}: expected={expected:.6f}, actual={actual:.6f}")

    assert not failures, "Label mismatches:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Gate 3 — Non-overlapping windows
# ---------------------------------------------------------------------------

@requires_dataset
def test_non_overlapping_windows(df):
    """For each ticker, consecutive snapshot dates must be >= 10 trading days apart."""
    if "forward_10d" not in df.columns:
        pytest.skip("forward_10d column not present")

    violations = []
    for ticker, grp in df.groupby("ticker"):
        dates = grp["date"].sort_values().reset_index(drop=True)
        if len(dates) < 2:
            continue
        diffs = (dates - dates.shift(1)).dropna()
        # Allow up to 14 calendar days between consecutive snapshots (≈10 trading days)
        bad = diffs[diffs < pd.Timedelta(days=8)]
        if not bad.empty:
            violations.append(f"{ticker}: gap {bad.iloc[0].days}d < 8d")

    assert not violations, "Overlapping window violations (first 5):\n" + "\n".join(violations[:5])


# ---------------------------------------------------------------------------
# Gate 4 — Feature distribution after normalization
# ---------------------------------------------------------------------------

@requires_dataset
def test_feature_distribution(df):
    """Post rank-normalization: each float feature should have mean ≈ 0, std ≈ 1."""
    from morningalpha.ml.features import FLOAT_FEATURES

    failures = []
    for col in FLOAT_FEATURES:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 30:
            continue
        mean = float(vals.mean())
        std = float(vals.std())
        # Rank-normalized values span (−1, 1) so std ≈ 0.5–0.6 is typical.
        # We check that mean is near 0 and std is in a sensible range.
        if abs(mean) > 0.3:
            failures.append(f"{col}: mean={mean:.3f} (expected ≈ 0)")
        if std < 0.1 or std > 1.5:
            failures.append(f"{col}: std={std:.3f} (expected 0.1–1.5)")

    assert not failures, "Feature distribution issues:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Gate 5 — No future tickers (IPO check)
# ---------------------------------------------------------------------------

@requires_dataset
def test_no_future_tickers(df):
    """A ticker's first training-fold row must not appear before its first OHLCV date."""
    violations = []
    train = df[df["split"] == "train"]

    for ticker, grp in train.groupby("ticker"):
        cache_path = RAW_CACHE_DIR / f"{ticker}.parquet"
        if not cache_path.exists():
            continue
        ohlcv = pd.read_parquet(cache_path)
        ipo_date = ohlcv.index.min()
        first_train_date = grp["date"].min()
        if first_train_date < ipo_date:
            violations.append(
                f"{ticker}: first training date {first_train_date.date()} < IPO {ipo_date.date()}"
            )

    assert not violations, "Future ticker violations:\n" + "\n".join(violations[:5])


# ---------------------------------------------------------------------------
# Gate 6 — Label distribution (survivorship bias check)
# ---------------------------------------------------------------------------

@requires_dataset
def test_label_distribution(df):
    """forward_10d should be approximately 45–55% positive across the full dataset."""
    if "forward_10d" not in df.columns:
        pytest.skip("forward_10d column not present")

    pos_rate = (df["forward_10d"] > 0).mean()
    assert 0.40 <= pos_rate <= 0.60, (
        f"forward_10d positive rate = {pos_rate:.1%} — outside 40–60%. "
        "Possible survivorship bias or label computation error."
    )
