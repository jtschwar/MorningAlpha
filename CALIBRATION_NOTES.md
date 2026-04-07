# Calibration & Fusion Pipeline — Implementation Notes

## What We Built

### Overview

A walk-forward calibration pipeline that:
1. Tracks every daily model prediction against future realized returns
2. Monitors live IC per model per horizon
3. Fits residual correction models to reduce systematic bias
4. Fits a cross-model calibration model to produce `P(positive return)` signals
5. Dynamically reweights the ensemble via the Hedge algorithm

---

### Data Files (all in `data/factors/`)

| File | Size | Purpose |
|---|---|---|
| `predictions_ledger.parquet` | ~20MB | Append-only record of every scored prediction. One row per ticker × scoring date. Stores raw model scores, context features, eval dates, and realized returns once matured. |
| `live_ic_timeseries.parquet` | ~12KB | Cross-sectional Spearman IC per (model, horizon, cohort). Ground truth on live predictive power. |
| `model_health.json` | ~2KB | Per-model rolling IC summaries + alerts (replaces old per-model `live_ic_*.json` files). |
| `correction_models.joblib` | ~2.5KB | Dict of fitted Ridge regression models keyed by `{model_id}_{horizon}`. Applied at inference to remove systematic bias. |
| `calibration_models.joblib` | ~1.9KB | Dict of fitted LogisticRegression models keyed by horizon (`5d`, `21d`, `63d`). Converts corrected scores to `P(positive return)`. |
| `model_weights.json` | ~1.4KB | Dynamic ensemble weights updated by the Hedge algorithm. |

---

### Predictions Ledger Schema

Each row represents one scoring event for one ticker:

```
ticker, scored_date, price_at_score
eval_after_5d, eval_after_21d, eval_after_63d
raw_lgbm_breakout_v10, raw_lgbm_composite_v10, ...  (one column per active model)
sector_code, momentum_bucket, market_return_21d, vix_at_prediction
realized_return_5d, realized_return_21d, realized_return_63d
matured_5d, matured_21d, matured_63d
is_backfill
```

Realized returns are filled in on future scoring runs once the `eval_after_*` date passes.

---

### Daily Scoring Flow (`alpha ml score`)

1. **Score** — All active models score `stocks_3m.csv`. Raw scores are rank-percentiled 0–100.
2. **Weight update** — Reads matured rows from ledger; computes IC per model; runs Hedge algorithm to update ensemble weights in `model_weights.json`.
3. **Ensemble** — Combines model scores using dynamic weights → `MLScore`.
4. **Sector diversity cap** — Re-ranks to enforce per-sector slot limits (e.g. Tech ≤ 15, Utilities ≤ 2).
5. **Load correction + calibration** — Reads `correction_models.joblib` and `calibration_models.joblib` once.
6. **Residual correction** — Applies Ridge correction per model (adds a small bias term based on sector, VIX, market return, momentum). Clipped to ±10%.
7. **Calibration** — LogisticRegression converts corrected multi-model scores → `CalibratedProb` (0–1). Mapped to signals: `STRONG BUY` (≥70%), `BUY` (≥55%), `HOLD` (≥45%), `SELL` (≥30%), `STRONG SELL` (<30%).
8. **Append ledger** — Today's raw scores + context appended for future IC evaluation.

---

### Daily Calibration Flow (`alpha ml score --calibrate`, runs in GitHub Actions)

1. **IC timeseries** — Computes cross-sectional IC for every matured (model, horizon, cohort) and appends to `live_ic_timeseries.parquet`.
2. **Model health** — Checks rolling 10-cohort IC per model. Writes `model_health.json`. Alerts on negative IC (CRITICAL) or 63d IC < 0.02 (WARNING).
3. **Re-fit correction models** — Ridge regression per (model, horizon) on last 3000 resolved rows. Features: `[raw_score, sector_code, market_return_21d, vix_at_prediction, momentum_bucket]`. Saved to `correction_models.joblib`.
4. **Re-fit calibration models** — LogisticRegression per horizon on all resolved rows. Features: multi-model score vector + pairwise agreement + context. Saved to `calibration_models.joblib`.

---

### Cold-Start Solution: `alpha ml backfill`

The calibration models need ~500 resolved predictions to activate. Waiting for 63d resolution would mean ~13 weeks before any calibration.

**Solution:** `alpha ml backfill` reads `data/training/dataset.parquet` (local only, ~500MB, never pushed to GitHub), scores every date after each model's `train_cutoff`, and pre-populates the ledger with the realized `forward_5d/21d/63d` returns that are already in the dataset. Rows are marked `is_backfill=True`.

After backfill: ~940K rows in the ledger with 93%/63%/12% maturity for 5d/21d/63d. All 3 calibration models activated immediately.

---

### Why All Models Score on `stocks_3m.csv`

The 4 LGBM models were **all trained on the same feature matrix** from `dataset.parquet`. The "medium" vs "long" distinction is in the **training target** (21d vs 63d forward return rank), not the input features. All 4 models share the same 77-feature set.

The `stocks_3m.csv` naming refers to how a handful of period-dependent features (`return_pct`, `sharpe_ratio`, `max_drawdown`) are computed in the spread pipeline. The remaining ~70 features — RSI, SMA ratios, momentum_12_1, volatility_20d, MACD, VIX regime, fundamentals — are fixed-window and are **identical across all period CSVs**. The 3m window was chosen as a reasonable default for the period-dependent features (enough history for Sharpe to be meaningful).

---

## Two Options for Richer Calibration

### Current State

Calibration and correction models use only 5 context features stored at prediction time:
```
raw_score, sector_code, market_return_21d, vix_at_prediction, momentum_bucket
```

This is intentionally lightweight but leaves signal on the table — the models can't learn things like "high-RSI biotech in high-VIX conditions = overconfident score."

---

### Option A — Keep Current Lightweight Ledger

Leave the ledger schema as-is. The 5-feature calibration is a simple bias-correction layer that handles:
- Cross-model agreement weighting
- Macro regime conditioning (VIX, market return)
- Sector-level adjustment

**Pros:**
- No schema changes, already working
- Ledger stays small (~20MB), comfortable on GitHub
- Calibration models are interpretable

**Cons:**
- Can't learn stock-level overconfidence patterns (momentum crowding, RSI extremes)
- Can't distinguish between "model is confident because RSI is extended" vs "model is confident because fundamentals are strong"

---

### Option B — Snapshot a Richer Feature Set at Prediction Time

At each scoring run, store a broader set of features per ticker alongside the raw score. This would live in a separate `calibration_features.parquet` (or extend the ledger), assembled from backfill today and appended daily.

**Proposed additional features to snapshot:**

| Feature | Why it helps calibration |
|---|---|
| `rsi` | Overbought/oversold — high scores on RSI > 80 are more likely to mean-revert |
| `momentum_12_1` | Momentum crowding — works until it doesn't |
| `price_to_sma200` | Trend regime — above/below 200d MA changes signal reliability |
| `volatility_20d` | Uncertainty — calibration should widen with volatility |
| `earnings_yield` | Valuation context — cheap vs expensive stocks behave differently post-signal |
| `short_pct_float` | Short interest — high short = squeeze risk, different return distribution |
| `bollinger_pct_b` | Where in the BB range — extremes are less reliable |
| `market_cap_cat` | Size regime — small caps have noisier signals |

This gives the calibration model enough context to learn regime-conditional reliability — essentially "when should I trust this score and when should I discount it."

**Pros:**
- Substantially better calibration, especially in volatile markets
- Can detect momentum crowding, overbought signals, sector bubbles
- Same backfill approach applies — `dataset.parquet` has all these features

**Cons:**
- Ledger grows larger (roughly 3–4× if adding ~8 float features per row)
- At ~940K rows × 8 features, still manageable (~30–40MB total)
- Requires a schema migration for existing ledger rows (backfill rows can be re-extracted; live rows from before the change would have NaN for the new columns)

---

## Recommended Next Step

Option B is the better long-term investment if retraining is on the roadmap anyway. The feature snapshot is cheap to add and the backfill path already exists. The main cost is a one-time ledger migration (re-run `alpha ml backfill` after adding the new columns).

If the priority is stability while waiting for LSTM v3, Option A is fine — the current calibration is already live and producing reasonable signals.
