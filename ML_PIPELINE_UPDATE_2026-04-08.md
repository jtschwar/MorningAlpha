# ML Pipeline Update — 2026-04-08

## What's Been Built Since Last Handoff

The last major milestone was the v7 LGBM champions (breakout IC 0.314, composite IC 0.240).
Since then, the pipeline has grown substantially:

---

## 1. New Models

### LGBM v10 (Champion + Candidates)
Retrained on an extended dataset through 2025-12-07, walk-forward CV through fold 95.

| Model | Target | 5d IC | 21d IC | 63d IC | Status |
|---|---|---|---|---|---|
| lgbm_breakout_v10 | raw return rank | +0.073 | +0.157 | **+0.315** | Champion |
| lgbm_composite_v10 | quality-adj rank | +0.042 | +0.115 | +0.194 | Candidate |
| lgbm_breakout_medium_v1 | raw return rank (21d) | +0.054 | +0.120 | — | Candidate |
| lgbm_composite_medium_v1 | composite rank (21d) | +0.053 | +0.124 | — | Candidate |

All IC figures are from backfill (historical) data in the predictions ledger.
No live predictions have matured yet — the pipeline only went live 2026-04-07.

### LSTM v3 (lstm_clip_v3) — Just Promoted
Trained on H100 at HPC. Walk-forward CV (6 folds) + 53-anchor sequential fine-tuning,
dataset through 2026-03-26. ~7.2 hours of training.

| Metric | Value |
|---|---|
| Architecture | h256-L2, 907K params, 76 features |
| WFCV IC-63d | +0.1343 (mean across 6 folds) |
| Best fold (fold 2, 2019–2020) | IC-63d = +0.360 |
| Weakest fold (fold 6, 2024–2026) | IC-63d = +0.001 |
| Final val loss | 0.000876 |
| Target mode | clip (±2.0) |

The fold 6 near-zero IC is a current-regime problem — the 2024–2026 period
(tariff shock, elevated VIX) is hard to predict for all models. The 53-anchor
fine-tuning adapts the model to the current regime sequentially.

---

## 2. Calibration & Correction Pipeline

A full online calibration system was implemented:

### Components
- **`predictions_ledger.parquet`** — 1,000,620 rows tracking every prediction with realized
  returns at 5d / 21d / 63d horizons. Backfill (Dec 2025 → Mar 2026) + 2 days of live data.
- **`calibration_daily.parquet`** — rolling 63-day window of spread features (snake_case ML
  names), 277,591 rows. Updated daily. Serves both LSTM inference and rich calibration context.
- **`correction_models.joblib`** — per-model Ridge regression residual correction (10 models,
  5 LGBM × 2 horizons each). Fit on ledger, applied at score time.
- **`calibration_models.joblib`** — cross-model LogisticRegression calibration at 5d / 21d /
  63d horizons. Outputs P(return > 0) → maps to STRONG BUY / BUY / HOLD / SELL / STRONG SELL.
- **`model_weights.json`** — hedge algorithm weights, updated each scoring run from 63d IC.

### Current Ensemble Weights
| Model | Weight |
|---|---|
| lgbm_breakout_v10 | 0.640 |
| lgbm_composite_v10 | 0.228 |
| lgbm_breakout_medium_v1 | 0.039 |
| lgbm_composite_medium_v1 | 0.039 |
| lstm_clip_v3 | 0.039 (floor — no history yet) |

LSTM weight will move up or down as live predictions mature (~5 weeks for first 5d read).

---

## 3. LGBM vs LSTM: Signal Comparison (Today, 2026-04-08)

### Correlation
LSTM is nearly orthogonal to the LGBM models — this is the best possible outcome
for an ensemble member:

| | lgbm_breakout | lgbm_composite | lgbm_breakout_med | lgbm_composite_med |
|---|---|---|---|---|
| lstm_clip_v3 | **0.148** | **0.076** | 0.187 | 0.191 |
| lgbm_breakout vs composite | 0.806 | — | — | — |

### Sector Bias
The two model families identify completely different stock universes:

| Rank | LGBM top-100 | LSTM top-100 |
|---|---|---|
| 1 | Financial Services (50%) | Healthcare (28%) |
| 2 | Industrials (11%) | Basic Materials (11%) |
| 3 | Healthcare (8%) | Technology (11%) |
| 4 | Technology (6%) | Financial Services (10%) |

LGBM favors value/dividend-paying sectors with stable fundamentals.
LSTM leans toward Healthcare and small-cap names — the known biotech bias
(clip targets amplify large movers, biotech has the largest absolute moves).

### Top-50 Overlap
Only **1 ticker** (IMMR) appears in both the LGBM top-50 and LSTM top-50.

### Consensus Picks (both LGBM > 97 AND LSTM > 80)
High-conviction names where both independent signals agree:
IMPP, ELE, MFIN, CMCSA, IMMR, ORMP, SAFE, BHRB, ESEA, UHS, GPN

---

## 4. Current Challenge: Calibration Signal Collapse

### Symptom
Today's calibration output: **STRONG BUY=6 / BUY=172 / HOLD=283 / SELL=1073 / STRONG SELL=1466**

~84% of stocks are SELL or STRONG SELL — on a day when NASDAQ/NYSE are up ~2%.

### Root Cause
The calibration model predicts **absolute** P(return > 0) using recent market context
(market_return_21d, vix_at_prediction). The ledger's recent backfill data reflects the
tariff shock selloff:

| Period | 5d % positive | 21d % positive |
|---|---|---|
| Dec 2025 | 51.2% | 65.3% |
| Jan 2026 | 54.1% | 56.1% |
| Feb 2026 | 47.3% | 31.1% |
| Mar 2026 | **34.5%** | — |

With only 34% of March predictions being positive, and current VIX elevated, the
model correctly learns that P(return > 0) is low in this environment. So it maps
nearly everything below the 0.45 HOLD threshold.

This is **technically correct** but **practically useless**: if everything is SELL,
you can't differentiate which stocks to hold vs avoid in the current regime.

### The Design Problem
The calibration model answers: *"Will this stock have a positive absolute return?"*

What we actually want: *"Will this stock outperform its peers?"*

These are different questions. In a bear market, a stock that falls 2% while peers
fall 10% is a good pick — but the current calibration would signal STRONG SELL on
both.

### Options

**Option A — Percentile thresholds (quick fix)**
Convert CalibratedProb to cross-sectional percentile rank within today's universe.
Top 10% → STRONG BUY, top 30% → BUY, middle 40% → HOLD, etc.
Pros: one-line code change, regime-invariant by design.
Cons: always outputs the same distribution regardless of whether it's actually a
good time to buy — loses the "don't buy anything right now" signal.

**Option B — Relative return target (proper fix)**
Retrain calibration on `return > market_return` (outperformance) instead of `return > 0`.
This means the model learns: "given these scores and market context, will this stock
beat the market?" Naturally centers at 50% HOLD in any regime.
Pros: answers the right question, preserves regime sensitivity for relative picks.
Cons: requires refitting calibration models; existing backfill data has `realized_return`
as absolute return, so we'd need to subtract the market return for the same date.

**Option C — Hybrid: relative signal + absolute gate**
Use relative calibration (Option B) for the BUY/SELL ranking, but add a separate
binary "is the market environment investable at all?" gate. E.g., suppress all BUY
signals if market_return_21d < -10% and VIX > 35.

---

## 5. Infrastructure Fixes Made Today

| Issue | Fix |
|---|---|
| lstm_lstm_clip_v3.pt double prefix | Renamed to lstm_clip_v3.pt |
| New model defaulted to weight=1.0 | `_load_model_weights` now defaults to `min(known weights)` |
| LGBM-only calibration filter | Removed — LSTM re-enabled in calibration pipeline |
| calibration_models.joblib sklearn 1.8 vs 1.7 mismatch | Re-fit and committed fresh models |
| lstm_clip_v1 in model_weights.json | Replaced with lstm_clip_v3 at same floor weight |

---

## 6. What We Don't Know Yet

- **Live LSTM IC**: lstm_clip_v3 has zero live predictions matured. First 5d read ~2026-04-14.
- **Live LGBM IC**: Pipeline went live 2026-04-07. All current IC figures are backfill only.
- **LSTM biotech bias magnitude**: Visible in sector distribution but not yet quantified in IC terms.
- **Calibration accuracy**: No live calibration accuracy data yet — CalibratedSignal has
  never been validated against real forward returns.

---

## 7. Next Steps

1. **Decide on calibration fix**: Option A (percentile) is shippable today.
   Option B (relative return) is the principled approach but requires a refit.
2. **Monitor first live IC reads**: ~2026-04-14 for 5d, ~2026-04-28 for 21d.
3. **LSTM biotech fix (deferred)**: Retraining with `--target-mode rank` would reduce
   the Healthcare overweight. Deprioritized for now — the low ensemble weight (0.039)
   limits the damage while the model proves itself.
4. **Rebuild dataset**: Current dataset ends 2026-03-26. The next LGBM retraining cycle
   should incorporate April 2026 data (tariff shock regime) for better current-regime fit.
