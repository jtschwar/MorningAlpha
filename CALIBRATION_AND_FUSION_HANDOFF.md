# Adaptive Calibration & Model Fusion — Claude Code Handoff

**Date:** 2026-04-07  
**Status:** Implement after LSTM v3 training + LGBM v10 production refit + LSTM inference workflow  
**Depends on:** Predictions ledger accumulating daily scores from all champion models

---

## Overview

This doc covers four interconnected systems that transform raw model scores into
adaptive, calibrated signals for the dashboard:

1. **Daily IC Monitoring** — real-time model health tracking
2. **Multi-Horizon Residual Correction** — learn from recent prediction errors
3. **Learned Calibration Model** — replace hedge weights with a model that fuses
   LGBM + LSTM scores, learns nonlinear interactions, and adapts to regime shifts
4. **Frontend Integration** — surface calibrated probabilities, model agreement,
   and forecast confidence on the dashboard

All four systems read from and write to the predictions ledger. They run as
sequential steps within `alpha ml score` (daily GitHub Actions pipeline).

---

## Architecture

```
alpha ml score (daily, 4:15pm ET)
│
├── Step 1: Score all tickers with all champion models (EXISTING)
│   └── Write raw scores to ticker_index.json + predictions_ledger.parquet
│
├── Step 2: Fill realized returns for past predictions (EXISTING)
│   └── Look back 5d, 21d, 63d and fill actuals in ledger
│
├── Step 3: Daily IC Measurement (NEW)
│   └── Compute per-model per-horizon IC from resolved predictions
│   └── Write to live_ic_timeseries.parquet
│   └── Check alert thresholds
│
├── Step 4: Residual Correction (NEW)
│   └── Fit correction model on recent resolved residuals
│   └── Apply corrections to today's raw scores
│
├── Step 5: Learned Calibration Model (NEW — replaces hedge + isotonic)
│   └── Fit calibration model on all resolved predictions
│   └── Combine corrected scores → calibrated probabilities
│   └── Write calibrated signals to ticker_index.json
│
└── Step 6: LSTM Forecast Paths (EXISTING)
    └── Generate fan charts with calibration-adjusted confidence bands
```

---

## Part 1: Daily IC Monitoring

### What It Does

Every day, compute how well each model's past predictions matched actual outcomes.
For each model and each horizon, we look at predictions made N days ago and compare
to today's realized returns.

### Implementation

Add to `alpha ml score`, after step 2 (filling realized returns):

```python
import pandas as pd
from scipy.stats import spearmanr

def compute_daily_ic(
    ledger: pd.DataFrame,
    today: pd.Timestamp,
    horizons: list[int] = [5, 21, 63],
) -> pd.DataFrame:
    """
    For each model and horizon, compute the cross-sectional rank IC
    between predictions made N days ago and today's realized returns.
    
    Returns one row per (model_id, horizon) with IC and metadata.
    """
    results = []
    
    for horizon in horizons:
        prediction_date = today - pd.offsets.BDay(horizon)
        
        # Get predictions made on that date
        preds = ledger[
            (ledger['date'] == prediction_date) &
            (ledger[f'actual_{horizon}d'].notna())  # only resolved outcomes
        ]
        
        if len(preds) < 50:  # minimum sample size for meaningful IC
            continue
        
        for model_id in preds['model_id'].unique():
            model_preds = preds[preds['model_id'] == model_id]
            
            ic, pvalue = spearmanr(
                model_preds['raw_score'],
                model_preds[f'actual_{horizon}d']
            )
            
            hit_rate = (
                (model_preds['raw_score'] > model_preds['raw_score'].median()) ==
                (model_preds[f'actual_{horizon}d'] > 0)
            ).mean()
            
            results.append({
                'date': today,
                'prediction_date': prediction_date,
                'model_id': model_id,
                'horizon': horizon,
                'ic': ic,
                'hit_rate': hit_rate,
                'n_tickers': len(model_preds),
                'market_return': _get_spy_return(today, horizon),  # SPY return over same period
            })
    
    return pd.DataFrame(results)
```

### Storage

```
data/factors/live_ic_timeseries.parquet

Columns:
  date              datetime64   — date IC was computed
  prediction_date   datetime64   — date the predictions were originally made
  model_id          str          — e.g. "lgbm_breakout_v10"
  horizon           int          — 5, 21, or 63
  ic                float64      — Spearman rank IC
  hit_rate          float64      — fraction of correct direction calls
  n_tickers         int          — number of stocks in the computation
  market_return     float64      — SPY return over the same horizon (context)
```

Append daily. Never overwrite — this is a time series log.

### Alert Thresholds

After computing daily IC, check for problems:

```python
ALERT_THRESHOLDS = {
    'warning': {
        'rolling_window': 10,        # 10 trading days
        'min_ic': 0.02,              # per-horizon IC
        'horizons': [63],            # only alert on primary target
    },
    'critical': {
        'rolling_window': 10,
        'min_ic': 0.0,               # negative IC = model is harmful
        'horizons': [5, 21, 63],     # alert on any horizon
    },
}

def check_alerts(ic_timeseries: pd.DataFrame, model_id: str) -> list[str]:
    alerts = []
    recent = ic_timeseries[ic_timeseries['model_id'] == model_id].tail(10)
    
    for horizon in [5, 21, 63]:
        horizon_ic = recent[recent['horizon'] == horizon]['ic']
        if len(horizon_ic) < 5:
            continue
        rolling_mean = horizon_ic.mean()
        
        if rolling_mean < 0.0:
            alerts.append(
                f"CRITICAL: {model_id} {horizon}d rolling IC = {rolling_mean:.4f} "
                f"(negative for {len(horizon_ic)} days)"
            )
        elif horizon == 63 and rolling_mean < 0.02:
            alerts.append(
                f"WARNING: {model_id} 63d rolling IC = {rolling_mean:.4f} "
                f"(below 0.02 threshold)"
            )
    
    return alerts
```

Write alerts to `data/factors/model_alerts.json` so the dashboard can display them.
Structure:

```json
{
  "last_checked": "2026-04-07",
  "alerts": [
    {
      "level": "WARNING",
      "model_id": "lstm_clip_v3",
      "message": "63d rolling IC = 0.018 (below 0.02 threshold)",
      "date": "2026-04-07"
    }
  ],
  "model_health": {
    "lgbm_breakout_v10": { "5d_ic": 0.12, "21d_ic": 0.18, "63d_ic": 0.25, "status": "healthy" },
    "lgbm_composite_v10": { "5d_ic": 0.08, "21d_ic": 0.14, "63d_ic": 0.19, "status": "healthy" },
    "lstm_clip_v3": { "5d_ic": 0.05, "21d_ic": 0.03, "63d_ic": 0.018, "status": "warning" }
  }
}
```

---

## Part 2: Multi-Horizon Residual Correction

### What It Does

The base models make systematic errors that shift with market regimes. The
residual correction layer learns these errors from recently resolved predictions
and patches tomorrow's scores.

The key insight: 5-day predictions resolve every week, giving fast feedback.
Even though the primary target is 63d, the 5d residuals tell you HOW the model
is miscalibrated right now — and that information transfers across horizons
because the errors stem from the same regime shift.

### How Residuals Flow In

```
Day 0:   Model predicts all tickers for 5d, 21d, 63d horizons
Day 5:   5d predictions resolve → compute residuals → update correction
Day 21:  21d predictions resolve → compute residuals → update correction
Day 63:  63d predictions resolve → compute residuals → update correction
```

Each resolved batch provides thousands of (prediction, actual) pairs across the
full ticker universe. The correction model is refitted daily on the most recent
resolved residuals.

### Implementation

```python
from sklearn.linear_model import Ridge

def fit_residual_correction(
    ledger: pd.DataFrame,
    model_id: str,
    horizon: int,
    lookback_pairs: int = 3000,
) -> Ridge:
    """
    Fit a lightweight correction model on recent residuals.
    
    The correction predicts: actual_rank - raw_score
    given: (raw_score, sector, market_return, vix, momentum_bucket)
    
    Applied as: corrected_score = raw_score + correction.predict(features)
    """
    # Get resolved predictions for this model + horizon
    resolved = ledger[
        (ledger['model_id'] == model_id) &
        (ledger[f'actual_{horizon}d'].notna())
    ].sort_values('date', ascending=False).head(lookback_pairs)
    
    if len(resolved) < 200:  # minimum sample size
        return None  # not enough data, skip correction
    
    # Compute residuals
    resolved = resolved.copy()
    resolved['residual'] = resolved[f'actual_{horizon}d'] - resolved['raw_score']
    
    # Correction features (must be available at prediction time)
    correction_features = [
        'raw_score',
        'sector_code',           # ordinal sector encoding
        'market_return_21d',     # SPY return over trailing 21d at prediction time
        'vix_at_prediction',     # VIX level when prediction was made
        'momentum_bucket',       # quintile of stock's trailing momentum
    ]
    
    X = resolved[correction_features].fillna(0)
    y = resolved['residual']
    
    correction_model = Ridge(alpha=10.0)  # heavy regularization — correction should be small
    correction_model.fit(X, y)
    
    return correction_model


def apply_correction(
    raw_scores: pd.DataFrame,
    correction_model: Ridge,
    correction_features: list[str],
) -> pd.Series:
    """Apply residual correction to today's raw scores."""
    if correction_model is None:
        return raw_scores['raw_score']  # no correction during cold start
    
    X = raw_scores[correction_features].fillna(0)
    corrections = correction_model.predict(X)
    
    # Clip corrections to prevent wild swings (max ±0.1 rank adjustment)
    corrections = np.clip(corrections, -0.1, 0.1)
    
    return raw_scores['raw_score'] + corrections
```

### What to Store in the Ledger

The ledger needs additional columns to support correction features. Update the
ledger schema:

```
data/factors/predictions_ledger.parquet

Existing columns:
  date              datetime64
  ticker            str
  model_id          str
  model_version     str          ← NEW: e.g. "lgbm_breakout_v10" vs "lgbm_breakout_v11"
  raw_score         float64
  horizon           int

New columns for correction features (recorded at prediction time):
  sector_code       int8         ← ordinal sector of the ticker
  market_return_21d float64      ← SPY trailing 21d return at prediction time
  vix_at_prediction float64      ← VIX close on prediction date
  momentum_bucket   int8         ← quintile of ticker's trailing 63d momentum

New columns for realized outcomes (filled in later):
  actual_5d         float64      ← filled 5 trading days later
  actual_21d        float64      ← filled 21 trading days later
  actual_63d        float64      ← filled 63 trading days later
  
New columns for calibrated output:
  corrected_score   float64      ← raw_score + residual correction
  calibrated_prob   float64      ← output of calibration model (Part 3)

Backfill flag:
  is_backfill       bool         ← True if this row was generated by `alpha ml backfill`
```

**Important:** `model_version` is distinct from `model_id`. When v10 is promoted to
v11, predictions from v10 and v11 are tracked separately. The correction model
resets on promotion (old model's residuals don't apply to new model).

**Important:** `is_backfill = True` rows are excluded from calibration model training
and IC computation. Backfilled predictions were made with hindsight and would inflate
apparent accuracy.

---

## Part 3: Learned Calibration Model (Replaces Hedge + Isotonic)

### What It Does

Instead of blending model scores with scalar hedge weights and then applying
isotonic calibration per model, train a single calibration model that:

1. Takes ALL model scores as input (LGBM breakout, LGBM composite, LSTM)
2. Learns nonlinear interactions (model agreement boosts confidence)
3. Conditions on market regime (VIX, recent market return)
4. Outputs a calibrated probability: P(positive return over horizon H)

### Why This Replaces Hedge Weights

The hedge algorithm learns one scalar weight per model. The calibration model
learns how to combine scores contextually:

- When both LGBM and LSTM agree (both > 0.8): probability is 85%
- When they disagree (LGBM 0.9, LSTM 0.3): probability is only 50%
- When VIX is high: all probabilities shift down
- When LGBM is confident but LSTM is missing: use LGBM alone with wider uncertainty

A linear blend can't capture these interactions. The calibration model can.

### Implementation

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

def build_calibration_features(ledger_row: pd.Series) -> dict:
    """
    Build the feature vector for the calibration model from a ledger row.
    All features must be available at prediction time.
    """
    score_lgbm_breakout = ledger_row.get('score_lgbm_breakout', np.nan)
    score_lgbm_composite = ledger_row.get('score_lgbm_composite', np.nan)
    score_lstm = ledger_row.get('score_lstm', np.nan)
    
    # Handle missing models (e.g., LSTM not scoring during cache gap)
    has_lstm = not np.isnan(score_lstm)
    
    return {
        'score_lgbm_breakout': score_lgbm_breakout,
        'score_lgbm_composite': score_lgbm_composite,
        'score_lstm': score_lstm if has_lstm else 0.5,  # neutral default
        'has_lstm': float(has_lstm),                      # flag for missing model
        'agreement_breakout_lstm': (
            score_lgbm_breakout * score_lstm if has_lstm
            else score_lgbm_breakout * 0.5
        ),
        'spread_breakout_lstm': (
            abs(score_lgbm_breakout - score_lstm) if has_lstm
            else 0.0
        ),
        'agreement_breakout_composite': (
            score_lgbm_breakout * score_lgbm_composite
        ),
        'market_return_21d': ledger_row['market_return_21d'],
        'vix_at_prediction': ledger_row['vix_at_prediction'],
        'sector_code': ledger_row['sector_code'],
    }


def fit_calibration_model(
    ledger: pd.DataFrame,
    horizon: int,
    min_samples: int = 500,
    rolling_window: int = 3000,
) -> LogisticRegression | None:
    """
    Fit calibration model on resolved predictions.
    
    Returns None during cold-start (not enough resolved predictions).
    
    The model learns: P(actual return > 0) given all model scores + regime features.
    """
    # Get resolved predictions with all model scores
    resolved = _pivot_to_wide_format(ledger, horizon)  # see helper below
    
    # Exclude backfilled rows — they have hindsight bias
    resolved = resolved[~resolved['is_backfill']]
    
    # Apply rolling window
    resolved = resolved.sort_values('date', ascending=False).head(rolling_window)
    
    if len(resolved) < min_samples:
        return None  # cold-start: not enough data
    
    # Build features
    feature_rows = resolved.apply(build_calibration_features, axis=1)
    X = pd.DataFrame(feature_rows.tolist())
    
    # Binary target: did the stock have positive return?
    y = (resolved[f'actual_{horizon}d'] > 0).astype(int)
    
    # Logistic regression — interpretable, fast, regularized
    model = LogisticRegression(
        C=1.0,               # regularization strength
        max_iter=1000,
        class_weight='balanced',  # handle imbalanced bull/bear periods
    )
    model.fit(X, y)
    
    return model


def _pivot_to_wide_format(ledger: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Pivot the ledger from long format (one row per model per ticker per day)
    to wide format (one row per ticker per day, with columns for each model's score).
    
    Only includes rows where the actual return for the given horizon has resolved.
    """
    resolved = ledger[ledger[f'actual_{horizon}d'].notna()].copy()
    
    # Pivot model scores to columns
    scores_wide = resolved.pivot_table(
        index=['date', 'ticker'],
        columns='model_id',
        values='raw_score',
        aggfunc='first'
    ).reset_index()
    
    # Rename columns to score_* prefix
    score_cols = {col: f'score_{col}' for col in scores_wide.columns 
                  if col not in ['date', 'ticker']}
    scores_wide = scores_wide.rename(columns=score_cols)
    
    # Merge back with actuals and context features
    context = resolved[['date', 'ticker', f'actual_{horizon}d',
                        'market_return_21d', 'vix_at_prediction', 
                        'sector_code', 'is_backfill']].drop_duplicates(
                            subset=['date', 'ticker'])
    
    return scores_wide.merge(context, on=['date', 'ticker'])


def apply_calibration(
    today_scores: pd.DataFrame,
    calibration_model: LogisticRegression | None,
    horizon: int,
) -> pd.Series:
    """
    Apply calibration model to today's raw scores.
    Returns calibrated probabilities. Falls back to raw rank percentile
    during cold-start.
    """
    if calibration_model is None:
        # Cold-start fallback: raw score IS the "probability" (uncalibrated)
        return today_scores['raw_score']
    
    feature_rows = today_scores.apply(build_calibration_features, axis=1)
    X = pd.DataFrame(feature_rows.tolist())
    
    # predict_proba returns [P(negative), P(positive)]
    probabilities = calibration_model.predict_proba(X)[:, 1]
    
    return pd.Series(probabilities, index=today_scores.index)
```

### Cold-Start Rollout Schedule

The calibration model requires resolved outcomes. Availability by horizon:

| Horizon | First resolved predictions | Min samples (500) reached | Calibration model active |
|---------|--------------------------|--------------------------|--------------------------|
| 5d      | ~April 14                | ~April 16 (2 days × 1500 tickers) | ~April 17 |
| 21d     | ~May 5                   | ~May 5 (1 day × 1500 tickers)     | ~May 6 |
| 63d     | ~mid-June                | ~mid-June                          | ~mid-June |

During cold-start, the pipeline falls back gracefully:

- **Weeks 1–2** (no resolved predictions): raw scores only, equal model weights
- **Weeks 2–4** (5d predictions resolving): 5d correction model active, other horizons raw
- **Weeks 4–8** (21d predictions resolving): 5d + 21d corrections active, 63d still raw
- **Weeks 8+** (63d predictions resolving): full calibration model active on all horizons

Each transition is automatic — gated by `min_samples` check in `fit_calibration_model`.

### Cross-Horizon Transfer

The 5d calibration model's corrections provide early signal for longer horizons.
If the 5d model learns that LGBM breakout is systematically overconfident on
momentum stocks this week, that bias likely affects 21d and 63d predictions too.

Implementation: when 63d calibration model is in cold-start but 5d model is active,
apply a dampened version of the 5d correction to the 63d raw scores:

```python
CROSS_HORIZON_DAMPENING = {
    (5, 21): 0.5,    # 5d correction applied at 50% strength to 21d
    (5, 63): 0.3,    # 5d correction applied at 30% strength to 63d
    (21, 63): 0.6,   # 21d correction applied at 60% strength to 63d
}
```

This is conservative (dampened) because 5d errors don't perfectly predict 63d errors.
But it's better than nothing during the cold-start gap.

---

## Part 4: Frontend Integration

### Screener Page

Update `ticker_index.json` to include calibrated fields:

```json
{
  "AXTI": {
    "investment_score": 0.87,
    "calibrated_prob_5d": 0.68,
    "calibrated_prob_21d": 0.72,
    "calibrated_prob_63d": 0.78,
    "calibration_status": "active",
    "model_agreement": "aligned",
    "signal_strength": "strong_buy",
    ...
  }
}
```

**Signal pill mapping:**

| Calibrated Prob (63d) | Signal | Color |
|-----------------------|--------|-------|
| ≥ 0.70               | Strong Buy | Green |
| 0.55 – 0.70          | Buy | Light Green |
| 0.45 – 0.55          | Neutral | Yellow |
| 0.30 – 0.45          | Sell | Light Red |
| < 0.30               | Strong Sell | Red |

**Model agreement indicator:**

| Condition | Label |
|-----------|-------|
| All models score > 0.65 | "Aligned" (high confidence) |
| Models spread > 0.3 | "Divergent" (interpret with caution) |
| LSTM missing | "Partial" (LGBM only) |

During cold-start, show `"calibration_status": "warming_up"` and the dashboard
renders a subtle badge: "Calibrating — 34 days remaining" based on the 63d horizon.

### Stock Detail Page — ML Score Card

The ML Score Card section on the stock detail page shows per-model breakdown:

```json
{
  "ml_score_card": {
    "combined_signal": {
      "probability_63d": 0.78,
      "probability_21d": 0.72,
      "probability_5d": 0.68,
      "label": "Strong Buy",
      "calibration_status": "active"
    },
    "model_scores": {
      "lgbm_breakout": {
        "raw_score": 0.87,
        "corrected_score": 0.84,
        "contribution": "bullish"
      },
      "lgbm_composite": {
        "raw_score": 0.79,
        "corrected_score": 0.76,
        "contribution": "bullish"
      },
      "lstm": {
        "raw_score": 0.81,
        "corrected_score": 0.78,
        "contribution": "bullish"
      }
    },
    "model_agreement": "aligned",
    "correction_note": "Calibration adjusted -3% due to recent momentum overconfidence",
    "shap_top_features": [
      { "feature": "momentum_12_1", "contribution": +0.12 },
      { "feature": "vix_regime", "contribution": -0.08 },
      { "feature": "rs_rating", "contribution": +0.05 }
    ]
  }
}
```

The `correction_note` is human-readable text generated from the largest correction
model coefficient. If the biggest correction factor is `market_return_21d` with a
negative coefficient, the note says "adjusted down due to recent market weakness."

### Forecast Page — Calibration-Adjusted Confidence Bands

The LSTM fan chart currently shows raw prediction intervals. With calibration:

```python
def adjust_forecast_bands(
    lstm_paths: np.ndarray,           # shape: (n_paths, n_days)
    calibrated_prob: float,            # from calibration model
    raw_prob: float,                   # uncalibrated probability (base rate)
    model_agreement: str,              # "aligned" or "divergent"
) -> dict:
    """
    Adjust LSTM fan chart confidence bands based on calibration.
    
    - If calibration says lower probability than raw: widen bands, shift center down
    - If models disagree: widen bands further
    - If models agree: tighten bands (higher confidence)
    """
    median_path = np.median(lstm_paths, axis=0)
    
    # Calibration shift: adjust center path
    prob_ratio = calibrated_prob / max(raw_prob, 0.01)
    # ratio < 1 means calibration is more pessimistic
    # ratio > 1 means calibration is more optimistic
    
    # Band width adjustment
    base_width = np.percentile(lstm_paths, 90, axis=0) - np.percentile(lstm_paths, 10, axis=0)
    
    if model_agreement == "divergent":
        width_multiplier = 1.4   # 40% wider bands
    elif model_agreement == "aligned":
        width_multiplier = 0.85  # 15% tighter bands
    else:
        width_multiplier = 1.0
    
    adjusted_width = base_width * width_multiplier
    
    return {
        'median': median_path.tolist(),
        'p10': (median_path - adjusted_width / 2).tolist(),
        'p90': (median_path + adjusted_width / 2).tolist(),
        'p25': (median_path - adjusted_width / 4).tolist(),
        'p75': (median_path + adjusted_width / 4).tolist(),
        'calibrated_prob': calibrated_prob,
        'model_agreement': model_agreement,
    }
```

Write adjusted paths to `data/latest/forecast_paths_*.json` alongside raw paths.

### Forecast Page — Predicted vs Actual Chart

Add a rolling accuracy chart showing resolved predictions:

```json
{
  "predicted_vs_actual": {
    "ticker": "AXTI",
    "points": [
      {
        "prediction_date": "2026-03-01",
        "horizon": 5,
        "predicted_rank": 0.85,
        "actual_rank": 0.72,
        "correct_direction": true
      },
      {
        "prediction_date": "2026-03-01",
        "horizon": 21,
        "predicted_rank": 0.85,
        "actual_rank": 0.68,
        "correct_direction": true
      }
    ],
    "rolling_hit_rate_5d": 0.64,
    "rolling_hit_rate_21d": 0.71,
    "rolling_hit_rate_63d": null
  }
}
```

The dashboard renders this as a sparkline: dots for each resolved prediction,
colored green (correct direction) or red (wrong direction), with a rolling
hit rate percentage. As more predictions resolve, the chart fills in. Users
can literally watch the model's accuracy over time.

### Model Health Page (New — Optional)

A dedicated page (or tab on the Forecast page) showing:

1. **IC time series** — three rolling charts (5d, 21d, 63d) with all models overlaid
2. **Calibration model diagnostics** — learned coefficients, which features matter most
3. **Alert history** — past warnings and resolutions
4. **Correction magnitude** — how much the calibration layer is adjusting raw scores

Data source: `data/factors/live_ic_timeseries.parquet` and `data/factors/model_alerts.json`

This page is primarily for you (the developer/operator) rather than end users.
Consider putting it behind a power-user toggle or a separate URL path.

---

## Updated Daily Pipeline (`alpha ml score`)

Full step-by-step after all changes:

```python
def daily_score_pipeline(data_dir, models_dir):
    """
    Complete daily scoring pipeline.
    Called by GitHub Actions at 4:15pm ET.
    """
    # === Step 1: Load models and score all tickers ===
    config = load_config(models_dir / 'config.json')
    models = load_champion_models(config, models_dir)
    universe = load_universe(data_dir)
    
    raw_scores = {}  # {model_id: {ticker: score}}
    for model_id, model in models.items():
        raw_scores[model_id] = score_universe(model, universe)
    
    # === Step 2: Update predictions ledger ===
    ledger = load_ledger(data_dir / 'factors/predictions_ledger.parquet')
    
    # Append today's predictions (with context features for correction)
    today_rows = build_ledger_rows(
        raw_scores, universe, 
        include_context=True  # market_return_21d, vix, sector, momentum_bucket
    )
    ledger = pd.concat([ledger, today_rows])
    
    # Fill realized returns for past predictions
    ledger = fill_realized_returns(ledger, universe, horizons=[5, 21, 63])
    
    # === Step 3: Daily IC measurement ===
    today_ic = compute_daily_ic(ledger, pd.Timestamp.today())
    ic_timeseries = load_ic_timeseries(data_dir / 'factors/live_ic_timeseries.parquet')
    ic_timeseries = pd.concat([ic_timeseries, today_ic])
    save_ic_timeseries(ic_timeseries, data_dir / 'factors/live_ic_timeseries.parquet')
    
    # Check alerts
    alerts = []
    for model_id in config['models']:
        alerts.extend(check_alerts(ic_timeseries, model_id['id']))
    save_alerts(alerts, ic_timeseries, data_dir / 'factors/model_alerts.json')
    
    # === Step 4: Residual correction ===
    correction_models = {}
    for model_id in raw_scores.keys():
        for horizon in [5, 21, 63]:
            cm = fit_residual_correction(ledger, model_id, horizon)
            if cm is not None:
                correction_models[(model_id, horizon)] = cm
    
    corrected_scores = apply_all_corrections(raw_scores, correction_models, universe)
    
    # === Step 5: Calibration model ===
    calibrated = {}
    for horizon in [5, 21, 63]:
        cal_model = fit_calibration_model(ledger, horizon)
        
        if cal_model is None:
            # Cold-start: try cross-horizon transfer
            shorter = next((h for h in [5, 21] if h < horizon 
                           and (dampening := CROSS_HORIZON_DAMPENING.get((h, horizon)))), None)
            # Apply dampened shorter-horizon correction if available
            calibrated[horizon] = apply_cold_start_fallback(
                corrected_scores, horizon, correction_models
            )
        else:
            calibrated[horizon] = apply_calibration(
                corrected_scores, cal_model, horizon
            )
        
        # Save calibrator for inspection
        save_calibrator(cal_model, data_dir / f'factors/calibrator_{horizon}d.pkl')
    
    # === Step 6: Write outputs ===
    # Update ledger with corrected scores and calibrated probs
    update_ledger_calibrated(ledger, corrected_scores, calibrated)
    save_ledger(ledger, data_dir / 'factors/predictions_ledger.parquet')
    
    # Build ticker_index.json with all signals
    ticker_index = build_ticker_index(
        raw_scores, corrected_scores, calibrated, 
        model_agreement=compute_agreement(raw_scores),
        calibration_status=get_calibration_status(calibrated),
    )
    save_json(ticker_index, data_dir / 'latest/ticker_index.json')
    
    # LSTM forecast paths with calibration-adjusted bands
    if 'lstm' in models:
        forecast_paths = generate_lstm_forecasts(models['lstm'], universe)
        adjusted_paths = adjust_all_forecast_bands(
            forecast_paths, calibrated, 
            compute_agreement(raw_scores)
        )
        save_json(adjusted_paths, data_dir / 'latest/forecast_paths_lstm.json')
    
    # Model health summary for dashboard
    save_model_health(ic_timeseries, alerts, data_dir / 'factors/model_alerts.json')
```

---

## Updated File Reference

| File | Purpose | Updated | New? |
|------|---------|---------|------|
| `data/factors/predictions_ledger.parquet` | All predictions + actuals + corrections | Daily | Modified schema |
| `data/factors/live_ic_timeseries.parquet` | Per-model per-horizon IC history | Daily | **NEW** |
| `data/factors/model_alerts.json` | Alert history + model health summary | Daily | **NEW** |
| `data/factors/calibrator_5d.pkl` | 5d calibration model | Daily | **NEW** |
| `data/factors/calibrator_21d.pkl` | 21d calibration model | Daily | **NEW** |
| `data/factors/calibrator_63d.pkl` | 63d calibration model | Daily | **NEW** |
| `data/latest/ticker_index.json` | All ticker scores + calibrated probs | Daily | Modified schema |
| `data/latest/forecast_paths_lstm.json` | LSTM fan charts with adjusted bands | Daily | Modified schema |
| `models/config.json` | Champion model registry | On promotion | No change |

---

## Implementation Order

This system has internal dependencies. Implement in this order:

1. **Ledger schema update** — add `model_version`, `is_backfill`, context columns,
   `corrected_score`, `calibrated_prob`, actual columns per horizon.
   Must be backward-compatible (new columns default to NaN for existing rows).

2. **Daily IC measurement** (Part 1) — can run immediately once ledger has any
   resolved predictions. No dependencies on other new systems.

3. **Residual correction** (Part 2) — needs ~200 resolved predictions per model
   per horizon. Active for 5d within ~1 day, 21d within ~1 day, 63d within ~1 day
   (since universe is ~1500 tickers, one day of resolved predictions = 1500 pairs).

4. **Calibration model** (Part 3) — needs ~500 resolved predictions in wide format
   (all models scored same ticker on same day). Active for 5d within ~1 week,
   21d within ~1 month, 63d within ~3 months.

5. **Frontend integration** (Part 4) — can start immediately for raw score display.
   Calibrated fields populate as each calibrator comes online.

6. **Cross-horizon transfer** — implement after 5d calibrator is active but before
   63d comes online. Low priority if 21d calibrator bridges the gap adequately.

---

## What NOT to Change

- **Base model training** — LGBM and LSTM training pipelines are untouched.
  The calibration layer sits on top of frozen model outputs.
- **`alpha spread`** — daily data fetching is unchanged.
- **GitHub Actions triggers** — keep 4:15pm ET weekday schedule.
- **Model promotion flow** — `config.json` + `alpha ml promote` stays the same.
  When promoting a new model version, reset that model's correction model
  (old residuals don't apply to new weights).
