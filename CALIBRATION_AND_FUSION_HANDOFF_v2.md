# Adaptive Calibration & Model Fusion — Claude Code Handoff (Revised)

**Date:** 2026-04-07  
**Status:** Implement after LSTM v3 training + LGBM v10 production refit + LSTM inference workflow  
**Revision note:** Replaces previous handoff. Core change: single `calibration_daily.parquet`
file serves both LSTM inference and calibration context. No cold-start wait — backfill
seeds historical data, daily spread maintains it.

---

## Core Data Structure: `calibration_daily.parquet`

This is the central new file. A rolling 63-trading-day window of per-ticker features,
maintained by the daily spread pipeline. Fixed size (~5–8MB), never grows beyond ~95K rows.

```
data/factors/calibration_daily.parquet

Rows:    ~95,000 (63 trading days × ~1,500 tickers)
Size:    ~5–8MB compressed (zstd)
Updated: Daily by alpha ml score (append today, drop oldest day)

Columns: date, ticker, + all 77 features from stocks_3m.csv
         (rsi, momentum_12_1, volatility_20d, sma_200_ratio, bollinger_pct_b,
          earnings_yield, market_cap_cat, sector_code, vix_close,
          macd, macd_signal, volume_surge, ... everything the models use)
```

**Two consumers, one file:**

1. **LSTM inference** — filter to one ticker, sort by date, take 63 rows → input sequence.
   No `dataset.parquet` needed. No separate sequence cache. LSTM reads directly from
   this file in GitHub Actions.

2. **Calibration model** — join with predictions ledger on `(ticker, date)` to get
   per-stock feature context alongside resolved outcomes. The correction and calibration
   models fit on these features.

### Seeding (one-time, on HPC)

```python
def seed_calibration_daily(dataset_path: str, output_path: str):
    """
    Extract the most recent 63 trading days from dataset.parquet.
    Run once locally after dataset extension. Commit output to repo.
    """
    dataset = pd.read_parquet(dataset_path)
    
    # Get the last 63 unique trading dates
    dates = sorted(dataset['date'].unique())
    cutoff_date = dates[-63] if len(dates) >= 63 else dates[0]
    recent = dataset[dataset['date'] >= cutoff_date].copy()
    
    # Keep only columns that match stocks_3m.csv output
    # (assert column compatibility before saving)
    spread_cols = _get_spread_column_names()  # from alpha spread output
    missing = set(spread_cols) - set(recent.columns)
    if missing:
        raise ValueError(f"Column mismatch — missing in dataset: {missing}")
    
    recent = recent[['date', 'ticker'] + spread_cols]
    recent.to_parquet(output_path, compression='zstd', index=False)
    
    print(f"Seeded {len(recent):,} rows ({recent['date'].nunique()} days, "
          f"{recent['ticker'].nunique()} tickers) → {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1e6:.1f} MB")
```

Run after the LGBM v10 / LSTM v3 training weekend:
```bash
# On HPC / local machine with dataset.parquet
python -c "
from morningalpha.ml.calibration import seed_calibration_daily
seed_calibration_daily('data/training/dataset.parquet', 'data/factors/calibration_daily.parquet')
"
git add data/factors/calibration_daily.parquet
git commit -m "chore(ml): seed calibration daily context"
git push
```

### Daily Update (automated, GitHub Actions)

```python
def update_calibration_daily(
    calibration_path: str,
    stocks_3m_path: str,
    today: pd.Timestamp,
    max_days: int = 63,
):
    """
    Append today's spread features. Drop anything older than 63 trading days.
    Called by alpha ml score after alpha spread completes.
    """
    # Read today's spread output
    today_features = pd.read_csv(stocks_3m_path)
    today_features['date'] = today
    
    # Load existing rolling window
    existing = pd.read_parquet(calibration_path)
    
    # Check column compatibility
    shared_cols = sorted(set(existing.columns) & set(today_features.columns))
    if len(shared_cols) < 20:  # sanity check — should have 70+ shared columns
        raise ValueError(
            f"Column mismatch: only {len(shared_cols)} shared columns. "
            f"Expected 70+. Check spread pipeline output."
        )
    
    # Align columns (use existing schema, fill missing with NaN)
    for col in existing.columns:
        if col not in today_features.columns:
            today_features[col] = np.nan
    today_features = today_features[existing.columns]
    
    # Append and trim
    combined = pd.concat([existing, today_features], ignore_index=True)
    
    # Keep only last max_days trading days
    unique_dates = sorted(combined['date'].unique())
    if len(unique_dates) > max_days:
        cutoff_date = unique_dates[-max_days]
        combined = combined[combined['date'] >= cutoff_date]
    
    combined.to_parquet(calibration_path, compression='zstd', index=False)
    
    return len(combined)
```

---

## LSTM Inference From `calibration_daily.parquet`

The LSTM requires a 63-row sequence of features per ticker. Previously this required
`dataset.parquet` (3.4GB, HPC only). Now it reads from `calibration_daily.parquet`.

```python
def build_lstm_sequences(calibration_path: str, tickers: list[str]) -> dict:
    """
    Build LSTM input sequences from the rolling 63-day feature window.
    Returns {ticker: np.ndarray of shape (63, n_features)}.
    """
    df = pd.read_parquet(calibration_path)
    
    sequences = {}
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker].sort_values('date')
        
        if len(ticker_data) < 63:
            continue  # not enough history yet — skip this ticker
        
        # Take most recent 63 rows
        ticker_data = ticker_data.tail(63)
        
        # Extract the feature columns the LSTM was trained on
        feature_cols = _get_lstm_feature_cols()  # same as training
        sequence = ticker_data[feature_cols].values  # shape: (63, n_features)
        
        sequences[ticker] = sequence
    
    return sequences
```

**First-day behavior:** After seeding, all tickers have 63 rows immediately (from the
backfill). The LSTM can run inference on day one. As the daily pipeline appends new
days and drops old ones, the sequences stay fresh.

**Tickers with gaps:** If a ticker is missing from some trading days (delisted, halted,
or newly added to universe), it will have fewer than 63 rows. The LSTM skips these.
After a new ticker has been in the universe for 63 days, it automatically becomes
scoreable.

---

## Predictions Ledger (unchanged schema)

The predictions ledger remains the append-only log of predictions and outcomes.
It does NOT store per-stock features — those live in `calibration_daily.parquet`.

```
data/factors/predictions_ledger.parquet

Columns:
  ticker              str
  scored_date         datetime64
  price_at_score      float64
  
  # Raw model scores (one column per active model)
  raw_lgbm_breakout_v10      float64
  raw_lgbm_composite_v10     float64
  raw_lgbm_breakout_medium_v1  float64
  raw_lgbm_composite_medium_v1 float64
  raw_lstm_clip_v1           float64
  
  # Evaluation dates
  eval_after_5d       datetime64
  eval_after_21d      datetime64
  eval_after_63d      datetime64
  
  # Realized returns (filled in later)
  realized_return_5d  float64
  realized_return_21d float64
  realized_return_63d float64
  
  # Maturity flags
  matured_5d          bool
  matured_21d         bool
  matured_63d         bool
  
  # Metadata
  is_backfill         bool
  model_version       str       ← tracks which checkpoint generated each score
```

**Join pattern for calibration fitting:**

```python
def get_calibration_training_data(ledger_path, calibration_daily_path, horizon):
    """
    Join ledger with feature context for calibration model training.
    """
    ledger = pd.read_parquet(ledger_path)
    context = pd.read_parquet(calibration_daily_path)
    
    # Only resolved predictions
    resolved = ledger[ledger[f'matured_{horizon}d'] == True].copy()
    
    # Join feature context from the date the prediction was made
    merged = resolved.merge(
        context,
        left_on=['ticker', 'scored_date'],
        right_on=['ticker', 'date'],
        how='left'
    )
    
    return merged
```

**Note:** For live predictions (last 63 days), the context join will succeed because
`calibration_daily.parquet` holds exactly that window. For older backfill predictions,
the context columns will be NaN (those dates have rolled off). This is fine — the
calibration model fits primarily on recent resolved data where context is available.
Backfill rows still contribute their raw scores and realized returns; they just
lack the rich feature context.

---

## Daily IC Monitoring

### What It Does

Every day, compute how well each model's past predictions matched actual outcomes.
Look at predictions made 5, 21, and 63 days ago and compare to today's realized returns.

### Implementation

```python
from scipy.stats import spearmanr

def compute_daily_ic(
    ledger: pd.DataFrame,
    today: pd.Timestamp,
    horizons: list[int] = [5, 21, 63],
) -> pd.DataFrame:
    """
    For each model and horizon, compute cross-sectional rank IC between
    predictions made N days ago and today's realized returns.
    """
    results = []
    model_cols = [c for c in ledger.columns if c.startswith('raw_')]
    
    for horizon in horizons:
        # Get predictions that were made `horizon` days ago and have now matured
        cohort = ledger[
            (ledger[f'matured_{horizon}d'] == True) &
            (ledger['eval_after_{horizon}d'] <= today) &
            (ledger['eval_after_{horizon}d'] > today - pd.offsets.BDay(5))  # recent cohort only
        ]
        
        if len(cohort) < 50:
            continue
        
        for model_col in model_cols:
            model_id = model_col.replace('raw_', '')
            scores = cohort[model_col].dropna()
            actuals = cohort.loc[scores.index, f'realized_return_{horizon}d']
            
            valid = scores.notna() & actuals.notna()
            if valid.sum() < 50:
                continue
            
            ic, _ = spearmanr(scores[valid], actuals[valid])
            hit_rate = (
                (scores[valid] > scores[valid].median()) ==
                (actuals[valid] > 0)
            ).mean()
            
            results.append({
                'date': today,
                'model_id': model_id,
                'horizon': horizon,
                'ic': ic,
                'hit_rate': hit_rate,
                'n_tickers': valid.sum(),
                'market_return': _get_spy_return(today, horizon),
            })
    
    return pd.DataFrame(results)
```

### Storage

```
data/factors/live_ic_timeseries.parquet  (~12KB, grows slowly)

Columns: date, model_id, horizon, ic, hit_rate, n_tickers, market_return
```

Append daily. Never truncate — this is the historical record of model performance.

### Alert Thresholds

```python
def check_model_health(ic_timeseries: pd.DataFrame) -> dict:
    """
    Check rolling IC for each model. Returns health summary + alerts.
    """
    health = {}
    alerts = []
    
    for model_id in ic_timeseries['model_id'].unique():
        model_ic = ic_timeseries[ic_timeseries['model_id'] == model_id]
        model_health = {'model_id': model_id}
        
        for horizon in [5, 21, 63]:
            recent = model_ic[model_ic['horizon'] == horizon].tail(10)
            if len(recent) < 3:
                model_health[f'{horizon}d_ic'] = None
                continue
            
            rolling_ic = recent['ic'].mean()
            model_health[f'{horizon}d_ic'] = round(rolling_ic, 4)
            
            if rolling_ic < 0.0:
                alerts.append({
                    'level': 'CRITICAL',
                    'model_id': model_id,
                    'horizon': horizon,
                    'message': f'{horizon}d rolling IC = {rolling_ic:.4f} (negative)',
                })
                model_health['status'] = 'retrain'
            elif horizon == 63 and rolling_ic < 0.02:
                alerts.append({
                    'level': 'WARNING',
                    'model_id': model_id,
                    'horizon': horizon,
                    'message': f'63d rolling IC = {rolling_ic:.4f} (below threshold)',
                })
                model_health.setdefault('status', 'degrading')
            else:
                model_health.setdefault('status', 'healthy')
        
        health[model_id] = model_health
    
    return {
        'last_checked': str(pd.Timestamp.today().date()),
        'model_health': health,
        'alerts': alerts,
    }
```

Write to `data/factors/model_health.json`.

---

## Residual Correction

### What It Does

Learns the systematic errors each model makes in the current regime.
The 5d residuals resolve within a week, giving fast feedback. The correction
transfers across horizons — if the model is overconfident on momentum stocks
at 5d, it's probably overconfident at 63d too.

### Implementation

```python
from sklearn.linear_model import Ridge
import joblib

def fit_all_corrections(
    ledger: pd.DataFrame,
    context: pd.DataFrame,
    horizons: list[int] = [5, 21, 63],
    lookback_pairs: int = 3000,
    min_samples: int = 200,
) -> dict:
    """
    Fit Ridge correction model per (model, horizon).
    
    The correction predicts: actual_return - raw_score
    Features: [raw_score, sector, market_return_21d, vix, momentum_bucket]
    
    Returns dict keyed by "{model_id}_{horizon}d".
    """
    model_cols = [c for c in ledger.columns if c.startswith('raw_')]
    
    # Join ledger with context features
    merged = ledger.merge(
        context, left_on=['ticker', 'scored_date'],
        right_on=['ticker', 'date'], how='left'
    )
    
    correction_features = [
        'sector_code', 'market_return_21d', 'vix_at_prediction', 'momentum_bucket'
    ]
    
    corrections = {}
    
    for model_col in model_cols:
        model_id = model_col.replace('raw_', '')
        
        for horizon in horizons:
            resolved = merged[
                (merged[f'matured_{horizon}d'] == True) &
                (merged[model_col].notna())
            ].sort_values('scored_date', ascending=False).head(lookback_pairs)
            
            if len(resolved) < min_samples:
                continue
            
            # Residual = what actually happened - what we predicted
            residual = resolved[f'realized_return_{horizon}d'] - resolved[model_col]
            
            # Features for correction (raw_score + context)
            X = resolved[[model_col] + correction_features].fillna(0)
            X.columns = ['raw_score'] + correction_features
            
            model = Ridge(alpha=10.0)  # heavy regularization
            model.fit(X, residual)
            
            corrections[f'{model_id}_{horizon}d'] = model
    
    return corrections


def apply_corrections(
    raw_scores: pd.DataFrame,
    context_today: pd.DataFrame,
    corrections: dict,
    horizon: int = 63,
) -> pd.DataFrame:
    """
    Apply residual corrections to today's raw scores.
    Returns DataFrame with corrected_* columns added.
    """
    corrected = raw_scores.copy()
    model_cols = [c for c in raw_scores.columns if c.startswith('raw_')]
    
    correction_features = [
        'sector_code', 'market_return_21d', 'vix_at_prediction', 'momentum_bucket'
    ]
    
    for model_col in model_cols:
        model_id = model_col.replace('raw_', '')
        key = f'{model_id}_{horizon}d'
        
        if key not in corrections:
            corrected[f'corrected_{model_id}'] = corrected[model_col]
            continue
        
        X = corrected[[model_col]].copy()
        X.columns = ['raw_score']
        for feat in correction_features:
            X[feat] = context_today.get(feat, 0)
        X = X.fillna(0)
        
        correction = corrections[key].predict(X)
        correction = np.clip(correction, -10, 10)  # clip ±10 percentile points
        
        corrected[f'corrected_{model_id}'] = corrected[model_col] + correction
    
    return corrected
```

---

## Learned Calibration Model

### What It Does

Replaces the hedge weights + isotonic pipeline with a single model that:
- Takes all corrected model scores as input
- Learns nonlinear interactions (model agreement boosts confidence)
- Conditions on market regime (VIX, recent market return)
- Outputs P(positive return) per horizon

### Implementation

```python
from sklearn.linear_model import LogisticRegression

def build_calibration_features(row: pd.Series, model_ids: list[str]) -> dict:
    """
    Build feature vector for calibration model from a ledger+context row.
    """
    scores = {}
    for model_id in model_ids:
        col = f'corrected_{model_id}'
        fallback_col = f'raw_{model_id}'
        scores[model_id] = row.get(col, row.get(fallback_col, 50.0))
    
    # Pairwise agreement features
    score_values = list(scores.values())
    features = {}
    
    # Individual corrected scores
    for model_id, score in scores.items():
        features[f'score_{model_id}'] = score
    
    # Agreement: product of all scores (high when all agree bullish)
    features['agreement_all'] = np.prod(score_values) / (100 ** (len(score_values) - 1))
    
    # Spread: max disagreement between any two models
    features['max_spread'] = max(score_values) - min(score_values)
    
    # LGBM vs LSTM agreement (if LSTM present)
    lgbm_scores = [s for m, s in scores.items() if 'lgbm' in m]
    lstm_scores = [s for m, s in scores.items() if 'lstm' in m]
    if lgbm_scores and lstm_scores:
        features['lgbm_lstm_spread'] = np.mean(lgbm_scores) - np.mean(lstm_scores)
        features['lgbm_lstm_agreement'] = np.mean(lgbm_scores) * np.mean(lstm_scores) / 100
    
    # Context features (from calibration_daily.parquet join)
    features['rsi'] = row.get('rsi', 50.0)
    features['momentum_12_1'] = row.get('momentum_12_1', 0.0)
    features['volatility_20d'] = row.get('volatility_20d', 0.02)
    features['vix'] = row.get('vix_close', row.get('vix_at_prediction', 20.0))
    features['market_return_21d'] = row.get('market_return_21d', 0.0)
    features['sector_code'] = row.get('sector_code', 0)
    features['bollinger_pct_b'] = row.get('bollinger_pct_b', 0.5)
    features['price_to_sma200'] = row.get('sma_200_ratio', 1.0)
    
    return features


def fit_calibration_model(
    ledger: pd.DataFrame,
    context: pd.DataFrame,
    horizon: int,
    model_ids: list[str],
    min_samples: int = 500,
    rolling_window: int = 5000,
) -> LogisticRegression | None:
    """
    Fit calibration model on resolved predictions joined with feature context.
    
    Uses weighted samples: backfill rows get 0.3 weight, live rows get 1.0.
    This lets the model activate immediately from backfill while gradually
    shifting toward live signal.
    """
    # Join ledger with context
    merged = ledger.merge(
        context, left_on=['ticker', 'scored_date'],
        right_on=['ticker', 'date'], how='left'
    )
    
    # Only resolved predictions
    resolved = merged[merged[f'matured_{horizon}d'] == True]
    resolved = resolved.sort_values('scored_date', ascending=False).head(rolling_window)
    
    if len(resolved) < min_samples:
        return None
    
    # Build features
    feature_rows = resolved.apply(
        lambda row: build_calibration_features(row, model_ids), axis=1
    )
    X = pd.DataFrame(feature_rows.tolist())
    
    # Binary target: positive return?
    y = (resolved[f'realized_return_{horizon}d'] > 0).astype(int)
    
    # Sample weights: downweight backfill, full weight for live
    weights = np.where(resolved['is_backfill'], 0.3, 1.0)
    
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
    )
    model.fit(X, y, sample_weight=weights)
    
    return model


def apply_calibration(
    today_scores: pd.DataFrame,
    context_today: pd.DataFrame,
    calibration_model: LogisticRegression | None,
    model_ids: list[str],
) -> pd.Series:
    """
    Apply calibration to today's corrected scores.
    Returns calibrated probabilities (0–1).
    Falls back to rank percentile / 100 during cold start.
    """
    if calibration_model is None:
        return today_scores['MLScore'] / 100.0  # uncalibrated fallback
    
    # Merge today's scores with today's context
    merged = today_scores.merge(context_today, on='ticker', how='left')
    
    feature_rows = merged.apply(
        lambda row: build_calibration_features(row, model_ids), axis=1
    )
    X = pd.DataFrame(feature_rows.tolist())
    
    # P(positive return)
    probs = calibration_model.predict_proba(X)[:, 1]
    
    return pd.Series(probs, index=today_scores.index)
```

### Signal Mapping

```python
SIGNAL_THRESHOLDS = {
    'STRONG BUY':  0.70,
    'BUY':         0.55,
    'HOLD':        0.45,
    'SELL':        0.30,
    'STRONG SELL': 0.00,
}

def prob_to_signal(prob: float) -> str:
    for signal, threshold in SIGNAL_THRESHOLDS.items():
        if prob >= threshold:
            return signal
    return 'STRONG SELL'
```

---

## Updated Daily Pipeline (`alpha ml score`)

Full step-by-step with all changes integrated:

```python
def daily_score_pipeline(data_dir: Path, models_dir: Path):
    """
    Complete daily scoring pipeline.
    Called by GitHub Actions at 4:15pm ET after alpha spread.
    """
    today = pd.Timestamp.today().normalize()
    config = load_config(models_dir / 'config.json')
    model_ids = [m['id'] for m in config['models']]
    
    # === Paths ===
    stocks_path = data_dir / 'latest/stocks_3m.csv'
    calibration_daily_path = data_dir / 'factors/calibration_daily.parquet'
    ledger_path = data_dir / 'factors/predictions_ledger.parquet'
    ic_path = data_dir / 'factors/live_ic_timeseries.parquet'
    health_path = data_dir / 'factors/model_health.json'
    corrections_path = data_dir / 'factors/correction_models.joblib'
    calibrators_path = data_dir / 'factors/calibration_models.joblib'
    weights_path = data_dir / 'factors/model_weights.json'
    
    # === Step 1: Update calibration_daily.parquet ===
    # Append today's spread features, drop oldest day
    n_rows = update_calibration_daily(calibration_daily_path, stocks_path, today)
    print(f"  calibration_daily: {n_rows:,} rows ({n_rows // 1500:.0f} trading days)")
    
    # === Step 2: Score all tickers with all champion models ===
    stocks = pd.read_csv(stocks_path)
    models = load_champion_models(config, models_dir)
    
    raw_scores = score_all_models(models, stocks)  # {model_id: Series of scores}
    
    # For LSTM: build sequences from calibration_daily.parquet
    if any('lstm' in m for m in models):
        sequences = build_lstm_sequences(calibration_daily_path, stocks['ticker'].tolist())
        lstm_models = {k: v for k, v in models.items() if 'lstm' in k}
        lstm_scores = score_lstm_models(lstm_models, sequences)
        raw_scores.update(lstm_scores)
    
    # Rank-percentile raw scores 0–100
    for model_id in raw_scores:
        raw_scores[model_id] = raw_scores[model_id].rank(pct=True) * 100
    
    # === Step 3: Update predictions ledger ===
    ledger = pd.read_parquet(ledger_path)
    
    # Fill realized returns for past predictions whose eval dates have passed
    ledger = fill_realized_returns(ledger, stocks, today)
    
    # Append today's predictions
    today_rows = build_ledger_rows(raw_scores, stocks, today)
    ledger = pd.concat([ledger, today_rows], ignore_index=True)
    
    # === Step 4: Hedge weight update ===
    weights = update_hedge_weights(ledger, model_ids, weights_path)
    
    # === Step 5: Ensemble score ===
    ensemble_scores = compute_ensemble(raw_scores, weights)  # → MLScore
    
    # === Step 6: Sector diversity cap ===
    ensemble_scores = apply_sector_cap(ensemble_scores, stocks)
    
    # === Step 7: Load and apply corrections ===
    context_today = pd.read_parquet(calibration_daily_path)
    context_today = context_today[context_today['date'] == today]
    
    corrections = joblib.load(corrections_path) if corrections_path.exists() else {}
    corrected = apply_corrections(raw_scores, context_today, corrections)
    
    # === Step 8: Load and apply calibration ===
    calibrators = joblib.load(calibrators_path) if calibrators_path.exists() else {}
    
    calibrated_probs = {}
    for horizon in [5, 21, 63]:
        cal_model = calibrators.get(f'{horizon}d')
        calibrated_probs[horizon] = apply_calibration(
            corrected, context_today, cal_model, model_ids
        )
    
    # Map to signals
    signals = calibrated_probs[63].apply(prob_to_signal)
    
    # === Step 9: Write ticker_index.json / update CSVs ===
    write_scores_to_csvs(
        data_dir, stocks, ensemble_scores, corrected,
        calibrated_probs, signals, raw_scores
    )
    
    # === Step 10: LSTM forecast paths ===
    if any('lstm' in m for m in models):
        forecast_paths = generate_lstm_forecasts(
            lstm_models, sequences, calibrated_probs, raw_scores
        )
        save_json(forecast_paths, data_dir / 'latest/forecast_paths_lstm.json')
    
    # === Step 11: Save updated ledger ===
    ledger.to_parquet(ledger_path, index=False)
    
    print(f"  Scored {len(stocks)} tickers with {len(model_ids)} models")
    print(f"  Calibrated signal: {signals.value_counts().to_dict()}")
```

---

## Daily Calibration (`--calibrate` flag)

Runs as part of the same GitHub Actions job, after scoring:

```python
def daily_calibration_pipeline(data_dir: Path, model_ids: list[str]):
    """
    Re-fit correction and calibration models on latest resolved predictions.
    Update IC timeseries and model health.
    """
    today = pd.Timestamp.today().normalize()
    
    ledger = pd.read_parquet(data_dir / 'factors/predictions_ledger.parquet')
    context = pd.read_parquet(data_dir / 'factors/calibration_daily.parquet')
    
    # === IC Timeseries ===
    today_ic = compute_daily_ic(ledger, today)
    ic_ts = pd.read_parquet(data_dir / 'factors/live_ic_timeseries.parquet')
    ic_ts = pd.concat([ic_ts, today_ic], ignore_index=True)
    ic_ts.to_parquet(data_dir / 'factors/live_ic_timeseries.parquet', index=False)
    
    # === Model Health Alerts ===
    health = check_model_health(ic_ts)
    save_json(health, data_dir / 'factors/model_health.json')
    
    if health['alerts']:
        for alert in health['alerts']:
            print(f"  [{alert['level']}] {alert['message']}")
    
    # === Re-fit Correction Models ===
    corrections = fit_all_corrections(ledger, context)
    joblib.dump(corrections, data_dir / 'factors/correction_models.joblib')
    print(f"  Fitted {len(corrections)} correction models")
    
    # === Re-fit Calibration Models ===
    calibrators = {}
    for horizon in [5, 21, 63]:
        cal = fit_calibration_model(ledger, context, horizon, model_ids)
        if cal is not None:
            calibrators[f'{horizon}d'] = cal
            print(f"  Calibrator {horizon}d: active (fitted on "
                  f"{len(ledger[ledger[f'matured_{horizon}d']])} resolved rows)")
        else:
            print(f"  Calibrator {horizon}d: cold start (insufficient data)")
    
    joblib.dump(calibrators, data_dir / 'factors/calibration_models.joblib')
```

---

## Frontend Integration

### ticker_index.json Schema Update

```json
{
  "AXTI": {
    "investment_score": 87,
    "calibrated_prob_5d": 0.68,
    "calibrated_prob_21d": 0.72,
    "calibrated_prob_63d": 0.78,
    "signal": "STRONG BUY",
    "calibration_status": "active",
    "model_agreement": "aligned",
    "model_scores": {
      "lgbm_breakout_v10": { "raw": 91, "corrected": 88 },
      "lgbm_composite_v10": { "raw": 82, "corrected": 79 },
      "lstm_clip_v3": { "raw": 85, "corrected": 83 }
    },
    "correction_note": "Adjusted -3pt: momentum overconfidence in current regime"
  }
}
```

### Screener Page

- **Signal pill** next to each ticker: colored by calibrated_prob_63d
  - Green (≥70%), Light green (≥55%), Yellow (≥45%), Light red (≥30%), Red (<30%)
- **Model agreement badge**: "Aligned" / "Divergent" / "Partial (LGBM only)"
- During cold-start: show "Calibrating" badge instead of signal pill

### Stock Detail Page — ML Score Card

Display per-model breakdown:
- Combined calibrated probability per horizon (5d, 21d, 63d)
- Individual model scores (raw → corrected) with contribution direction
- Model agreement indicator
- Correction note explaining the largest adjustment
- SHAP top features (from base model, not calibration)

### Forecast Page

- LSTM fan chart confidence bands adjusted by calibration:
  - Models agree → tighter bands (higher confidence)
  - Models disagree → wider bands
  - Calibrated probability shifts the center path
- Predicted vs Actual sparkline: resolved predictions colored green/red
- Rolling hit rate per horizon

### Model Health Page (operator view)

- IC time series charts (5d, 21d, 63d) with all models overlaid
- Alert history
- Calibration model diagnostics (learned coefficients, feature importance)
- Correction magnitude over time

---

## File Reference

| File | Size | Purpose | Updated |
|------|------|---------|---------|
| `data/factors/calibration_daily.parquet` | ~5–8MB | Rolling 63-day feature window (LSTM sequences + calibration context) | Daily |
| `data/factors/predictions_ledger.parquet` | ~20MB+ | Append-only prediction + outcome log | Daily |
| `data/factors/live_ic_timeseries.parquet` | ~12KB+ | Per-model per-horizon IC history | Daily |
| `data/factors/model_health.json` | ~2KB | Alerts + rolling IC summaries | Daily |
| `data/factors/correction_models.joblib` | ~3KB | Ridge correction models per (model, horizon) | Daily |
| `data/factors/calibration_models.joblib` | ~2KB | LogisticRegression calibrators per horizon | Daily |
| `data/factors/model_weights.json` | ~1KB | Hedge ensemble weights | Daily |
| `data/latest/ticker_index.json` | varies | All ticker scores + calibrated signals | Daily |
| `data/latest/forecast_paths_lstm.json` | varies | LSTM fan charts with adjusted bands | Daily |
| `models/config.json` | ~1KB | Champion model registry | On promotion |

---

## Implementation Order

1. **`calibration_daily.parquet` seeding + daily update** — this unblocks everything.
   Seed from `dataset.parquet` on HPC, add the append/trim logic to `alpha ml score`.
   Also unblocks LSTM inference in GitHub Actions.

2. **LSTM inference from `calibration_daily.parquet`** — replace whatever sequence
   loading currently exists with reads from this file. Test that LSTM scores match
   what you get from `dataset.parquet` (within floating point tolerance).

3. **Daily IC monitoring** — can run as soon as the ledger has any matured rows.
   No dependencies on correction or calibration.

4. **Residual correction** — needs resolved predictions + context. Active immediately
   from backfill data.

5. **Calibration model** — needs resolved predictions + context + all model scores.
   Active immediately from backfill (with 0.3 weight on backfill rows).

6. **Frontend integration** — progressive. Raw scores display immediately.
   Calibrated signals appear as soon as calibrators are fitted.

---

## What NOT to Change

- **Base model training** — LGBM and LSTM training pipelines untouched.
- **`alpha spread`** — daily data fetching unchanged. Its output feeds everything.
- **GitHub Actions schedule** — keep 4:15pm ET weekday trigger.
- **Model promotion flow** — `config.json` + `alpha ml promote` unchanged.
- **Ledger backfill** — `alpha ml backfill` unchanged. Just re-run after seeding
  `calibration_daily.parquet` if the ledger was reset.

When promoting a new model version: the correction model for the old version's
residuals no longer applies. The daily calibration pipeline will automatically
re-fit corrections for the new model as its predictions start resolving.
