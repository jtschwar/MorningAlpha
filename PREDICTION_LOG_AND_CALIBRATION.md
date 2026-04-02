# MorningAlpha — Prediction Ledger & Online Calibration

> **Status:** Planning
> **Phase:** Post-`composite_v4`, pre-Set Transformer
> **Prerequisite:** `alpha ml predict` running daily in GitHub Actions
> **Objective:** Build infrastructure to accumulate daily predictions, measure live model
> accuracy, and auto-calibrate signal thresholds — so every model we ship gets
> better over time without retraining.
>
> **Key principle:** The base model (LightGBM breakout/momentum) stays frozen. Only
> the calibration layer that maps raw scores → user-facing signals adapts daily.
> This is not a new model. It is measurement + recalibration infrastructure.

---

## Why This Matters Now

Today, `alpha ml predict` writes `predictions.csv` and each day's file **overwrites** the
previous day. Once it's gone, we lose the ability to compare what we predicted vs. what
actually happened. Without that feedback loop:

- We can't compute live rank IC (only historical backtest IC)
- Signal thresholds (STRONG BUY at ≥80, BUY at 65–79, etc.) are static guesses from
  backtest-era distributions — they drift as market regimes change
- We can't tell if the model is degrading until someone notices bad picks manually
- Every new model (MLP, Set Transformer, future ensembles) needs the same infrastructure

Building the prediction ledger now means every model we ship from this point forward
automatically gets live accuracy tracking and adaptive calibration.

---

## Part 1 — Prediction Ledger

### Concept

An append-only parquet file that accumulates every daily prediction. After the forward
return window elapses (63 trading days for the current label), a backfill job fills in
the actual realized return. This creates the (predicted, actual) pairs needed for
calibration and live IC.

### File Location

```
data/prediction_log/
  ledger.parquet              # append-only, one row per (date, ticker, model)
  calibrator_latest.pkl       # most recent fitted isotonic regression
  calibration_history.json    # weekly snapshots of calibration curve parameters
  live_ic.json                # rolling IC metrics for dashboard consumption
```

`data/prediction_log/` is committed to the repo. The ledger grows by ~1,500 rows/day
(one per stock in the universe). At that rate, one year = ~375K rows ≈ 15–20 MB in
parquet. Manageable in git for the first year; revisit if the universe expands
significantly.

### Ledger Schema (`ledger.parquet`)

| Column | Type | Description |
|---|---|---|
| `date` | date | Prediction date (market date of inference run) |
| `ticker` | str | Stock ticker |
| `model_name` | str | e.g. `composite_v4`, `set_transformer_v1` |
| `pred_score` | float | Raw model output (predicted forward return, unnormalized) |
| `ml_score` | float | Cross-sectional percentile 0–100 (from predictions.csv) |
| `ml_signal` | str | Signal at time of prediction (STRONG BUY / BUY / HOLD / SELL / STRONG SELL) |
| `calibrated_score` | float | Calibrated P(positive return) — NULL until calibrator exists |
| `calibrated_signal` | str | Signal from calibrated thresholds — NULL until calibrator exists |
| `actual_return_10d` | float | Realized 10-day forward return — NULL until matured |
| `actual_return_63d` | float | Realized 63-day forward return — NULL until matured |
| `matured` | bool | True once actual returns have been backfilled |

**Partitioning:** Single file for now. If it grows past 100 MB, partition by year:
`ledger_2026.parquet`, `ledger_2027.parquet`.

**Multi-model support:** The `model_name` column means every model's predictions go into
the same ledger. Calibration and IC are always computed per-model.

---

## Part 2 — CLI Commands

### `alpha ml log` — Append Today's Predictions to Ledger

Called by GitHub Actions immediately after `alpha ml predict`.

```bash
alpha ml log --predictions data/latest/predictions.csv --ledger data/prediction_log/ledger.parquet
```

**Behavior:**

```python
# Pseudocode for alpha ml log
def log_predictions(predictions_path, ledger_path):
    preds = pd.read_csv(predictions_path)

    new_rows = pd.DataFrame({
        'date': pd.Timestamp.today().normalize(),
        'ticker': preds['ticker'],
        'model_name': preds['model_name'],
        'pred_score': preds['predicted_return'],
        'ml_score': preds['ml_score'],
        'ml_signal': preds['ml_signal'],
        'calibrated_score': None,       # filled by alpha ml calibrate
        'calibrated_signal': None,      # filled by alpha ml calibrate
        'actual_return_10d': None,      # filled by alpha ml backfill
        'actual_return_63d': None,      # filled by alpha ml backfill
        'matured': False,
    })

    if ledger_path.exists():
        ledger = pd.read_parquet(ledger_path)
        # Deduplicate: skip if (date, ticker, model_name) already exists
        existing_keys = set(zip(ledger['date'], ledger['ticker'], ledger['model_name']))
        new_rows = new_rows[
            ~new_rows.apply(lambda r: (r['date'], r['ticker'], r['model_name']) in existing_keys, axis=1)
        ]
        ledger = pd.concat([ledger, new_rows], ignore_index=True)
    else:
        ledger = new_rows

    ledger.to_parquet(ledger_path, index=False)
    print(f"Logged {len(new_rows)} predictions for {new_rows['date'].iloc[0]}")
```

**Idempotency:** If the same date/ticker/model combination already exists, skip it.
This means re-running the GitHub Actions job won't create duplicates.

### `alpha ml backfill` — Fill In Actual Returns for Matured Predictions

Called by GitHub Actions daily, after `alpha ml log`. Looks back at predictions that
are now old enough to have realized returns, fetches actual prices, and fills in the
`actual_return_10d` and `actual_return_63d` columns.

```bash
alpha ml backfill --ledger data/prediction_log/ledger.parquet
```

**Behavior:**

```python
# Pseudocode for alpha ml backfill
def backfill_returns(ledger_path):
    ledger = pd.read_parquet(ledger_path)
    today = pd.Timestamp.today().normalize()
    updated = 0

    # Find rows where 10-day return is missing and enough time has passed
    mask_10d = ledger['actual_return_10d'].isna() & (today - ledger['date'] >= pd.Timedelta(days=14))
    # Find rows where 63-day return is missing and enough time has passed
    mask_63d = ledger['actual_return_63d'].isna() & (today - ledger['date'] >= pd.Timedelta(days=90))

    # Use calendar days with buffer (14 cal days ≈ 10 trading days, 90 cal ≈ 63 trading)
    # to account for weekends/holidays

    tickers_to_fetch = set()
    if mask_10d.any():
        tickers_to_fetch |= set(ledger.loc[mask_10d, 'ticker'])
    if mask_63d.any():
        tickers_to_fetch |= set(ledger.loc[mask_63d, 'ticker'])

    if not tickers_to_fetch:
        print("No predictions ready for backfill")
        return

    # Fetch price data for all tickers that need backfill
    # Reuse existing yfinance batched download infrastructure from search.py
    earliest_date = ledger.loc[mask_10d | mask_63d, 'date'].min()
    prices = fetch_close_prices(list(tickers_to_fetch), start=earliest_date, end=today)
    # prices: dict of ticker -> pd.Series(date -> close_price)

    for idx in ledger.index[mask_10d]:
        row = ledger.loc[idx]
        ticker_prices = prices.get(row['ticker'])
        if ticker_prices is None:
            continue
        pred_date = row['date']
        # Find the close price on prediction date and 10 trading days later
        p0 = find_close_on_or_after(ticker_prices, pred_date)
        p1 = find_close_on_or_after(ticker_prices, pred_date + pd.Timedelta(days=14))
        if p0 is not None and p1 is not None and p0 > 0:
            ledger.at[idx, 'actual_return_10d'] = (p1 / p0 - 1.0) * 100.0
            updated += 1

    for idx in ledger.index[mask_63d]:
        row = ledger.loc[idx]
        ticker_prices = prices.get(row['ticker'])
        if ticker_prices is None:
            continue
        pred_date = row['date']
        p0 = find_close_on_or_after(ticker_prices, pred_date)
        p1 = find_close_on_or_after(ticker_prices, pred_date + pd.Timedelta(days=90))
        if p0 is not None and p1 is not None and p0 > 0:
            ledger.at[idx, 'actual_return_63d'] = (p1 / p0 - 1.0) * 100.0
            updated += 1

    # Mark rows as matured when both return columns are filled
    ledger['matured'] = ledger['actual_return_10d'].notna() & ledger['actual_return_63d'].notna()

    ledger.to_parquet(ledger_path, index=False)
    print(f"Backfilled {updated} return values")
```

**Price lookup helper — `find_close_on_or_after`:** Given a price series and a target date,
return the closing price on that date. If the target is a weekend/holiday, use the next
available trading day's close. If no price exists within 5 calendar days, return None.

**Batch efficiency:** Don't fetch prices one ticker at a time. Collect all tickers needing
backfill, determine the earliest prediction date, and do a single `yf.download()` call
for all of them covering `[earliest_date, today]`. This is the same batched download
pattern used in `search.py`'s `fetch_returns()`.

### `alpha ml calibrate` — Fit/Update the Calibration Layer

Runs after `alpha ml backfill`. Fits an isotonic regression on matured predictions
and writes the calibrator + updated signal thresholds.

```bash
alpha ml calibrate --ledger data/prediction_log/ledger.parquet \
                   --model composite_v4 \
                   --output data/prediction_log/calibrator_latest.pkl
```

**Behavior:**

```python
# Pseudocode for alpha ml calibrate
from sklearn.isotonic import IsotonicRegression
import joblib

def calibrate(ledger_path, model_name, output_path, history_path):
    ledger = pd.read_parquet(ledger_path)

    # Filter to matured predictions for this model
    mask = (ledger['model_name'] == model_name) & ledger['matured']
    matured = ledger.loc[mask].copy()

    if len(matured) < 500:
        print(f"Only {len(matured)} matured predictions — need ≥500 for reliable calibration")
        print("Skipping calibration, using static thresholds")
        return

    # Fit isotonic regression: ml_score → P(10d return > 0)
    X = matured['ml_score'].values
    y = (matured['actual_return_10d'] > 0).astype(float).values

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    calibrator.fit(X, y)

    # Save calibrator
    joblib.dump(calibrator, output_path)

    # Derive new signal thresholds from calibrated probabilities
    # Map: find the ml_score where calibrated probability crosses each threshold
    thresholds = derive_signal_thresholds(calibrator)

    # Save calibration history snapshot
    snapshot = {
        'date': str(pd.Timestamp.today().date()),
        'model': model_name,
        'n_samples': len(matured),
        'thresholds': thresholds,
        'brier_score': brier_score_loss(y, calibrator.predict(X)),
    }
    append_to_history(history_path, snapshot)

    # Backfill calibrated scores for all rows of this model (including non-matured)
    all_model = ledger['model_name'] == model_name
    ledger.loc[all_model, 'calibrated_score'] = calibrator.predict(
        ledger.loc[all_model, 'ml_score'].values
    )
    ledger.loc[all_model, 'calibrated_signal'] = ledger.loc[all_model, 'calibrated_score'].apply(
        lambda p: score_to_signal(p, thresholds)
    )

    ledger.to_parquet(ledger_path, index=False)
    print(f"Calibrated {model_name} on {len(matured)} samples, Brier={snapshot['brier_score']:.4f}")


def derive_signal_thresholds(calibrator):
    """
    Find ml_score cutoffs that correspond to meaningful probability levels.

    Instead of fixed ml_score thresholds (≥80 = STRONG BUY), we find the ml_score
    where the calibrated P(positive return) crosses each probability level.

    Default probability thresholds:
      STRONG BUY:  P ≥ 0.70  (70% chance of positive 10d return)
      BUY:         P ≥ 0.60
      HOLD:        P ≥ 0.45
      SELL:        P ≥ 0.35
      STRONG SELL: P < 0.35
    """
    test_scores = np.linspace(0, 100, 1000)
    calibrated_probs = calibrator.predict(test_scores)

    prob_thresholds = {
        'STRONG_BUY': 0.70,
        'BUY': 0.60,
        'HOLD': 0.45,
        'SELL': 0.35,
    }

    score_thresholds = {}
    for signal, prob_cutoff in prob_thresholds.items():
        # Find the lowest ml_score where calibrated prob >= cutoff
        mask = calibrated_probs >= prob_cutoff
        if mask.any():
            score_thresholds[signal] = float(test_scores[mask][0])
        else:
            score_thresholds[signal] = 100.0  # unreachable = never triggers

    return score_thresholds


def score_to_signal(calibrated_prob, thresholds):
    """Map calibrated probability to signal string."""
    if calibrated_prob >= thresholds.get('STRONG_BUY', 0.70):
        return 'STRONG BUY'
    elif calibrated_prob >= thresholds.get('BUY', 0.60):
        return 'BUY'
    elif calibrated_prob >= thresholds.get('HOLD', 0.45):
        return 'HOLD'
    elif calibrated_prob >= thresholds.get('SELL', 0.35):
        return 'SELL'
    else:
        return 'STRONG SELL'
```

**Minimum sample requirement:** 500 matured predictions. At ~1,500 stocks/day, this means
the calibrator first becomes usable after the first matured batch arrives (~14 calendar days
for 10d returns). It becomes reliable after ~1 month of accumulated data.

**Probability thresholds rationale:** The default 0.70/0.60/0.45/0.35 levels are starting
points. After calibration, STRONG BUY means "the model historically assigned scores in this
range to stocks that went up 70% of the time over 10 days." These probability cutoffs are
themselves tunable — but start with these and adjust based on observed precision.

### `alpha ml live-ic` — Compute Rolling Live IC for Dashboard

Computes live rank IC from the ledger and writes a JSON file for the frontend dashboard.

```bash
alpha ml live-ic --ledger data/prediction_log/ledger.parquet \
                 --model composite_v4 \
                 --output data/prediction_log/live_ic.json
```

**Behavior:**

```python
def compute_live_ic(ledger_path, model_name, output_path):
    ledger = pd.read_parquet(ledger_path)

    # Filter to matured predictions for this model
    mask = (ledger['model_name'] == model_name) & ledger['actual_return_10d'].notna()
    data = ledger.loc[mask].copy()

    if len(data) == 0:
        print("No matured predictions yet")
        return

    # Compute daily cross-sectional rank IC
    daily_ics = []
    for date, group in data.groupby('date'):
        if len(group) < 30:  # need enough stocks for meaningful correlation
            continue
        ic = group['ml_score'].corr(group['actual_return_10d'], method='spearman')
        daily_ics.append({
            'date': str(date.date()) if hasattr(date, 'date') else str(date),
            'ic': round(ic, 4),
            'n_stocks': len(group),
        })

    if not daily_ics:
        print("Not enough data for IC computation")
        return

    ic_series = pd.DataFrame(daily_ics)

    # Compute rolling 21-day IC (smoothed)
    ic_series['ic_rolling_21d'] = ic_series['ic'].rolling(21, min_periods=5).mean()

    # Summary stats
    summary = {
        'model': model_name,
        'updated_at': str(pd.Timestamp.today().date()),
        'total_days': len(ic_series),
        'ic_mean': round(ic_series['ic'].mean(), 4),
        'ic_std': round(ic_series['ic'].std(), 4),
        'icir': round(ic_series['ic'].mean() / ic_series['ic'].std(), 4) if ic_series['ic'].std() > 0 else 0,
        'ic_hit_rate': round((ic_series['ic'] > 0).mean(), 4),
        'latest_ic': round(ic_series['ic'].iloc[-1], 4),
        'latest_rolling_ic': round(ic_series['ic_rolling_21d'].iloc[-1], 4) if ic_series['ic_rolling_21d'].notna().any() else None,
    }

    output = {
        'summary': summary,
        'daily': daily_ics,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Live IC for {model_name}: mean={summary['ic_mean']}, "
          f"ICIR={summary['icir']}, hit_rate={summary['ic_hit_rate']}")

    # Retrain trigger warning
    if summary['latest_rolling_ic'] is not None and summary['latest_rolling_ic'] < 0.02:
        print(f"⚠️  WARNING: Rolling 21d IC ({summary['latest_rolling_ic']}) below 0.02 threshold")
        print("   Consider retraining if this persists for 2+ weeks")
```

---

## Part 3 — Updated GitHub Actions Workflow

### Changes to `daily-data.yml`

Three new steps added after the existing `alpha ml predict` step:

```yaml
name: Daily Data Update
on:
  schedule:
    - cron: '15 20 * * 1-5'  # 4:15pm ET weekdays (unchanged)
  workflow_dispatch:

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -e .

      # ── EXISTING STEPS (unchanged) ──────────────────────────
      - name: Run daily spread
        run: alpha spread --universe nasdaq --universe sp500 --output-dir data/latest

      - name: Run ML predictions
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: alpha ml predict --model composite_v4 --input data/latest/ --output data/latest/

      # ── NEW STEPS ───────────────────────────────────────────
      - name: Log predictions to ledger
        run: alpha ml log --predictions data/latest/predictions.csv --ledger data/prediction_log/ledger.parquet

      - name: Backfill matured returns
        run: alpha ml backfill --ledger data/prediction_log/ledger.parquet

      - name: Update calibration (if enough data)
        run: alpha ml calibrate --ledger data/prediction_log/ledger.parquet --model composite_v4 --output data/prediction_log/calibrator_latest.pkl

      - name: Compute live IC
        run: alpha ml live-ic --ledger data/prediction_log/ledger.parquet --model composite_v4 --output data/prediction_log/live_ic.json

      # ── EXISTING COMMIT STEP (updated paths) ────────────────
      - name: Commit data
        run: |
          git add data/latest/ data/prediction_log/
          git commit -m "chore: daily data update $(date +%Y-%m-%d)" || echo "No changes"
          git push
```

**Execution order matters:**
1. `predict` — generates today's predictions.csv
2. `log` — appends today's predictions to the ledger
3. `backfill` — looks at old predictions, fills in actual returns for matured ones
4. `calibrate` — refits isotonic regression on all matured data
5. `live-ic` — computes rolling IC from matured data, writes JSON for dashboard

Steps 3–5 are no-ops for the first ~14 days (no matured predictions yet).
Step 4 is a no-op until ≥500 matured rows exist (~1 day of matured data, since
each day produces ~1,500 rows — so effectively usable after 14 calendar days).

### Failure Handling

Each step should be resilient to missing data:
- `alpha ml log` — creates ledger.parquet if it doesn't exist
- `alpha ml backfill` — skips gracefully if no rows are ready for backfill
- `alpha ml calibrate` — prints "need ≥500 samples" and exits 0 if insufficient data
- `alpha ml live-ic` — prints "no matured predictions" and exits 0 if insufficient data

No step should exit non-zero for "not enough data yet" conditions. Only exit non-zero
for actual errors (file I/O failures, corrupt parquet, etc.).

---

## Part 4 — Applying Calibrated Signals to predictions.csv

Once the calibrator exists, `alpha ml predict` should also apply it to today's predictions
before writing `predictions.csv`. This adds two columns to the existing schema:

### Updated predictions.csv Schema

Existing columns unchanged. Two new columns added:

| Column | Type | Description |
|---|---|---|
| `calibrated_score` | float | P(positive 10d return), 0.0–1.0. NULL if no calibrator yet. |
| `calibrated_signal` | str | Signal from calibrated thresholds. NULL if no calibrator yet. |

**Implementation in `alpha ml predict`:**

```python
# After computing ml_score and ml_signal (existing logic)...

calibrator_path = Path('data/prediction_log/calibrator_latest.pkl')
if calibrator_path.exists():
    calibrator = joblib.load(calibrator_path)
    preds['calibrated_score'] = calibrator.predict(preds['ml_score'].values)

    # Load latest thresholds from calibration history
    history = json.load(open('data/prediction_log/calibration_history.json'))
    latest_thresholds = history[-1]['thresholds']
    preds['calibrated_signal'] = preds['calibrated_score'].apply(
        lambda p: score_to_signal(p, latest_thresholds)
    )
else:
    preds['calibrated_score'] = None
    preds['calibrated_signal'] = None
```

**Frontend behavior:** The React dashboard should prefer `calibrated_signal` over `ml_signal`
when it exists. Display logic:

```typescript
// In the signal display component
const signal = stock.calibrated_signal ?? stock.ml_signal;
const score = stock.calibrated_score != null
  ? `${(stock.calibrated_score * 100).toFixed(0)}% confidence`
  : `Score: ${stock.ml_score}`;
```

The `SignalBanner` component already handles all five signal values with correct color
coding — no changes needed to the component itself, just the data source.

---

## Part 5 — Dashboard Integration

### Live IC Chart on Backtest Page

`live_ic.json` feeds directly into the existing "IC Over Time" chart on the `/backtest`
page. The chart currently shows only historical backtest IC. Add a second trace for live IC:

```typescript
// In the IC Over Time chart component
const backtestIC = backtestData.ic_over_time;  // existing
const liveIC = liveICData?.daily;               // new, from live_ic.json

// Combine into one chart with two visually distinct traces
const traces = [
  {
    x: backtestIC.map(d => d.month),
    y: backtestIC.map(d => d.ic),
    name: 'Backtest IC',
    line: { color: '#4299e1', dash: 'dot' },
  },
  {
    x: liveIC.map(d => d.date),
    y: liveIC.map(d => d.ic),
    name: 'Live IC',
    line: { color: '#48bb78' },
  },
];
```

### Live IC Summary Card

Add to the Model Leaderboard tab, below the existing model metrics:

| Metric | Source | Display |
|---|---|---|
| Live IC (mean) | `live_ic.json → summary.ic_mean` | "0.052" |
| Live ICIR | `live_ic.json → summary.icir` | "1.23" |
| Live Hit Rate | `live_ic.json → summary.ic_hit_rate` | "64%" |
| Days Tracked | `live_ic.json → summary.total_days` | "47 days" |
| Status | derived from `latest_rolling_ic` | 🟢 Healthy / 🟡 Degrading / 🔴 Retrain |

Status logic:
- 🟢 Healthy: `latest_rolling_ic >= 0.03`
- 🟡 Degrading: `0.02 <= latest_rolling_ic < 0.03`
- 🔴 Retrain: `latest_rolling_ic < 0.02`

### Calibration Curve Visualization

On the backtest page, add a "Calibration" sub-section (below Feature Importance tab or
as a 5th tab) showing:

1. **Reliability diagram:** predicted probability (x-axis) vs observed frequency of
   positive returns (y-axis), 10 bins. Perfect calibration = diagonal line.
2. **Threshold drift chart:** from `calibration_history.json`, plot how the ml_score
   cutoff for each signal level has shifted over time. Shows regime adaptation.

Data source: `calibration_history.json` (array of weekly snapshots).

### Screener — Calibrated Signals (Silent Upgrade)

No new UI elements. The screener automatically prefers `calibrated_signal` over
`ml_signal` when a calibrator exists. The signal badge (STRONG BUY / BUY / etc.)
and its color remain unchanged — only the threshold that triggers each level adapts.

```typescript
// In the signal display component (screener stock card / table row)
const signal = stock.calibrated_signal ?? stock.ml_signal;
const score  = stock.calibrated_score != null
  ? `${(stock.calibrated_score * 100).toFixed(0)}% win probability`
  : `Score: ${stock.ml_score}`;
```

No data migration or schema change needed in the existing CSV parser — the new columns
(`calibrated_score`, `calibrated_signal`) are optional and fall back gracefully.

### Forecast Page — Model Health Pills

Each model toggle chip in `ModelControls` gets a small health indicator dot driven by
`live_ic.json`. When `live_ic.json` exists for a model, render a colored dot next to
the model label. On hover, show a tooltip with the rolling 21d IC value.

```typescript
// Status derived from latest_rolling_ic (same thresholds as leaderboard card)
// 🟢  rolling_ic >= 0.03   — healthy
// 🟡  0.02 <= rolling_ic < 0.03  — degrading
// 🔴  rolling_ic < 0.02   — retrain recommended
// ⚪  no live_ic.json yet — not enough data
```

**Component location:** `ModelControls.tsx` — add an optional `modelHealth` prop
(map of model_id → `'healthy' | 'degrading' | 'retrain' | 'unknown'`). The parent
`Forecast/index.tsx` fetches `./data/prediction_log/live_ic_{model_id}.json` for each
active model on mount (same pattern as `useForecastCalibration`).

**Why here:** Before trusting a fan chart, a user should know if the underlying model
is currently performing at backtest levels or has drifted. The health dot answers that
without requiring a trip to the Backtest page.

---

## Summary: UI Placement

| Feature | Page | Component | Data source |
|---|---|---|---|
| Live IC chart (2nd trace) | `/backtest` | IC Over Time chart | `live_ic.json → daily` |
| Model health status card | `/backtest` | Leaderboard tab | `live_ic.json → summary` |
| Calibration reliability diagram | `/backtest` | New "Calibration" tab (5th) | `calibration_history.json` |
| Threshold drift chart | `/backtest` | New "Calibration" tab (5th) | `calibration_history.json` |
| Calibrated signal badges | `/` (Screener) | Stock table / card | `predictions.csv` (new columns) |
| Model health dots + tooltip | `/forecast` | `ModelControls` chips | `live_ic_{model}.json → summary` |

---

## Part 6 — Multi-Model Support

The ledger is designed to hold predictions from any model. When new models come online
(MLP, Set Transformer, ensembles), each one:

1. Writes its own `model_name` into predictions.csv
2. Gets logged into the same ledger by `alpha ml log`
3. Gets its own calibrator fitted by `alpha ml calibrate --model <name>`
4. Gets its own live IC computed by `alpha ml live-ic --model <name>`

To run calibration and IC for all models at once:

```bash
# In daily-data.yml, replace the single-model calibrate/live-ic steps with:
- name: Calibrate and compute IC for all models
  run: |
    for model in $(alpha ml list-models --ledger data/prediction_log/ledger.parquet); do
      alpha ml calibrate --ledger data/prediction_log/ledger.parquet --model "$model" \
        --output "data/prediction_log/calibrator_${model}.pkl"
      alpha ml live-ic --ledger data/prediction_log/ledger.parquet --model "$model" \
        --output "data/prediction_log/live_ic_${model}.json"
    done
```

`alpha ml list-models` is a trivial helper that reads unique `model_name` values from
the ledger.

---

## Part 7 — Implementation Sequence

Execute in this order. Each block is a self-contained Claude Code session.

### Session 1: Ledger Infrastructure

**Files to create/modify:**
- `morningalpha/ml/ledger.py` — `log_predictions()`, `backfill_returns()` functions
- `morningalpha/ml/calibrate.py` — `calibrate()`, `derive_signal_thresholds()`, `compute_live_ic()`
- `morningalpha/main.py` — register `alpha ml log`, `alpha ml backfill`, `alpha ml calibrate`, `alpha ml live-ic` subcommands

**Dependencies to add:** `scikit-learn` (for `IsotonicRegression`), `joblib` (likely already
available via scikit-learn)

**Testing:**
1. Create a synthetic ledger with 1,000 fake predictions and known actual returns
2. Verify `log_predictions` is idempotent (run twice, same result)
3. Verify `backfill_returns` only fills rows where enough time has elapsed
4. Verify `calibrate` produces a monotonically increasing calibration curve
5. Verify `live-ic` JSON matches expected schema

### Session 2: GitHub Actions Integration

**Files to modify:**
- `.github/workflows/daily-data.yml` — add the 4 new steps after `alpha ml predict`

**Testing:**
- Run the workflow manually via `workflow_dispatch`
- Verify `ledger.parquet` is created on first run
- Verify steps 3–5 exit 0 with "not enough data" messages (no matured predictions yet)
- Verify `data/prediction_log/` is committed

### Session 3: Predictions.csv Calibration Pass

**Files to modify:**
- `morningalpha/ml/predict.py` — add calibrator loading and `calibrated_score`/`calibrated_signal` columns

**Testing:**
- Without calibrator present: `calibrated_score` and `calibrated_signal` are NULL
- With calibrator present: values are populated, signals may differ from uncalibrated

### Session 4: Dashboard Integration

**Files to modify:**
- Backtest page component — add live IC trace to IC Over Time chart
- Backtest page component — add Live IC summary card to leaderboard tab
- Stock detail / screener — prefer `calibrated_signal` over `ml_signal`

**Data files consumed:**
- `data/prediction_log/live_ic.json`
- `data/prediction_log/calibration_history.json`
- `data/latest/predictions.csv` (updated schema with calibrated columns)

---

## Timeline & Expectations

| Milestone | When | What Unlocks |
|---|---|---|
| Ledger starts accumulating | Day 1 (after Session 1 deploys) | Nothing yet — just collecting data |
| First 10d returns mature | ~Day 14 | `backfill` starts filling actuals |
| Calibrator first fits | ~Day 15 (500+ matured rows) | `calibrated_signal` appears in predictions.csv |
| Live IC becomes meaningful | ~Day 30 (21+ days of matured ICs) | Rolling IC chart has enough data to show trends |
| Calibrator becomes reliable | ~Day 60 | Enough regime variation to trust threshold drift |
| Retrain trigger possible | ~Day 90+ | Enough history to detect sustained IC degradation |

**First 2 weeks:** The system is in "listen mode" — logging predictions, nothing to
calibrate yet. This is expected and correct.

**Key insight:** The calibration layer improves continuously without any manual
intervention. Once deployed, every trading day that passes makes the signals more
reliable. This is the "online learning" behavior — not the model changing, but our
interpretation of its output becoming better calibrated to current market conditions.

---

## Kill Criteria / Guardrails

- **If calibrated signals are worse than uncalibrated:** Compare Brier score of calibrated
  vs. raw ml_score thresholds. If calibrated Brier is higher (worse) for 4 consecutive
  weeks, disable the calibration layer and revert to static thresholds. Log the event.

- **If the calibrator has too few samples for a regime:** The isotonic regression can
  overfit to a short period. Require ≥500 matured samples before using, and ≥2,000
  before trusting threshold drift signals.

- **If live IC is persistently negative:** This means the model is anti-predictive.
  Possible causes: data pipeline bug (stale features), market regime completely outside
  training distribution, or feature leakage was present in training but not in live.
  Investigate before retraining — retraining on a broken pipeline just learns the bug.
