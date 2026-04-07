# Model Deployment & Daily Calibration Playbook

**Date:** 2026-04-07  
**Status:** Current architecture + weekend retraining cadence

---

## Overview

```
HPC Training (weekend)
        │
        ▼
  models/*.pkl / *.pt          ← checkpoints committed to repo
        │
        ▼
  alpha ml score (daily)       ← GitHub Actions 4:15pm ET
        │
        ├── data/latest/ticker_index.json     (raw scores, all tickers)
        ├── data/factors/predictions_ledger.parquet  (predictions log)
        ├── data/factors/model_weights.json   (ensemble weights, hedge-updated)
        └── data/latest/forecast_paths_*.json (LSTM fan charts)
        │
        ▼
  alpha ml score --calibrate (weekly, Fridays 5pm ET)
        │
        ├── data/factors/live_ic_*.json       (per-model realized IC)
        └── data/factors/calibrator_*.pkl     (isotonic calibrators)
        │
        ▼
  GitHub Pages (pages.yml)     ← React dashboard reads data/latest/
```

---

## Step 1: Promote New Models After Overnight Training

After HPC training completes and checkpoints are pulled to local:

### 1a. Inspect results
```bash
# Check test IC for all new models
cat results/lgbm_breakout_v10_wfcv.csv | tail -5
cat results/lgbm_composite_v10_wfcv.csv | tail -5
cat results/lgbm_medium_v1_wfcv.csv | tail -5

# LSTM: check WFCV summary in train_lstm.log
grep "Mean IC" train_lstm.log
```

### 1b. Decide what to promote
Promotion criteria (rough guidelines):
- LGBM: test IC ≥ 0.25 AND improvement over current champion
- LSTM: WFCV IC-63d ≥ 0.15 AND fold 6 IC > +0.05 (recent regime)
- If both pass: promote both and update ensemble weights
- If only one passes: promote that one, keep other champion running

### 1c. Update `models/config.json`
```bash
# Manually edit to set new champion(s)
# Or use:
alpha ml promote --model lgbm_breakout_v10
alpha ml promote --model lstm_clip_v3
```

`config.json` structure:
```json
{
  "champion": "lgbm_breakout_v10",
  "models": [
    { "id": "lgbm_breakout_v10",  "type": "lgbm", "status": "champion" },
    { "id": "lgbm_composite_v10", "type": "lgbm", "status": "champion" },
    { "id": "lgbm_medium_v1",     "type": "lgbm", "status": "champion" },
    { "id": "lstm_clip_v3",       "type": "lstm", "status": "champion" }
  ]
}
```

### 1d. Backfill predictions ledger
After promoting, repopulate the ledger with fresh scores from the new models
so calibration has a clean baseline:
```bash
alpha ml backfill --lookback 90
```
This rescores the last 90 trading days using the new model weights and appends
to `data/factors/predictions_ledger.parquet`. Run once per promotion event.

---

## Step 2: Daily Pipeline (Automated — GitHub Actions)

**Trigger:** Weekdays 4:15pm ET (20:15 UTC), after market close.

**What runs:**
```
alpha spread --universe nasdaq --universe nyse --universe sp500
alpha ml score --data-dir data/latest --models-dir models
git commit + push
```

**What `alpha ml score` does each day:**
1. Loads all champion models from `models/config.json`
2. Scores every ticker in the universe → `ticker_index.json`
3. Appends today's predictions to `predictions_ledger.parquet`
   - Records `(date, ticker, model_id, raw_score, horizon)` for 5d/13d/63d
4. As realized returns come in (5d later, 13d later, 63d later), the ledger
   fills in actuals automatically on subsequent runs
5. Updates ensemble weights via hedge algorithm using recent IC
   - Writes `model_weights.json` with new weights
6. Generates LSTM forecast paths → `forecast_paths_lstm_clip_v3.json`

**What you see in the dashboard:**
- Updated `InvestmentScore` and `CalibratedSignal` for all stocks
- LSTM fan charts reflect today's sequence cache
- Model weights reflect the last hedge update

---

## Step 3: Daily Calibration (Within `alpha ml score`)

Calibration runs automatically as part of the daily score. Two layers:

### Layer 1: Ensemble weight update (daily, hedge algorithm)
- Reads `predictions_ledger.parquet` for the last 63d of realized returns
- Computes IC per model per horizon
- Updates `model_weights.json` using exponential hedge:
  ```
  weight_i ← weight_i × exp(η × IC_i)
  weights ← normalize(weights)
  ```
- Weights converge toward models with consistently positive recent IC

### Layer 2: Isotonic calibration (weekly, Fridays)
The `weekly-ml.yml` workflow runs `alpha ml score --calibrate` on Fridays:
- Fits isotonic regression: `raw_score → P(positive return)`  
- Saves `calibrator_{model_id}.pkl` per model
- Computes `live_ic_{model_id}.json` with per-decile hit rates
- Dashboard `CalibratedSignal` field is blank until ~13 weeks of ledger data 
  accumulates (need enough realized outcomes to fit calibrator reliably)

**Current state:** Ledger was reset 2026-04-07 (clean baseline, full 3000-ticker
universe + all champion models). Calibrated signals will appear ~mid-July 2026.

---

## Weekend Retraining Cadence

### When to retrain
| Trigger | Action |
|---------|--------|
| Monthly (first weekend of month) | Extend dataset + retrain LGBM |
| Quarterly (first weekend of quarter) | Full dataset rebuild + retrain all models |
| Live IC drops below 0.05 for 2+ weeks | Emergency retrain |
| New feature added to spread | Retrain all models |

### Saturday morning: Extend dataset
```bash
# On HPC — add the latest month of daily data to dataset.parquet
alpha ml dataset --extend --output data/training/dataset.parquet
```

### Saturday: Retrain LGBM (CPU nodes, ~5 hours)
```bash
# SLURM job
alpha ml train lgbm --target forward_63d_rank --name lgbm_breakout_v11 --n-trials 30
alpha ml train lgbm --target forward_63d_composite_rank --name lgbm_composite_v11 --n-trials 30
alpha ml train lgbm --target forward_21d_rank --name lgbm_medium_v2 --n-trials 30
```

### Saturday/Sunday: Retrain LSTM (GPU node, ~4 hours)
```bash
# SLURM job — GPU partition
alpha ml train lstm --target-mode clip \
    --walk-forward --wf-finetune \
    --name lstm_clip_v4 \
    --dataset data/training/dataset.parquet
```

### Sunday evening: Evaluate + promote
```bash
# Review logs, decide which models pass promotion criteria
# Update models/config.json
# Backfill ledger
alpha ml backfill --lookback 90
# Commit checkpoints and config
git add models/ && git commit -m "chore(ml): promote vN models"
git push
```

### Monday: Verify
- Daily workflow runs at 4:15pm ET with new models
- Check `ticker_index.json` — all models should be scoring
- Check `model_weights.json` — weights initialized from backfill IC
- Dashboard should reflect new model scores

---

## LSTM Sequence Cache (Pending)

The LSTM currently requires `dataset.parquet` (3.4GB) for inference, which
is not available in GitHub Actions. Until the sequence cache is implemented,
LSTM scores are only generated when running locally or on HPC.

**Planned fix:** Seed `data/latest/lstm_sequence_cache.parquet` from the last
63 rows per ticker from `dataset.parquet`. Daily workflow appends new spread
data to this cache instead of loading the full dataset.

**Status:** Phase 1 (seed script) pending — blocked on new dataset pull after
the extend bug fix.

---

## File Reference

| File | Purpose | Updated |
|------|---------|---------|
| `models/config.json` | Champion model registry | Manually on promotion |
| `models/*.pkl` | LGBM checkpoints | Weekend retrain |
| `models/*.pt` | LSTM checkpoints | Weekend retrain |
| `data/factors/predictions_ledger.parquet` | Prediction log + realized returns | Daily |
| `data/factors/model_weights.json` | Ensemble weights (hedge) | Daily |
| `data/factors/live_ic_*.json` | Per-model realized IC | Weekly (Fridays) |
| `data/factors/calibrator_*.pkl` | Isotonic calibrators | Weekly (Fridays) |
| `data/latest/ticker_index.json` | All ticker scores | Daily |
| `data/latest/forecast_paths_*.json` | LSTM fan charts | Daily |
