# MorningAlpha — ML Module Plan

**Last updated:** 2026-03-27

---

## Core Objective

> **Find explosive breakout stocks early — when they're up 30–50% and heading to 100%+ — and surface warning signals before they deflate.**

The traditional screener (`alpha spread`) is already good at identifying quality uptrends. It surfaces names like SNDK, AXTI, FSLY, and ERAS. The ML module's job is not to find *different* stocks — it is to:

1. **Score those same candidates earlier and with more conviction** — before the screener's trailing metrics catch up
2. **Rank them relative to each other** — among 10 screener picks, which 2–3 are the best to act on now?
3. **Provide timing/exit signals** — `MLScoreDelta` (falling score on a held stock = warning to reduce or exit)

The mental model is a "Wall Street Bets"-style signal layered on top of a fundamental screener. Think of it as: the screener finds the universe of candidates; the ML model assigns conviction and timing.

**What this is NOT:**
- A replacement for the traditional screener
- A long/short hedge fund model (L/S metrics are secondary)
- A value or quality model (those systematically underrank the stocks we want)

---

## System Architecture

```
alpha spread (daily)
  └── stocks_3m.csv + fundamentals.csv
        │
        ▼
alpha ml score (daily, after spread)
  ├── Momentum gate: price > SMA200 AND MomentumAccel > 0
  ├── Scores each stock with all registered models
  ├── Consensus: mean raw score → re-ranked percentile
  ├── Sector diversity cap: limits per-sector representation
  ├── MLScoreDelta: compare to prior run cache (data/factors/mlscore_cache.parquet)
  └── Merges MLScore, MLScoreDelta, MLScore_{model_id} into all period CSVs
        │
        ▼
React dashboard (morningalpha/web/)
  └── Displays MLScore alongside screener metrics
```

**Key files:**
- `morningalpha/ml/features.py` — feature definitions (FEATURE_COLUMNS, MARKET_CONTEXT_COLUMNS)
- `morningalpha/ml/dataset.py` — builds point-in-time training dataset from OHLCV history
- `morningalpha/ml/train.py` — training pipeline (LightGBM + Optuna, purged k-fold CV)
- `morningalpha/ml/inference.py` — live feature construction + scoring
- `morningalpha/ml/score.py` — production scoring command
- `models/config.json` — model registry (champion + candidates)

---

## Feature Set

Features are organized into four tiers:

**Technical (per-stock, rank-normalized cross-sectionally):**
Sharpe, Sortino, MaxDrawdown, RSI, Momentum12_1, log_momentum_12_1, MomentumAccel,
VolumeTrend, VolumeConsistency, BollingerPctB, MACD, StochK/D, ATR14,
PriceToSMA{20/50/200}, PriceVs52wkHigh, info_discreteness, rs_rating, and derived
cross-sectional features (sector_return_rank, sector_momentum_rank, return_pct_x_regime)

**Market context (constant per date, winsorized but NOT rank-normalized):**
SPY: return_10d, return_21d, vol_20d, RSI_14, above_SMA200, momentum_regime
VIX: vix_level, vix_1m_change, vix_term_structure
WML: wml_realized_vol_126d, wml_trailing_1m, wml_trailing_3m

**Fundamental (per-stock, rank-normalized):**
earnings_yield, book_to_market, ROE, revenue_growth, profit_margin, debt_to_equity,
short_pct_float, earnings_yield_vs_sector, book_to_market_vs_sector

**Categorical:**
sector, exchange, market_cap_cat, has_fundamentals

---

## Training Targets Available

| Target | Description |
|--------|-------------|
| `forward_63d_rank` | Cross-sectional percentile rank of raw 63-day return. No quality adjustment. |
| `forward_63d_composite_rank` | Blended rank: 30% return + 35% Sharpe + 20% consistency + 15% drawdown. Rewards smooth, consistent outperformance. |
| `forward_63d_market_excess_rank` | Rank of (stock return − SPY return) over 63 days. Rewards beating the market. |
| `forward_63d_sector_relative_rank` | Rank of (stock return − sector median return). Rewards beating sector peers. |

---

## All Experiments (Chronological)

### Exp 1: lgbm_composite_63d
- **Target:** `forward_63d_composite_rank`
- **Features:** 66 (no VIX/WML/info_discreteness)
- **IC:** 0.187 | L/S Sharpe: 1.421 | MaxDD: −7.3%
- **Result:** Solid baseline. Superseded by v2.

### Exp 2: lgbm_sector_relative_63d
- **Target:** `forward_63d_sector_relative_rank`
- **Features:** 66
- **IC:** 0.167 | L/S Sharpe: 0.54 | ICIR: 1.966 (best of early models)
- **Result:** Good ICIR but regime drawdowns without VIX/WML. VIX/WML added in later experiments.

### Exp 3a: lgbm_composite_v2 *(prior champion)*
- **Target:** `forward_63d_composite_rank`
- **Features:** 73 (added VIX/WML + info_discreteness)
- **IC:** 0.200 | L/S Sharpe: 2.33 | MaxDD: **0.0%**
- **Result:** Best risk-adjusted performance. VIX/WML was the key — regime awareness eliminated drawdowns.
  The WML regime features ranked #1 and #2 in SHAP importance.

### Exp 3b: lgbm_composite_v3
- **Target:** `forward_63d_composite_rank`
- **Features:** 73 (added raw fundamentals: PE, PB, short_pct_float)
- **IC:** 0.181 | L/S Sharpe: 0.933 | MaxDD: −20.6%
- **Result:** Failed. Fundamentals created a value tilt. `short_pct_float` caused squeeze exposure
  in the Jan 2025 momentum rally. Raw fundamentals are dangerous in L/S context.

### Exp 3c: lgbm_sector_relative_v2
- **Target:** `forward_63d_sector_relative_rank`
- **Features:** 73 (with fundamentals)
- **IC:** 0.164 | L/S Sharpe: 0.02 | MaxDD: −47.6%
- **Top-decile Sharpe: 1.72 | Consistency: 0.823** ← best long-only metrics of any model
- **Result:** L/S completely failed (fundamentals dominated, value-short in growth market). But long-only
  metrics were excellent. Useful as a long-only screener signal, not for L/S.

### Exp 4: lgbm_breakout_63d *(current champion)*
- **Target:** `forward_63d_market_excess_rank`
- **Filter:** `--momentum-universe` (momentum_12_1 > 10%, < 400%, price > SMA200, return_pct > 20%)
- **IC:** 0.129–0.136 | Positive persistence IC: +0.016
- **Result:** Only model with positive persistence IC (momentum continuation signal). But the 400% cap
  in `--momentum-universe` silently excluded AXTI (2234%), SNDK (1139%), ERAS (817%) from training.
  In live scoring, these stocks score 0.2–3.4/100.

### Exp 5: lgbm_breakout_v2 *(current champion, promoted 2026-03-27)*
- **Target:** `forward_63d_rank` (raw return rank — no quality penalty)
- **Filter:** None (all stocks, no momentum-universe)
- **IC:** 0.176
- **Dataset:** Rebuilt with `log_momentum_12_1` feature added
- **Result:** Improved FSLY (13.8 → 57.3), SNDK (0.2 → 44.1), ERAS (3.4 → 48.6), AXTI (0.3 → 47.6).
  But ZIM (shipping, +27% return, -0.24 MomentumAccel) still scored 100/100 — unanimously #1 across
  all 7 models. Root cause: shipping stocks have textbook-clean technical profiles (low drawdown,
  steady trend) that all LightGBM models reward regardless of target.

---

## Current Scoring Pipeline (as of 2026-03-27)

**Active models (7 total):**

| Model | Target | Filter | IC |
|-------|--------|--------|----|
| lgbm_breakout_v2 | forward_63d_rank | none | 0.176 |
| lgbm_breakout_63d | forward_63d_market_excess_rank | momentum-universe | 0.136 |
| lgbm_composite_v2 | forward_63d_composite_rank | none | 0.200 |
| lgbm_composite_v3 | forward_63d_composite_rank | none | 0.181 |
| lgbm_composite_63d | forward_63d_composite_rank | none | 0.187 |
| lgbm_sector_relative_v2 | forward_63d_sector_relative_rank | none | 0.164 |
| lgbm_sector_relative_63d | forward_63d_sector_relative_rank | none | 0.167 |

**Momentum gate (score.py):** `price > SMA200 AND MomentumAccel > 0`
Removes 1,538 of 2,128 stocks (downtrending + fading). ZIM and GSL both filtered here.

**Sector diversity cap:**
Technology: 15, Healthcare: 10, Financial Services: 10, Consumer Cyclical: 7,
Communication Services: 7, Industrials: 3, Energy: 3, Utilities: 2, default: 5

---

## Current Challenges

### 1. Consensus is dominated by wrong-objective models
5 of 7 active models optimize for composite quality (smooth returns, low drawdown). These
systematically penalize the volatile explosive stocks we actually want. The consensus score
for ERAS/AXTI is ~32–36/100 while ZIM/GSL (now gated out) were 99–100/100.

**Root cause:** Low-drawdown stocks are abundant in training data. The model learns this
pattern overwhelmingly. Raw return rank (`forward_63d_rank`) helps, but most models still
use quality targets.

### 2. Extreme momentum stocks underrepresented in training data
The training dataset (2019–2025) has essentially zero examples of stocks with momentum_12_1
> 400% that also pass quality filters. Stocks like AXTI at 2234% momentum are completely
outside the training distribution. The model cannot score what it has never seen.

**Evidence:** After raising the momentum_universe cap from 4x to 50x, zero new training rows
were added — the historical data simply doesn't contain those patterns at scale.

### 3. LightGBM only sees individual stock features
The model has no concept of "anomalous relative to peers." AXTI at 2234% momentum is
extraordinary within Technology — but LightGBM sees only AXTI's own features, not how they
compare to the sector cohort. `sector_momentum_rank` partially addresses this, but it's a
single aggregate vs. rich cross-stock attention.

### 4. Single consensus score conflates objective
Blending breakout and quality scores into one number produces a signal that is neither —
a breakout stock gets dragged down by quality models, a quality stock gets dragged down by
breakout models. The user can't tell if a low score means "bad breakout setup" or "bad
quality profile."

---

## Lessons Learned

1. **Composite target is wrong for explosive stocks** — it penalizes exactly the features
   (high drawdown, lumpy returns) that characterize the stocks we want.
2. **Raw fundamentals create value tilt** — raw P/E, P/B break L/S. Sector-relative
   fundamentals (earnings_yield_vs_sector) are safer.
3. **short_pct_float is dangerous** — squeeze risk in L/S; exclude from all future models.
4. **VIX/WML features are essential** — they gave lgbm_composite_v2 0% max drawdown. Must
   be in every model going forward. Currently only in inference, NOT in training dataset yet.
5. **MomentumAccel > 0 is a clean gate** — better than sector caps for removing stocks that
   have peaked. ZIM/GSL both have negative MomentumAccel despite clean technical profiles.
6. **MLScoreDelta is the right exit signal** — falling score on a held position is the warning.
7. **IC alone is insufficient** — lgbm_breakout_63d has lower IC (0.136) than v2 (0.200) but
   positive persistence IC (+0.016), which better matches the use case.

---

## Next Steps — Options

### Option A: Narrow the consensus to breakout-only (quick win)
Remove the 5 composite/quality models from active scoring. Run only lgbm_breakout_v2
(champion) and lgbm_breakout_63d. The composite models can remain in `config.json` as
candidates with `"status": "retired"` for reference.

**Pros:** Immediate improvement. FSLY, SNDK, SNDK scores already 43–56 with breakout_v2.
**Cons:** Loses the quality/risk signal entirely. A high breakout score on a deteriorating
business would go undetected.
**Effort:** 10 minutes.

### Option B: Separate scores — Breakout + Quality, shown independently
Instead of one consensus `MLScore`, compute two scores:
- `MLScore_Breakout`: average of breakout model(s) only
- `MLScore_Quality`: average of composite model(s) only

Show both in the dashboard. A conviction pick needs both scores to be high. A high breakout
+ low quality score = speculative. A high quality + low breakout score = hold, not chase.

**Pros:** Preserves both signals, gives user richer information. Naturally surfaces the
AXTI vs. GSL distinction (AXTI: breakout high, quality moderate; GSL: both moderate).
**Cons:** More complexity in the UI. Two numbers to interpret.
**Effort:** ~2 hours (score.py + React dashboard columns).

### Option C: Train a growth-quality model (targeted fundamentals)
Train a new model explicitly for growth stocks — using revenue_growth, ROE, and momentum
features but NOT P/E, P/B, or short_pct_float. Target: `forward_63d_rank`.

This separates "is the business growing?" from "is the stock cheap?" — only the former
matters for breakout stocks.

```bash
alpha ml train \
  --model lgbm \
  --target forward_63d_rank \
  --name lgbm_growth_quality \
  --exclude-features earnings_yield,book_to_market,sales_to_price,earnings_yield_vs_sector,book_to_market_vs_sector,short_pct_float \
  --n-trials 30
```

**Pros:** Adds fundamental signal without value tilt. Revenue growth + ROE are legitimate
signals for momentum stocks (real business vs. pure technical move).
**Cons:** Fundamentals coverage is incomplete (~60% of stocks). Another model in an already
crowded consensus.
**Effort:** 1 training run (~30 min).

### Option D: Add VIX/WML features to training dataset
Currently VIX term structure and WML factor features are computed at INFERENCE TIME (today's
values) but are NOT in the training dataset (dataset.py). This means models are trained
with these features as zeros and learn nothing from them. Adding them to dataset.py requires
fetching historical VIX and WML data for every snapshot date.

**Pros:** Models could actually learn regime-conditional patterns. VIX/WML was the #1 and #2
SHAP feature in v2 (even though they were computed live). With historical data, the signal
could be significantly stronger.
**Cons:** Complex to implement — requires merging time-series data into point-in-time snapshots.
**Effort:** ~4 hours (dataset.py + Ken French historical data pipeline).

### Option E: Set Transformer (cross-stock attention)
The architecture that addresses Challenge #3 directly. Instead of scoring each stock
independently, the transformer attends over a sector cohort and learns "is this stock
anomalous relative to its peers?"

**Why this matters for the objective:** AXTI at 2234% momentum is extraordinary within
Technology. A transformer attending over the 50 Technology stocks in the current universe
would encode that it's a 4-sigma outlier; LightGBM just sees normalized rank values.

**Prerequisites before building this:**
1. Option A or B is in place (clean scoring baseline)
2. Training target is stable (`forward_63d_rank` appears correct)
3. LightGBM hits a visible IC ceiling on that target
4. Dataset includes VIX/WML historical features (Option D)

**Architecture sketch:**
```
Input: sector cohort of N stocks × F features per snapshot date
→ Linear projection → Positional encoding (none — set, not sequence)
→ Multi-head self-attention (each stock attends to all peers)
→ Per-stock output head → forward_63d_rank prediction
```

**Effort:** 1–2 weeks (model code + training infrastructure + evaluation).

---

## Recommended Path

**Immediate (this week):**
1. **Option A** — retire composite models from consensus scoring, run breakout_v2 + breakout_63d only
2. Observe live scoring for 1–2 weeks; verify target stocks score 60–80+ range

**Short-term (2–4 weeks):**
3. **Option D** — add VIX/WML historical data to training dataset; retrain breakout_v2
4. **Option B** — add separate Breakout / Quality score columns to the dashboard

**Medium-term (1–2 months):**
5. **Option C** — growth-quality model once fundamental signal is better understood
6. Begin **Option E** (Set Transformer) once LightGBM IC plateaus

---

## Open Questions

- **What IC ceiling should trigger the transformer?** LightGBM IC of ~0.18 is decent but not
  remarkable. If it can't exceed 0.25 after adding VIX/WML historical data, the transformer
  is likely the next meaningful lever.

- **Should the momentum gate threshold be dynamic?** Current gate is static (MomentumAccel > 0).
  In high-VIX regimes, a tighter gate (MomentumAccel > 0.5) might be appropriate. In low-VIX
  bull markets, even slightly decelerating momentum stocks might be worth keeping.

- **How do we backtest the full pipeline?** Current evaluation is model IC on held-out test set.
  What we really want is: "if I bought the top-10 MLScore stocks each week, what were my returns?"
  A proper backtest with realistic transaction costs and holding periods has not been implemented.

- **Should MLScoreDelta be model-specific?** Currently it tracks the consensus `MLScore`. It could
  instead track per-model deltas, which would let you see "breakout model is falling even though
  quality model is stable" — an early deterioration signal.
