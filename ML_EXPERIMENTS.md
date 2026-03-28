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
  ├── Market cap filter: ≥ $1B
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

Features are organized into four tiers (75 total):

**Technical (per-stock, rank-normalized cross-sectionally):**
Sharpe, Sortino, MaxDrawdown, RSI, Momentum12_1, `log_momentum_12_1` (log-compressed to handle 2000%+ values),
MomentumAccel, VolumeTrend, VolumeConsistency, BollingerPctB, MACD, StochK/D, ATR14,
PriceToSMA{20/50/200}, PriceVs52wkHigh, `info_discreteness`, `rs_rating`, and derived
cross-sectional features (sector_return_rank, sector_momentum_rank, return_pct_x_regime)

**Market context (constant per date, NOT rank-normalized — regime features):**
SPY: return_10d, return_21d, vol_20d, RSI_14, above_SMA200, momentum_regime
VIX: vix_level, `vix_percentile`, vix_1m_change, vix_term_structure
WML (Ken French momentum factor): wml_realized_vol_126d, wml_trailing_1m, wml_trailing_3m

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

### Phase 1: Early Exploration (lgbm_v1 through lgbm_v5)

These were the earliest experiments before a formal naming scheme was established.
No backtest data was retained; performance numbers below are from the leaderboard.

| Model | IC | Top Decile Sharpe | Notes |
|-------|----|----|-------|
| lgbm_v1 | 0.051 | — | Initial baseline, 58 features, L/S MaxDD −96% |
| lgbm_v4 | 0.027 | — | Variant of v1, poor signal |
| lgbm_momentum | 0.045 | 1.19 | Early composite target attempt, L/S Sharpe −0.68 |
| lgbm_v5_21d | 0.042 | 1.06 | 21-day horizon, 66 features |
| lgbm_v5_63d | 0.118 | 1.34 | 63-day horizon, 66 features — meaningful jump in IC vs v1/v4 |
| lgbm_v5_sector | — | — | Sector-relative variant; no leaderboard data retained |

**Feature ablations (exp3a/b/c, exp4a/b/d):** Sub-experiments conducted around the v5 era
to isolate feature group contributions. Run as leave-one-group-out ablations:
- `exp3a_no_value`: removed value features (earnings_yield, book_to_market) — minimal IC impact
- `exp3b_momentum_only`: technical momentum features only — strong IC, poor consistency
- `exp3c_value_only`: value features only — weak IC, confirmed fundamentals alone insufficient
- `exp4a_market_excess`: forward_63d_market_excess_rank target test
- `exp4b_sector_relative`: forward_63d_sector_relative_rank target test
- `exp4d_21d`: 21-day horizon variant

These ablations confirmed: (1) momentum features drive most of the signal, (2) value features
add modest marginal IC on composite targets, (3) the 63-day horizon is better than 21-day.

---

### Phase 2: VIX/WML + Composite Target (lgbm_composite_63d through lgbm_composite_v4)

The key insight from Phase 1 was that adding market-regime features (VIX, WML) should
dramatically improve drawdown control. This phase explores that hypothesis.

---

### Exp 1: lgbm_composite_63d *(retired)*
- **Target:** `forward_63d_composite_rank`
- **Features:** 66 (no VIX/WML, no info_discreteness, no log_momentum)
- **Training:** 2019–2024 OHLCV, all stocks
- **IC:** 0.157 | ICIR: 1.693 | L/S Sharpe: 1.63 | L/S MaxDD: −4.8% | Top Decile Sharpe: 1.53
- **Result:** Solid baseline. First model to show meaningful IC. Superseded by v2 which added VIX/WML.

---

### Exp 2: lgbm_sector_relative_63d *(retired)*
- **Target:** `forward_63d_sector_relative_rank`
- **Features:** 66 (no VIX/WML)
- **IC:** 0.122 | ICIR: 1.480 | L/S Sharpe: 0.667 | L/S MaxDD: −25.5% | Top Decile Sharpe: 1.68
- **Result:** Good long-only top-decile Sharpe but L/S drawdowns without regime conditioning.
  Sector-relative target produces cleaner consistency but misses market-wide momentum.

---

### Exp 3a: lgbm_composite_v2 *(active candidate)*
- **Target:** `forward_63d_composite_rank`
- **Features:** 73 (added VIX/WML + info_discreteness; VIX/WML were zeros in training — training-serving skew)
- **IC:** 0.159 | ICIR: 1.606 | L/S Sharpe: 1.54 | **L/S MaxDD: −0.4%** | Top Decile Sharpe: 1.71
- **SHAP #1/2:** wml_realized_vol_126d, wml_trailing_1m (even though zeros in training — inference-time values still helped)
- **Result:** Prior champion. Regime-awareness produced near-zero drawdown. The VIX/WML features
  were computed at inference time with today's values but were zeros during training — a
  training-serving skew that was later fixed in composite_v4. Despite this, live inference
  values gave meaningful regime signal.

---

### Exp 3b: lgbm_composite_v3 *(retired)*
- **Target:** `forward_63d_composite_rank`
- **Features:** 73 (added raw fundamentals: PE, PB, short_pct_float)
- **IC:** 0.140 | L/S Sharpe: 0.799 | L/S MaxDD: −32.4% | Top Decile Sharpe: 1.65
- **Result:** Failed. Raw fundamentals created a value tilt. `short_pct_float` caused squeeze
  exposure in the Jan 2025 momentum rally. Raw fundamentals are dangerous in L/S context.
  Key lesson: use sector-relative fundamentals, never raw P/E or P/B.

---

### Exp 3c: lgbm_sector_relative_v2 *(retired)*
- **Target:** `forward_63d_sector_relative_rank`
- **Features:** 73 (with fundamentals)
- **IC:** 0.121 | L/S Sharpe: −0.018 | L/S MaxDD: −36.4% | **Top Decile Sharpe: 1.75 | Consistency: 82.7%**
- **Result:** L/S completely failed (fundamentals dominated, effectively value-short in growth market).
  But top-decile long-only metrics were the best of any model tested. Useful insight: the sector-relative
  target does a better job of separating intra-sector winners, but fundamentals break L/S.

---

### Exp 4: lgbm_breakout_63d *(retired)*
- **Target:** `forward_63d_market_excess_rank`
- **Filter:** `--momentum-universe` (momentum_12_1 > 10%, < 400%, price > SMA200, return_pct > 20%)
- **IC:** 0.068 | ICIR: 1.021 | L/S Sharpe: 0.62 | L/S MaxDD: −51.2% | Top Decile Sharpe: 1.43
- **Result:** The 400% cap in `--momentum-universe` silently excluded AXTI (2234%), SNDK (1139%),
  ERAS (817%) from training. In live scoring, these stocks scored 0.2–3.4/100.
  Market-excess target also produced lower IC than composite targets.

---

### Exp 5: lgbm_breakout_v2 *(retired)*
- **Target:** `forward_63d_rank` (raw return rank — no quality penalty)
- **Filter:** None (full universe)
- **Features:** 73 (added `log_momentum_12_1` to handle extreme momentum values)
- **IC:** 0.118 | ICIR: 1.112 | L/S Sharpe: 2.263 | **L/S MaxDD: −1.9%** | Top Decile Sharpe: 1.70, Consistency: 83.0%
- **Result:** Improved target stock scores dramatically (SNDK 0.2→44, AXTI 0.3→48, ERAS 3.4→49).
  Good L/S profile. However, ZIM/GSL (shipping stocks) still scored 99–100 despite being wrong-objective;
  the momentum gate (MomentumAccel > 0) was added to score.py to handle this.

---

### Exp 6: lgbm_composite_v4 *(CHAMPION — as of 2026-03-27)*
- **Target:** `forward_63d_composite_rank`
- **Features:** 73 — same as v2/v3 but `short_pct_float` removed
- **Training fix:** VIX/WML historical data now populated in dataset.py (381,622 rows with real values;
  fixes the training-serving skew present in v2/v3). Ken French WML data fetched from Dartmouth;
  VIX/VIX3M from Yahoo Finance; cached to `data/raw_ohlcv/factors/`.
- **Momentum-universe cap:** Raised from 4x (400%) to 50x (5000%) — covers AXTI (2234%)
- **IC:** 0.170 | ICIR: 1.531 | L/S Sharpe: 0.595 | Top Decile Sharpe: 1.43
- **Live scores (2026-03-27):** SNDK 97.8 (#14), AXTI 93.1 (#42), ERAS 92.4 (#46), FSLY 86.6 (#80)
- **Result:** Promoted to champion. Highest IC of all models (0.170 vs. v2's 0.159). Scores the
  target explosive stocks in the top decile. Lower L/S Sharpe vs. v2 (0.60 vs. 1.54) is acceptable
  because L/S is secondary to our objective. `short_pct_float` removal reduced squeeze risk.

---

## Final Model Leaderboard

| Model | IC | Top Decile Sharpe | Consistency | L/S Sharpe | L/S MaxDD | Status |
|-------|:--:|:-----------------:|:-----------:|:----------:|:---------:|--------|
| lgbm_composite_v4 | **0.170** | 1.43 | 73.8% | 0.60 | −52.6% | **Champion** |
| lgbm_composite_v2 | 0.159 | 1.71 | 81.2% | 1.54 | **−0.4%** | Candidate |
| lgbm_breakout_v2 | 0.118 | 1.70 | **83.0%** | **2.26** | −1.9% | Retired |
| lgbm_sector_relative_v2 | 0.121 | **1.75** | **82.7%** | −0.02 | −36.4% | Retired |
| lgbm_composite_63d | 0.157 | 1.53 | 77.5% | 1.63 | −4.8% | Retired |
| lgbm_composite_v3 | 0.140 | 1.65 | 81.5% | 0.80 | −32.4% | Retired |
| lgbm_sector_relative_63d | 0.122 | 1.68 | 81.5% | 0.67 | −25.5% | Retired |
| lgbm_breakout_63d | 0.068 | 1.43 | 75.3% | 0.62 | −51.2% | Retired |

**Champion rationale:** composite_v4 has the highest IC (best raw predictive power), scores target
momentum stocks (SNDK/AXTI/ERAS) in the top decile, and has the cleanest feature set (no
short_pct_float, real VIX/WML in training). The lower L/S Sharpe vs. breakout_v2 is outweighed
by dramatically better momentum stock identification, which is our primary objective.

**Consensus score (2026-03-27):** Composite_v4 (champion) + composite_v2 (candidate) combined.
SNDK 93.2, AXTI 76.9, ERAS 71.2, FSLY 69.0.

---

## Current Scoring Pipeline

**Active models:** lgbm_composite_v4 (champion) + lgbm_composite_v2 (candidate)

**Momentum gate (score.py):**
- `price > SMA200` — filters downtrends and value traps
- `MomentumAccel > 0` — filters fading/peaking stocks (ZIM −0.24, GSL −0.11 both correctly removed)

**Market cap filter:** ≥ $1B — removes illiquid micro/small caps

**Sector diversity cap:**
Technology: 15, Healthcare: 10, Financial Services: 10, Consumer Cyclical: 7,
Communication Services: 7, Industrials: 3, Energy: 3, Utilities: 2, default: 5

**MLScoreDelta:** Cached comparison vs. prior run (`data/factors/mlscore_cache.parquet`).
Falling score on a held position = warning signal.

---

## Key Lessons Learned

1. **Composite target is right for the champion** — but it penalizes extreme-volatility stocks.
   The composite target rewards consistent outperformance (Sharpe, drawdown protection), which
   means explosive names only score well if their quality is also strong. For composite_v4,
   the ROE + revenue_growth features capture the "real business vs. pure technicals" distinction.

2. **Raw fundamentals create value tilt** — raw P/E, P/B break L/S. Always use sector-relative
   fundamentals (earnings_yield_vs_sector). `short_pct_float` is dangerous — squeeze risk.

3. **VIX/WML features must be in training data** — originally computed only at inference time
   (zeros during training). Added historical pipeline to dataset.py in composite_v4. Models
   can now actually learn regime-conditional patterns.

4. **log_momentum_12_1 is essential** — standard winsorization clips AXTI (2234%) flat to the
   same value as a 100% momentum stock. Log compression preserves the ordering.

5. **MomentumAccel > 0 gate is cleaner than sector caps** for removing peaked/fading stocks.
   Both ZIM and GSL had negative MomentumAccel despite textbook-clean technical profiles.

6. **The 63-day horizon is better than 21-day** — IC drops from 0.118→0.042 at 21d.
   Signal-to-noise is too low at shorter horizons for this feature set.

7. **MLScoreDelta is the right exit signal** — falling score over consecutive runs signals
   momentum deterioration before the technicals turn.

8. **IC alone is insufficient for model selection** — must verify target stock scores (SNDK,
   AXTI, ERAS, FSLY) in live scoring. A model with IC 0.200 can still score the target names
   at 3/100 if the training distribution doesn't include extreme momentum.

---

## Next Steps — Set Transformer

With lgbm_composite_v4 in production, the clear next phase is the **Set Transformer**.
LightGBM sees only each stock's own features; it cannot learn "this stock is anomalous
relative to its sector peers." AXTI at 2234% momentum is extraordinary within Technology —
a transformer attending over the 50 Technology stocks would encode that it's a 4-sigma outlier.

**Architecture:**
```
Input: sector cohort of N stocks × F features per snapshot date
→ Linear projection → (no positional encoding — set, not sequence)
→ Multi-head self-attention (each stock attends to all peers)
→ Per-stock output head → forward_63d_composite_rank prediction
```

Use the MASTER architecture pattern: separate intra-stock (temporal) and inter-stock (cross-sectional)
attention modules. Do NOT use naive full attention over all stocks × all dates.

**Trigger condition:** Begin when LightGBM IC plateaus below 0.25 after adding more training data
or feature engineering. Current ceiling appears to be ~0.17–0.20.

**Prerequisites (all met):**
- Clean production baseline: composite_v4 champion ✓
- Stable training target: `forward_63d_composite_rank` confirmed ✓
- VIX/WML historical data in dataset.py ✓
- log_momentum_12_1 handles extreme values ✓

---

## Open Questions

- **Dynamic momentum gate?** Current gate is static (MomentumAccel > 0). In high-VIX regimes,
  a tighter gate might filter more noise. In low-VIX bull markets, even slightly decelerating
  stocks might be worth keeping. Could be implemented as a VIX-conditional threshold.

- **Separate Breakout / Quality scores in the dashboard?** Two score columns
  (`MLScore_Breakout`, `MLScore_Quality`) would let the user distinguish "speculative breakout"
  from "quality momentum." A conviction pick needs both to be high.

- **MLScoreDelta per-model?** Currently tracks the consensus score. Per-model deltas would
  surface "breakout model falling even though quality model stable" — an earlier deterioration signal.

- **Pipeline backtest?** Current evaluation is IC on held-out test set. A proper backtest
  ("if I bought the top-10 MLScore stocks each Monday, what were my returns?") with realistic
  transaction costs has not been implemented.
