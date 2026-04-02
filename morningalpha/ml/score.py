"""alpha ml score — score all period CSVs with registered ML models.

Usage:
    alpha ml score
    alpha ml score --data-dir data/latest --models-dir models
"""
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rich_click as click
from rich.console import Console
from rich.table import Table
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_DATA_DIR = "data/latest"
_WEIGHTS_PATH   = Path("data/factors/model_weights.json")
_LEDGER_PATH    = Path("data/factors/predictions_ledger.parquet")
# How many trading days before we evaluate a prediction against realized returns
_EVAL_HORIZON   = 63
DEFAULT_MODELS_DIR = "models"
# 3M is the canonical scoring file — matches the ~3-month lookback used in training.
# Scores are computed once here, then merged into all period CSVs by ticker.
SCORE_SOURCE = "stocks_3m.csv"
PERIOD_FILES = ["stocks_2w.csv", "stocks_1m.csv", "stocks_3m.csv", "stocks_6m.csv"]


# ---------------------------------------------------------------------------
# Dynamic model weighting (Hedge algorithm on rolling IC)
# ---------------------------------------------------------------------------

def _load_model_weights(active_model_ids: list) -> dict:
    """Return normalized weights for each active model.

    Reads data/factors/model_weights.json. Falls back to equal weights if the
    file is missing or a model isn't listed yet.
    """
    equal = {m: 1.0 / len(active_model_ids) for m in active_model_ids}
    if not _WEIGHTS_PATH.exists():
        return equal
    try:
        with open(_WEIGHTS_PATH) as f:
            data = json.load(f)
        stored = data.get("weights", {})
        weights = {m: float(stored.get(m, 1.0)) for m in active_model_ids}
        total = sum(weights.values())
        return {m: w / total for m, w in weights.items()} if total > 0 else equal
    except Exception as exc:
        logger.warning("Could not load model weights (%s) — using equal weights", exc)
        return equal


def _save_model_weights(weights: dict, ic_entry: dict | None = None) -> None:
    """Persist updated weights and append an IC history entry."""
    try:
        _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if _WEIGHTS_PATH.exists():
            with open(_WEIGHTS_PATH) as f:
                existing = json.load(f)

        history: list = existing.get("ic_history", [])
        if ic_entry:
            history.append(ic_entry)
            window = existing.get("window_weeks", 12)
            history = history[-window:]  # keep rolling window only

        payload = {
            "updated":      str(date.today()),
            "method":       existing.get("method", "rolling_ic_hedge"),
            "window_weeks": existing.get("window_weeks", 12),
            "eta":          existing.get("eta", 0.5),
            "weights":      weights,
            "ic_history":   history,
        }
        with open(_WEIGHTS_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save model weights (%s)", exc)


def _hedge_update(current_weights: dict, ic_by_model: dict, eta: float = 0.5) -> dict:
    """Apply one step of the Hedge update rule.

    w_i(t+1) ∝ w_i(t) * exp(η * IC_i)

    IC acts as the reward signal: higher IC = model gets more weight next period.
    Negative IC models are penalized but never zeroed out entirely.
    """
    new_weights = {
        m: current_weights.get(m, 1.0) * np.exp(eta * ic_by_model.get(m, 0.0))
        for m in current_weights
    }
    total = sum(new_weights.values())
    return {m: w / total for m, w in new_weights.items()}


# ---------------------------------------------------------------------------
# Prediction ledger — append-only record for future IC evaluation
# ---------------------------------------------------------------------------

def _append_predictions_ledger(df_score: pd.DataFrame, raw_scores: dict) -> None:
    """Append this week's raw model scores to the predictions ledger.

    Each row = one ticker scored today. 63 trading days later (~13 weeks),
    _evaluate_and_update_weights() joins these rows against realized returns
    to compute live IC and update the ensemble weights.
    """
    try:
        _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        today = pd.Timestamp(date.today())
        eval_date = today + timedelta(days=_EVAL_HORIZON * 7 // 5 + 14)  # ≈ 63 trading days

        rows = pd.DataFrame({"ticker": df_score["Ticker"].values if "Ticker" in df_score.columns else df_score.index})
        rows["scored_date"] = today
        rows["eval_after"]  = eval_date   # approximate calendar date when outcome is knowable

        price_col = next((c for c in df_score.columns if c.lower().startswith("price") or c == "Close"), None)
        rows["price_at_score"] = df_score[price_col].values if price_col else np.nan

        for model_id, raw in raw_scores.items():
            rows[f"raw_{model_id}"] = raw

        existing = pd.read_parquet(_LEDGER_PATH) if _LEDGER_PATH.exists() else pd.DataFrame()
        updated = pd.concat([existing, rows], ignore_index=True)
        updated.to_parquet(_LEDGER_PATH, index=False)
        console.print(f"[dim]Ledger updated: {len(rows)} predictions appended → {_LEDGER_PATH}[/dim]")
    except Exception as exc:
        logger.warning("Ledger append failed (%s) — skipping", exc)


def _evaluate_and_update_weights(df_current: pd.DataFrame, raw_scores: dict) -> None:
    """Evaluate predictions from ~63 trading days ago against realized returns.

    For each cohort of predictions whose eval_after date has passed, compute
    the realized return using the current spread data, then apply a Hedge
    update to the ensemble weights.
    """
    if not _LEDGER_PATH.exists():
        return

    try:
        ledger = pd.read_parquet(_LEDGER_PATH)
        today  = pd.Timestamp(date.today())
        mature = ledger[ledger["eval_after"] <= today].copy()
        if mature.empty:
            console.print("[dim]No mature predictions to evaluate yet (need ~13 weeks of history)[/dim]")
            return

        # Join with current prices to compute realized return
        price_col = next((c for c in df_current.columns if c.lower().startswith("price") or c == "Close"), None)
        if price_col is None or "Ticker" not in df_current.columns:
            return

        current_prices = df_current.set_index("Ticker")[price_col].rename("current_price")
        mature = mature.join(current_prices, on="ticker", how="inner")
        mature = mature.dropna(subset=["price_at_score", "current_price"])
        mature = mature[mature["price_at_score"] > 0]
        mature["realized_return"] = mature["current_price"] / mature["price_at_score"] - 1

        if len(mature) < 50:
            console.print(f"[dim]Only {len(mature)} mature predictions — skipping weight update[/dim]")
            return

        # Compute IC per model across all mature cohorts
        model_cols = [c for c in mature.columns if c.startswith("raw_")]
        ic_by_model: dict = {}
        for col in model_cols:
            model_id = col[len("raw_"):]
            valid = mature[["realized_return", col]].dropna()
            if len(valid) < 20:
                continue
            ic = float(spearmanr(valid[col], valid["realized_return"]).correlation)
            ic_by_model[model_id] = ic

        if not ic_by_model:
            return

        # Load and update weights
        with open(_WEIGHTS_PATH) as f:
            data = json.load(f)
        current_weights = data.get("weights", {m: 1.0 for m in ic_by_model})
        eta = data.get("eta", 0.5)

        new_weights = _hedge_update(current_weights, ic_by_model, eta=eta)
        ic_entry = {"week_end": str(today.date()), **ic_by_model}
        _save_model_weights(new_weights, ic_entry=ic_entry)

        # Remove evaluated rows from ledger
        remaining = ledger[ledger["eval_after"] > today]
        remaining.to_parquet(_LEDGER_PATH, index=False)

        ic_str = "  ".join(f"{m}: {ic:+.4f}" for m, ic in sorted(ic_by_model.items()))
        w_str  = "  ".join(f"{m}: {w:.3f}" for m, w in sorted(new_weights.items()))
        console.print(f"[bold cyan]Weight update:[/bold cyan]  IC → {ic_str}")
        console.print(f"[bold cyan]New weights  :[/bold cyan]  {w_str}  (n={len(mature)} stocks)")

    except Exception as exc:
        logger.warning("Weight evaluation failed (%s) — weights unchanged", exc)


def _load_config(models_dir: Path) -> dict:
    config_path = models_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    # Auto-discover: treat all .pkl files as models, no champion
    pkls = sorted(models_dir.glob("*.pkl"))
    return {
        "champion": pkls[0].stem if pkls else None,
        "models": [{"id": p.stem, "type": "lgbm", "status": "champion" if i == 0 else "challenger"}
                   for i, p in enumerate(pkls)],
    }


def _write_ticker_index(
    scored_df: pd.DataFrame,
    active_models: list,
    raw_scores: dict,
    output_dir: Path,
) -> None:
    """Write a lightweight JSON for frontend stock search / Forecast page autocomplete.

    Includes MLScore (consensus) and per-model scores for each scored ticker.
    Stored at: {output_dir}/ticker_index.json
    """
    # Determine which model IDs map to which friendly field names in the JSON
    MODEL_FIELD_MAP = {
        "lgbm_breakout_v5":      "mlScore_breakout",
        "lgbm_composite_v6":     "mlScore_composite",
        "st_sector_relative_v1": "mlScore_st",
    }

    def _f(val, default=None):
        """Convert a value to a rounded float, returning default for NaN/None/inf."""
        if val is None:
            return default
        try:
            f = float(val)
            return default if (f != f or f == float("inf") or f == float("-inf")) else round(f, 1)
        except (TypeError, ValueError):
            return default

    def _investment_score(row) -> float | None:
        """Mirror of scoring.ts calculateInvestmentScore — Return/Quality/Sharpe/Consistency/Drawdown."""
        return_pct = _f(row.get("Return_3M_%"))
        quality    = _f(row.get("QualityScore"))
        sharpe     = _f(row.get("SharpeRatio"))
        consistency = _f(row.get("ConsistencyScore"))
        drawdown   = _f(row.get("MaxDrawdown"))

        score = 0.0
        factors = 0.0
        if return_pct is not None:
            score += min(30.0, (return_pct / 200.0) * 30.0); factors += 30.0
        if quality is not None:
            score += (quality / 100.0) * 25.0; factors += 25.0
        if sharpe is not None:
            score += max(0.0, min(20.0, ((sharpe + 1.0) / 4.0) * 20.0)); factors += 20.0
        if consistency is not None:
            score += (consistency / 100.0) * 15.0; factors += 15.0
        if drawdown is not None:
            score += max(0.0, min(10.0, ((drawdown + 50.0) / 50.0) * 10.0)); factors += 10.0
        if factors == 0.0:
            return None
        return round((score / factors) * 1000.0) / 10.0

    records = []
    for _, row in scored_df.iterrows():
        sector = row.get("Sector", row.get("sector", None))
        entry: dict = {
            "ticker":  row.get("Ticker", ""),
            "name":    str(row.get("Name", row.get("name", ""))),
            "sector":  None if (sector is None or (isinstance(sector, float) and sector != sector)) else sector,
            "mlScore": _f(row.get("MLScore"), default=0.0),
            "investmentScore": _investment_score(row),
        }
        for m in active_models:
            if m["id"] not in raw_scores:
                continue
            col = f"MLScore_{m['id']}"
            field = MODEL_FIELD_MAP.get(m["id"], f"mlScore_{m['id']}")
            entry[field] = _f(row.get(col))

        records.append(entry)

    out_path = output_dir / "ticker_index.json"
    with open(out_path, "w") as f:
        json.dump(records, f)
    console.print(f"[dim]✓ Wrote ticker_index.json ({len(records)} tickers)[/dim]")

    # Mirror to the web public directory so the dev server and GitHub Pages build
    # pick up the latest scores without a manual copy step.
    web_public = Path(__file__).parents[2] / "morningalpha" / "web" / "public" / "data" / "latest"
    if web_public.exists():
        import shutil
        shutil.copy2(out_path, web_public / "ticker_index.json")
        console.print(f"[dim]✓ Mirrored ticker_index.json → {web_public}[/dim]")


@click.command("score")
@click.option("--data-dir", default=DEFAULT_DATA_DIR, show_default=True,
              help="Directory containing period CSVs from `alpha spread`.")
@click.option("--models-dir", default=DEFAULT_MODELS_DIR, show_default=True,
              help="Directory containing model .pkl files and config.json.")
def score(data_dir, models_dir):
    """Score all stocks in data-dir CSVs with registered ML models.

    Adds MLScore (champion/consensus 0-100) and MLScore_{model_id} per-model
    columns to each period CSV. Run after `alpha spread`.

    Examples:
        alpha ml score
        alpha ml score --data-dir data/latest --models-dir models
    """
    from morningalpha.ml.inference import get_raw_scores

    data_path = Path(data_dir)
    models_path = Path(models_dir)

    if not models_path.exists():
        console.print(f"[red]Models directory not found: {models_path}[/red]")
        raise SystemExit(1)

    config = _load_config(models_path)
    champion_id = config.get("champion")
    model_entries = config.get("models", [])

    # Only score models that are active (champion/candidate) and have a checkpoint file.
    # LightGBM models use .pkl; Set Transformer models use .pt (path from config or default).
    def _checkpoint_exists(m: dict) -> bool:
        if "checkpoint" in m:
            return Path(m["checkpoint"]).exists()
        if m.get("type") == "set_transformer":
            return (models_path / f"{m['id']}.pt").exists()
        return (models_path / f"{m['id']}.pkl").exists()

    active_models = [
        m for m in model_entries
        if m.get("status", "candidate") not in ("retired",)
        and _checkpoint_exists(m)
    ]

    if not active_models:
        console.print("[yellow]No model checkpoint files found — nothing to score.[/yellow]")
        return

    console.print(f"\n[bold]ML Scoring[/bold] — {len(active_models)} model(s): "
                  + ", ".join(
                      f"[cyan]{m['id']}[/cyan]" + (" [green](champion)[/green]" if m['id'] == champion_id else "")
                      for m in active_models
                  ))

    # -----------------------------------------------------------------------
    # Step 1: Score the 3M CSV — canonical source matching training distribution
    # -----------------------------------------------------------------------
    source_path = data_path / SCORE_SOURCE
    if not source_path.exists():
        console.print(f"[red]Canonical score source not found: {source_path}[/red]")
        raise SystemExit(1)

    df3m = pd.read_csv(source_path, index_col=0)

    # Backfill MarketCap from fundamentals.csv if the spread CSV has it empty
    # (happens when spread CSVs predate the fundamentals-merge fix in access.py).
    fund_path = data_path / "fundamentals.csv"
    mc_col = df3m.get("MarketCap", pd.Series(dtype=float))
    if fund_path.exists() and (mc_col.isna().all() or "MarketCap" not in df3m.columns):
        fund_df = pd.read_csv(fund_path, usecols=["Ticker", "MarketCap"])
        df3m = df3m.drop(columns=["MarketCap"], errors="ignore")
        df3m = df3m.merge(fund_df, on="Ticker", how="left")
        console.print("[dim]MarketCap backfilled from fundamentals.csv[/dim]")

    # Filter to large-enough stocks before scoring — micro/small caps produce
    # noisy signals (illiquid, mean-reversion artifacts) and are untradeable at scale.
    MIN_MARKET_CAP = 100_000_000  # $100M — matches training dataset floor
    mc_numeric = df3m["MarketCap"].apply(pd.to_numeric, errors="coerce") if "MarketCap" in df3m.columns else pd.Series(dtype=float)
    if mc_numeric.notna().any():
        eligible = mc_numeric >= MIN_MARKET_CAP
        n_filtered = int((~eligible & mc_numeric.notna()).sum())
        df_score = df3m[eligible | mc_numeric.isna()].copy()
        if n_filtered:
            console.print(f"[dim]Filtered out {n_filtered} stocks below ${MIN_MARKET_CAP/1e9:.0f}B market cap[/dim]")
    else:
        df_score = df3m.copy()

    # No pre-scoring return or quality filters — the model was trained on the full universe
    # and learned to rank good from bad. Pre-filtering by return defeats the purpose.

    # Trend gate — only score stocks above their 200-day MA.
    # Filters confirmed downtrends and value traps for a long-only portfolio.
    # MomentumAccel is intentionally NOT gated here — the model learned from it during
    # training and already prices it in; a hard gate would cut early-stage breakouts.
    gate_mask = pd.Series(True, index=df_score.index)
    if "PriceToSMA200Pct" in df_score.columns:
        sma200_pct = pd.to_numeric(df_score["PriceToSMA200Pct"], errors="coerce")
        gate_mask &= (sma200_pct > 0) | sma200_pct.isna()
    n_gated = (~gate_mask).sum()
    df_score = df_score[gate_mask].copy()
    if n_gated:
        console.print(f"[dim]Trend gate (price > SMA200): removed {n_gated} downtrending stocks[/dim]")

    raw_scores: dict[str, np.ndarray] = {}

    for m in active_models:
        model_type = m.get("type", "lgbm")
        # Resolve checkpoint path: explicit in config, or default by type
        if "checkpoint" in m:
            model_path = Path(m["checkpoint"])
        elif model_type == "set_transformer":
            model_path = models_path / f"{m['id']}.pt"
        else:
            model_path = models_path / f"{m['id']}.pkl"

        if not model_path.exists():
            console.print(f"[yellow]Checkpoint not found for {m['id']}: {model_path} — skipping[/yellow]")
            continue

        try:
            if model_type == "set_transformer":
                from morningalpha.ml.inference import get_st_raw_scores
                raw = get_st_raw_scores(df_score, model_path)
            else:
                raw = get_raw_scores(df_score, model_path)
            raw_scores[m["id"]] = raw
            pct = pd.Series(raw, index=df_score.index).rank(pct=True).mul(100).round(1)
            df_score[f"MLScore_{m['id']}"] = pct.values
        except Exception as exc:
            logger.warning("Scoring with %s failed: %s", m["id"], exc)

    if not raw_scores:
        console.print("[yellow]All models failed to score — nothing written.[/yellow]")
        return

    # Evaluate any mature predictions (63 trading days old) and update weights
    _evaluate_and_update_weights(df_score, raw_scores)

    # Append this week's predictions to the ledger for future IC evaluation
    _append_predictions_ledger(df_score, raw_scores)

    # Consensus: dynamic IC-weighted blend via Hedge algorithm.
    # Weights are loaded from data/factors/model_weights.json and updated
    # each time mature predictions (≥63 trading days old) are evaluated.
    model_weights = _load_model_weights(list(raw_scores.keys()))
    if len(raw_scores) > 1:
        weights = np.array([model_weights[mid] for mid in raw_scores])
        stacked = np.column_stack(list(raw_scores.values()))
        consensus = (stacked * weights).sum(axis=1)
        df_score["MLScore"] = pd.Series(consensus, index=df_score.index).rank(pct=True).mul(100).round(1).values
        w_str = "  ".join(f"{m}: {model_weights[m]:.3f}" for m in raw_scores)
        console.print(f"[dim]Ensemble weights: {w_str}[/dim]")
    else:
        df_score["MLScore"] = df_score[f"MLScore_{list(raw_scores.keys())[0]}"]

    # Sector diversity cap: per-sector limits on final MLScore ranking.
    # High-opportunity sectors (Technology, Healthcare) get more slots.
    # Individual MLScore_* columns are preserved unchanged for analysis.
    DEFAULT_SECTOR_CAP = 5
    SECTOR_CAPS: dict[str, int] = {
        "Technology": 15,
        "Healthcare": 10,
        "Financial Services": 10,
        "Consumer Cyclical": 7,
        "Communication Services": 7,
        "Industrials": 3,   # shipping/transport stocks often have clean technicals but slow growth
        "Energy": 3,
        "Utilities": 2,
    }
    if "Sector" in df_score.columns:
        sectors = df_score["Sector"].fillna("Unknown")
        sector_rank = df_score.groupby(sectors)["MLScore"].rank(ascending=False, method="first")
        cap_per_row = sectors.map(lambda s: SECTOR_CAPS.get(s, DEFAULT_SECTOR_CAP))
        within_cap = sector_rank <= cap_per_row
        diversity_bonus = within_cap.astype(float) * 1000
        df_score["MLScore"] = (
            (df_score["MLScore"] + diversity_bonus)
            .rank(pct=True).mul(100).round(1)
        )
        n_deprioritized = int((~within_cap).sum())
        cap_summary = ", ".join(f"{s}:{c}" for s, c in sorted(SECTOR_CAPS.items()))
        console.print(
            f"[dim]Sector diversity cap (default {DEFAULT_SECTOR_CAP}, overrides: {cap_summary}): "
            f"{n_deprioritized} stocks deprioritized[/dim]"
        )

    # Score delta — compare to previous run's cached scores (falling = warning signal)
    _score_cache_path = Path("data/factors/mlscore_cache.parquet")
    try:
        if _score_cache_path.exists():
            prev = pd.read_parquet(_score_cache_path).set_index("Ticker")["MLScore"]
            current = df_score.set_index("Ticker")["MLScore"]
            delta = current.sub(prev, fill_value=float("nan")).reindex(current.index).round(1)
            df_score["MLScoreDelta"] = delta.values
        _score_cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_score[["Ticker", "MLScore"]].to_parquet(_score_cache_path, index=False)
    except Exception as exc:
        logger.warning("MLScoreDelta computation failed (%s) — delta will be absent", exc)

    # -----------------------------------------------------------------------
    # Step 2: Build ticker → score lookup from the eligible scored stocks
    # -----------------------------------------------------------------------
    delta_col = ["MLScoreDelta"] if "MLScoreDelta" in df_score.columns else []
    score_cols = ["MLScore"] + delta_col + [f"MLScore_{m['id']}" for m in active_models if m["id"] in raw_scores]
    scores_by_ticker = df_score.set_index("Ticker")[score_cols]

    top_ticker = scores_by_ticker["MLScore"].idxmax()

    # -----------------------------------------------------------------------
    # Step 3: Merge scores into all period CSVs by Ticker
    # -----------------------------------------------------------------------
    for filename in PERIOD_FILES:
        csv_path = data_path / filename
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, index_col=0)

        # Drop any stale MLScore columns before merging
        stale = [c for c in df.columns if c.startswith("MLScore")]
        if stale:
            df = df.drop(columns=stale)

        df = df.merge(scores_by_ticker, on="Ticker", how="left")
        df.to_csv(csv_path, index_label="Rank")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    table = Table(title="ML Scoring Summary", show_header=True, header_style="bold cyan")
    table.add_column("Source", style="dim")
    table.add_column("Stocks scored", justify="right")
    table.add_column("Models", justify="right")
    table.add_column("Top ML Pick", style="bold green")
    table.add_column("Merged into", style="dim")
    table.add_row(
        SCORE_SOURCE,
        f"{len(df_score)} / {len(df3m)}",
        str(len(raw_scores)),
        str(top_ticker),
        ", ".join(f for f in PERIOD_FILES if (data_path / f).exists()),
    )
    console.print(table)

    # Write generation timestamp for the dashboard
    import datetime
    generated_path = data_path / "_generated.json"
    with open(generated_path, "w") as f:
        json.dump({"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}, f)

    # Write ticker_index.json for the Forecast/Portfolio frontend pages
    _write_ticker_index(df_score, active_models, raw_scores, data_path)

    console.print("\n[bold green]✓ ML scoring complete[/bold green]")
