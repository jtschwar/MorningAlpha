"""alpha ml score — score all period CSVs with registered ML models.

Usage:
    alpha ml score
    alpha ml score --data-dir data/latest --models-dir models
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rich_click as click
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_DATA_DIR = "data/latest"
DEFAULT_MODELS_DIR = "models"
# 3M is the canonical scoring file — matches the ~3-month lookback used in training.
# Scores are computed once here, then merged into all period CSVs by ticker.
SCORE_SOURCE = "stocks_3m.csv"
PERIOD_FILES = ["stocks_2w.csv", "stocks_1m.csv", "stocks_3m.csv", "stocks_6m.csv"]


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

    # Only score models that are active (champion/candidate) and have a .pkl file
    active_models = [
        m for m in model_entries
        if m.get("status", "candidate") not in ("retired",)
        and (models_path / f"{m['id']}.pkl").exists()
    ]

    if not active_models:
        console.print("[yellow]No model .pkl files found — nothing to score.[/yellow]")
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
        model_path = models_path / f"{m['id']}.pkl"
        try:
            raw = get_raw_scores(df_score, model_path)
            raw_scores[m["id"]] = raw
            pct = pd.Series(raw, index=df_score.index).rank(pct=True).mul(100).round(1)
            df_score[f"MLScore_{m['id']}"] = pct.values
        except Exception as exc:
            logger.warning("Scoring with %s failed: %s", m["id"], exc)

    if not raw_scores:
        console.print("[yellow]All models failed to score — nothing written.[/yellow]")
        return

    # Consensus: weighted average raw scores → re-ranked percentile.
    # composite models weight higher than breakout — composite is better calibrated for
    # extended breakout stocks (AXTI, SNDK) where breakout underscores due to extension.
    MODEL_WEIGHTS = {
        "lgbm_breakout_v4": 0.3,
        "lgbm_composite_v5": 0.7,
    }
    if len(raw_scores) > 1:
        weights = np.array([MODEL_WEIGHTS.get(mid, 0.5) for mid in raw_scores])
        weights = weights / weights.sum()  # normalize in case of missing models
        stacked = np.column_stack(list(raw_scores.values()))
        consensus = (stacked * weights).sum(axis=1)
        df_score["MLScore"] = pd.Series(consensus, index=df_score.index).rank(pct=True).mul(100).round(1).values
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

    console.print("\n[bold green]✓ ML scoring complete[/bold green]")
