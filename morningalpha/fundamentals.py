"""Fundamental data fetching and caching for the ML pipeline.

Fetches yfinance .info + quarterly financial statements, caches per-ticker to
data/fundamentals/cache/{ticker}.json, and computes derived features.

Used by `alpha ml dataset` via `load_cached_fundamentals()`.
CLI: `alpha ml fundamentals`
"""
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
import rich_click as click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)
console = Console()

CACHE_DIR = Path("data/fundamentals/cache")
FUNDAMENTALS_DIR = Path("data/fundamentals")
STALE_THRESHOLD = timedelta(days=7)

INFO_FIELDS = [
    "trailingEps", "bookValue", "totalRevenue", "marketCap",
    "returnOnEquity", "debtToEquity", "revenueGrowth", "profitMargins",
    "freeCashflow", "currentRatio", "shortPercentOfFloat",
    "sector", "industry", "currentPrice",
    # Additional fields for spread CSV canonical columns
    "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
    "pegRatio", "earningsGrowth", "returnOnAssets", "grossMargins",
    "operatingMargins", "dividendYield", "beta", "heldPercentInstitutions",
]

SECTOR_MAP = {
    "technology": 0,
    "healthcare": 1,
    "financial services": 2,
    "consumer cyclical": 3,
    "communication services": 4,
    "industrials": 5,
    "consumer defensive": 6,
    "energy": 7,
    "utilities": 8,
    "real estate": 9,
    "basic materials": 10,
}


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (ValueError, TypeError):
        return None


def _safe_divide(a, b) -> Optional[float]:
    a, b = _safe_float(a), _safe_float(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def _encode_sector(sector_str: Optional[str]) -> int:
    if not sector_str:
        return -1
    key = str(sector_str).strip().lower()
    if key in SECTOR_MAP:
        return SECTOR_MAP[key]
    # fuzzy match: find closest key
    for k, v in SECTOR_MAP.items():
        if k in key or key in k:
            return v
    logger.debug("Unrecognized sector: %r", sector_str)
    return -1


def _extract_financial_statements(t: yf.Ticker) -> Dict[str, Any]:
    result = {
        "total_assets_current": None,
        "total_assets_prev": None,
        "net_income": None,
        "operating_cashflow": None,
    }
    try:
        bs = t.quarterly_balance_sheet
        if bs is not None and not bs.empty and bs.shape[1] >= 1:
            for label in ["Total Assets", "TotalAssets"]:
                if label in bs.index:
                    result["total_assets_current"] = _safe_float(bs.loc[label].iloc[0])
                    if bs.shape[1] >= 2:
                        result["total_assets_prev"] = _safe_float(bs.loc[label].iloc[1])
                    break
    except Exception:
        pass

    try:
        inc = t.quarterly_income_stmt
        if inc is not None and not inc.empty and inc.shape[1] >= 1:
            for label in ["Net Income", "NetIncome", "Net Income Common Stockholders"]:
                if label in inc.index:
                    result["net_income"] = _safe_float(inc.loc[label].iloc[0])
                    break
    except Exception:
        pass

    try:
        cf = t.quarterly_cashflow
        if cf is not None and not cf.empty and cf.shape[1] >= 1:
            for label in [
                "Total Cash From Operating Activities",
                "Operating Cash Flow",
                "OperatingCashFlow",
                "Cash Flows From Used In Operating Activities",
            ]:
                if label in cf.index:
                    result["operating_cashflow"] = _safe_float(cf.loc[label].iloc[0])
                    break
    except Exception:
        pass

    return result


def fetch_ticker_fundamentals(ticker: str) -> Optional[Dict[str, Any]]:
    """Fetch all fundamental data for a single ticker. Returns None on failure."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        result: Dict[str, Any] = {
            "ticker": ticker,
            "fetched_at": datetime.now().isoformat(),
        }
        for field in INFO_FIELDS:
            result[field] = info.get(field)

        result.update(_extract_financial_statements(t))

        # 5-year high — computed from price history, not info dict.
        # Used to detect stocks far below their peak ("falling knife" filter).
        # Stored as raw high price; ratio vs current price computed at feature time.
        try:
            hist = t.history(period="5y", auto_adjust=True)
            if hist is not None and not hist.empty and "High" in hist.columns:
                result["high_5yr"] = float(hist["High"].max())
            else:
                result["high_5yr"] = None
        except Exception:
            result["high_5yr"] = None

        non_null = sum(1 for k, v in result.items()
                       if k not in ("ticker", "fetched_at") and v is not None)
        if non_null == 0:
            return None
        return result

    except Exception as exc:
        logger.debug("Failed to fetch fundamentals for %s: %s", ticker, exc)
        return None


def load_cached_fundamentals(ticker: str) -> Optional[Dict[str, Any]]:
    """Load cached fundamental data for a single ticker. Returns None if not cached."""
    cache_file = CACHE_DIR / f"{ticker}.json"
    if not cache_file.exists():
        return None
    try:
        return json.loads(cache_file.read_text())
    except Exception:
        return None


def compute_fundamental_features(fundamentals: Dict[str, Any], close_price: float) -> Dict[str, Any]:
    """Compute the 14 ML features from raw fundamental data and current close price.

    Call this at dataset-build time with the snapshot-date close price.
    """
    mc = _safe_float(fundamentals.get("marketCap"))
    ta = _safe_float(fundamentals.get("total_assets_current"))
    ta_prev = _safe_float(fundamentals.get("total_assets_prev"))
    ni = _safe_float(fundamentals.get("net_income"))
    ocf = _safe_float(fundamentals.get("operating_cashflow"))

    features = {
        "earnings_yield": _safe_divide(fundamentals.get("trailingEps"), close_price),
        "book_to_market": _safe_divide(fundamentals.get("bookValue"), close_price),
        "sales_to_price": _safe_divide(fundamentals.get("totalRevenue"), mc),
        "roe": _safe_float(fundamentals.get("returnOnEquity")),
        "debt_to_equity": _safe_divide(fundamentals.get("debtToEquity"), 100.0),
        "revenue_growth": _safe_float(fundamentals.get("revenueGrowth")),
        "profit_margin": _safe_float(fundamentals.get("profitMargins")),
        "fcf_yield": _safe_divide(fundamentals.get("freeCashflow"), mc),
        "current_ratio": _safe_float(fundamentals.get("currentRatio")),
        "short_pct_float": _safe_float(fundamentals.get("shortPercentOfFloat")),
        "asset_growth": (ta / ta_prev - 1) if (ta is not None and ta_prev is not None and ta_prev != 0) else None,
        "accruals_ratio": ((ni - ocf) / ta) if (ni is not None and ocf is not None and ta is not None and ta != 0) else None,
        "sector": _encode_sector(fundamentals.get("sector")),
        "has_fundamentals": 1,
    }
    return features


def _null_fundamental_features() -> Dict[str, Any]:
    """Return a dict of all fundamental features set to None/0."""
    from morningalpha.ml.features import FUNDAMENTAL_FEATURE_NAMES
    row = {name: None for name in FUNDAMENTAL_FEATURE_NAMES}
    row["has_fundamentals"] = 0
    row["sector"] = -1
    return row


def fetch_universe_fundamentals(
    tickers: List[str],
    batch_size: int = 50,
    pause: float = 2.0,
    ticker_pause: float = 0.5,
    refresh_stale: bool = False,
) -> pd.DataFrame:
    """Fetch fundamentals for all tickers with batching and per-ticker JSON caching.

    Args:
        batch_size: Tickers per batch (pauses `pause` seconds between batches).
        pause: Seconds to sleep between batches.
        ticker_pause: Seconds to sleep between individual ticker fetches within a batch.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    to_fetch = []

    # Fields that must be present in cache; missing any triggers a re-fetch
    _required_fields = set(INFO_FIELDS)

    for ticker in tickers:
        cache_file = CACHE_DIR / f"{ticker}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                fetched_at = datetime.fromisoformat(data.get("fetched_at", "2000-01-01"))
                is_stale = refresh_stale and (datetime.now() - fetched_at) >= STALE_THRESHOLD
                missing_fields = _required_fields - set(data.keys())
                if not is_stale and not missing_fields:
                    results.append(data)
                    continue
            except Exception:
                pass
        to_fetch.append(ticker)

    if to_fetch:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching fundamentals", total=len(to_fetch))
            batches = [to_fetch[i:i + batch_size] for i in range(0, len(to_fetch), batch_size)]
            for bi, batch in enumerate(batches):
                for ti, ticker in enumerate(batch):
                    data = fetch_ticker_fundamentals(ticker)
                    if data is not None:
                        cache_file = CACHE_DIR / f"{ticker}.json"
                        cache_file.write_text(json.dumps(data, default=str))
                        results.append(data)
                    progress.advance(task)
                    if ticker_pause > 0 and ti < len(batch) - 1:
                        time.sleep(ticker_pause)
                if bi < len(batches) - 1:
                    time.sleep(pause)

    df = pd.DataFrame(results) if results else pd.DataFrame()
    return _compile_derived_features(df)


def _compile_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute asset_growth and accruals_ratio from raw statement fields."""
    if df.empty:
        return df

    for col in ("total_assets_current", "total_assets_prev", "net_income", "operating_cashflow"):
        if col not in df.columns:
            df[col] = None

    df["asset_growth"] = None
    mask = df["total_assets_current"].notna() & df["total_assets_prev"].notna() & (df["total_assets_prev"] != 0)
    if mask.any():
        df.loc[mask, "asset_growth"] = (
            df.loc[mask, "total_assets_current"] / df.loc[mask, "total_assets_prev"] - 1
        )

    df["accruals_ratio"] = None
    mask = (
        df["net_income"].notna()
        & df["operating_cashflow"].notna()
        & df["total_assets_current"].notna()
        & (df["total_assets_current"] != 0)
    )
    if mask.any():
        df.loc[mask, "accruals_ratio"] = (
            (df.loc[mask, "net_income"] - df.loc[mask, "operating_cashflow"])
            / df.loc[mask, "total_assets_current"]
        )

    # Compute ML ratios using stored currentPrice
    price_col = "currentPrice" if "currentPrice" in df.columns else None
    mc_col = "marketCap" if "marketCap" in df.columns else None

    df["earnings_yield"] = None
    df["book_to_market"] = None
    df["sales_to_price"] = None
    df["fcf_yield"] = None
    df["sector_encoded"] = -1

    if price_col:
        price_mask = df[price_col].notna() & (df[price_col] != 0)
        if price_mask.any():
            if "trailingEps" in df.columns:
                eps = df.loc[price_mask, "trailingEps"].astype(float, errors="ignore")
                price = df.loc[price_mask, price_col].astype(float, errors="ignore")
                # Only meaningful for profitable companies; cap at 0.20 (P/E floor=5)
                # to prevent data artifacts from pre-revenue stocks dominating sector ranks
                profitable = eps > 0
                df.loc[price_mask & profitable, "earnings_yield"] = (
                    eps[profitable] / price[profitable]
                ).clip(upper=0.20)
            if "bookValue" in df.columns:
                df.loc[price_mask, "book_to_market"] = (
                    df.loc[price_mask, "bookValue"].astype(float, errors="ignore")
                    / df.loc[price_mask, price_col].astype(float, errors="ignore")
                )

    if mc_col:
        mc_mask = df[mc_col].notna() & (df[mc_col] != 0)
        if mc_mask.any():
            if "totalRevenue" in df.columns:
                df.loc[mc_mask, "sales_to_price"] = (
                    df.loc[mc_mask, "totalRevenue"].astype(float, errors="ignore")
                    / df.loc[mc_mask, mc_col].astype(float, errors="ignore")
                )
            if "freeCashflow" in df.columns:
                df.loc[mc_mask, "fcf_yield"] = (
                    df.loc[mc_mask, "freeCashflow"].astype(float, errors="ignore")
                    / df.loc[mc_mask, mc_col].astype(float, errors="ignore")
                )

    if "sector" in df.columns:
        df["sector_encoded"] = df["sector"].apply(_encode_sector)

    # 5-year high proximity — (current_price - 5yr_high) / 5yr_high.
    # Range: [-1, 0] where 0 = at peak, -0.96 = 96% below peak.
    # Stocks like API ($3.68 vs $100 ATH) get -0.963; momentum stocks near ATH get ~0.
    df["price_vs_5yr_high"] = None
    if price_col and "high_5yr" in df.columns:
        mask = (
            df[price_col].notna() & (df[price_col] != 0)
            & df["high_5yr"].notna() & (df["high_5yr"] != 0)
        )
        if mask.any():
            price = df.loc[mask, price_col].astype(float)
            high = df.loc[mask, "high_5yr"].astype(float)
            df.loc[mask, "price_vs_5yr_high"] = (price - high) / high

    return df


# ---------------------------------------------------------------------------
# EDGAR validation helpers (optional — used by --validate flag)
# ---------------------------------------------------------------------------

REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
]


def _fetch_edgar_facts(ticker: str, cik: str) -> Optional[Dict]:
    import requests
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    headers = {"User-Agent": "MorningAlpha research@morningalpha.com"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _extract_edgar_value(edgar_data: Dict, tags: List[str]) -> Optional[float]:
    try:
        facts = edgar_data.get("facts", {})
        us_gaap = facts.get("us-gaap", {})
        for tag in tags:
            if tag in us_gaap:
                units = us_gaap[tag].get("units", {})
                usd = units.get("USD", [])
                annual = [e for e in usd if e.get("form") in ("10-K", "10-Q")]
                if annual:
                    annual.sort(key=lambda x: x.get("end", ""), reverse=True)
                    return _safe_float(annual[0].get("val"))
    except Exception:
        pass
    return None


def _validate_fundamentals(yf_data: Dict, edgar_data: Dict, ticker: str) -> List[str]:
    warnings_out = []
    edgar_revenue = _extract_edgar_value(edgar_data, REVENUE_TAGS)
    yf_revenue = _safe_float(yf_data.get("totalRevenue"))
    if edgar_revenue and yf_revenue:
        diff_pct = abs(edgar_revenue - yf_revenue) / max(abs(edgar_revenue), 1)
        if diff_pct > 0.10:
            warnings_out.append(
                f"{ticker}: Revenue mismatch — yfinance={yf_revenue:.0f}, "
                f"EDGAR={edgar_revenue:.0f} ({diff_pct:.0%} diff)"
            )
    return warnings_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("fundamentals")
@click.option("--tickers-from", "tickers_from", default="data/latest/stocks_3m.csv", show_default=True)
@click.option("--output", default="data/fundamentals/fundamentals.parquet", show_default=True)
@click.option("--batch-size", "batch_size", default=50, show_default=True)
@click.option("--pause", default=2.0, show_default=True, help="Seconds between batches.")
@click.option("--ticker-pause", "ticker_pause", default=0.5, show_default=True, help="Seconds between individual ticker fetches.")
@click.option("--refresh-stale", "refresh_stale", is_flag=True, default=False, help="Re-fetch tickers older than 7 days.")
@click.option("--validate", is_flag=True, default=False, help="Cross-validate top N against SEC EDGAR.")
@click.option("--top", default=50, show_default=True, help="Number of tickers to validate by market cap.")
@click.option("--status", is_flag=True, default=False, help="Print cache summary and exit.")
def fundamentals_cmd(tickers_from, output, batch_size, pause, ticker_pause, refresh_stale, validate, top, status):
    """Fetch and cache fundamental data for the ML feature set.

    \b
    Examples:
      alpha ml fundamentals --tickers-from data/latest/stocks_3m.csv
      alpha ml fundamentals --refresh-stale
      alpha ml fundamentals --validate --top 50
      alpha ml fundamentals --status
    """
    if status:
        cache_files = list(CACHE_DIR.glob("*.json"))
        console.print(f"Cache directory: {CACHE_DIR}")
        console.print(f"Cached tickers:  {len(cache_files)}")
        if cache_files:
            ages = []
            for f in cache_files:
                try:
                    data = json.loads(f.read_text())
                    fetched_at = datetime.fromisoformat(data.get("fetched_at", "2000-01-01"))
                    ages.append((datetime.now() - fetched_at).days)
                except Exception:
                    pass
            if ages:
                console.print(f"Median age: {sorted(ages)[len(ages)//2]}d  Max age: {max(ages)}d")
        return

    # Load tickers
    tickers_path = Path(tickers_from)
    if not tickers_path.exists():
        console.print(f"[red]Ticker file not found: {tickers_from}[/red]")
        return
    df_t = pd.read_csv(tickers_path)
    ticker_col = next((c for c in ("Ticker", "ticker", "Symbol") if c in df_t.columns), None)
    if ticker_col is None:
        console.print("[red]No Ticker column found in input CSV.[/red]")
        return
    tickers = df_t[ticker_col].dropna().astype(str).str.strip().tolist()
    console.print(f"Fetching fundamentals for [bold]{len(tickers)}[/bold] tickers")

    df = fetch_universe_fundamentals(tickers, batch_size=batch_size, pause=pause, ticker_pause=ticker_pause, refresh_stale=refresh_stale)

    if df.empty:
        console.print("[red]No data fetched.[/red]")
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    console.print(f"[bold green]Saved {len(df)} rows → {output_path}[/bold green]")

    # Optional EDGAR validation
    if validate and "marketCap" in df.columns:
        console.print(f"\n[bold]Validating top {top} by market cap against SEC EDGAR...[/bold]")
        top_df = df.nlargest(top, "marketCap")
        all_warnings = []
        for _, row in top_df.iterrows():
            ticker = row.get("ticker", "")
            # CIK lookup would be needed here; for now log tickers without CIK
            console.print(f"  [yellow]CIK lookup not implemented — skipping EDGAR for {ticker}[/yellow]")
            break
        if all_warnings:
            for w in all_warnings:
                console.print(f"  [yellow]{w}[/yellow]")
        else:
            console.print("  No data quality warnings found.")
