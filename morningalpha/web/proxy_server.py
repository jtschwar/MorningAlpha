#!/usr/bin/env python3
"""
yfinance proxy server — replaces Alpha Vantage for stock detail OHLCV data.
No API key required. Run via `alpha launch`.
"""
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS

# In-memory cache: key -> (data, timestamp)
cache: dict = {}
CACHE_DURATION = timedelta(hours=4)

# ---------------------------------------------------------------------------
# LSTM forecast — lazy-loaded singleton (model + checkpoint)
# ---------------------------------------------------------------------------
_lstm_model = None
_lstm_ckpt  = None
_lstm_device = None

def _get_lstm_model():
    """Load LSTM model once and keep it in memory across requests."""
    global _lstm_model, _lstm_ckpt, _lstm_device
    if _lstm_model is not None:
        return _lstm_model, _lstm_ckpt, _lstm_device

    try:
        from morningalpha.ml.inference import _lstm_load_model_and_device, ckpt_dropout
        import torch
        _REPO_ROOT  = Path(__file__).parents[2]
        _MODEL_DIR  = _REPO_ROOT / "models"
        # Find the first available LSTM checkpoint
        candidates = sorted(_MODEL_DIR.glob("lstm_*.pt"))
        if not candidates:
            return None, None, None
        model_path = candidates[-1]  # most recently modified
        ckpt_tmp   = torch.load(str(model_path), map_location="cpu", weights_only=False)
        dropout    = float(ckpt_tmp.get("config", {}).get("dropout", 0.3))
        _lstm_model, _lstm_ckpt, _lstm_device = _lstm_load_model_and_device(
            model_path, dropout=dropout
        )
        print(f"LSTM model loaded: {model_path.name} (device={_lstm_device})")
    except Exception as exc:
        print(f"LSTM model load failed: {exc}")
        return None, None, None

    return _lstm_model, _lstm_ckpt, _lstm_device


def fetch_forecast(ticker: str, n_paths: int = 6) -> dict:
    """Generate MC-dropout forecast paths for a single ticker on demand.

    Loads only the requested ticker's rows from dataset.parquet via
    pyarrow predicate pushdown — fast (~100ms) without loading full dataset.
    """
    import math
    import numpy as np
    import pandas as pd
    import torch

    model, ckpt, device = _get_lstm_model()
    if model is None:
        raise RuntimeError("LSTM model not available")

    feat_cols    = ckpt["feature_cols"]
    lookback     = ckpt.get("lookback", 60)
    horizon_days = ckpt.get("horizon_days", [1, 5, 10, 21, 63])
    scaler_info  = ckpt.get("feature_scaler", {})

    # Load only this ticker's rows — pyarrow handles the filter efficiently
    _REPO_ROOT   = Path(__file__).parents[2]
    dataset_path = _REPO_ROOT / "data" / "training" / "dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    grp = pd.read_parquet(dataset_path, filters=[("ticker", "==", ticker)])
    if len(grp) < lookback:
        raise ValueError(f"{ticker}: only {len(grp)} rows in dataset (need {lookback})")

    grp = grp.sort_values("date")

    # Apply stored scaler
    if scaler_info and "cols" in scaler_info:
        full_cols = scaler_info["cols"]
        mean_map  = dict(zip(full_cols, scaler_info["mean"]))
        scale_map = dict(zip(full_cols, scaler_info["scale"]))
        for col in [c for c in full_cols if c in grp.columns]:
            grp[col] = (grp[col].fillna(0) - mean_map[col]) / max(scale_map[col], 1e-8)

    for c in [c for c in feat_cols if c not in grp.columns]:
        grp[c] = 0.0

    seq = np.nan_to_num(grp[feat_cols].tail(lookback).values.astype(np.float32), nan=0.0)
    x   = torch.tensor(seq[np.newaxis]).to(device)  # [1, lookback, F]

    # Fetch last close price via yfinance for price-level conversion
    last_price = None
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d")
        if not hist.empty:
            last_price = round(float(hist["Close"].iloc[-1]), 4)
    except Exception:
        pass

    # MC dropout — N forward passes
    paths = []
    for _ in range(n_paths):
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0]  # [n_horizons]
        # Filter: reject paths implying price < 0 or > 3× last_price
        if last_price:
            implied = [last_price * math.exp(float(v)) for v in pred]
            if any(p <= 0 or p > last_price * 3 for p in implied):
                continue
        paths.append([round(float(v), 5) for v in pred])

    if not paths:
        raise ValueError(f"{ticker}: all {n_paths} MC paths were filtered as unstable")

    return {
        "ticker":      ticker,
        "horizons":    horizon_days,
        "last_price":  last_price,
        "paths":       paths,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

YF_PERIOD_MAP = {
    '1M': '1mo',
    '3M': '3mo',
    '6M': '6mo',
    '1Y': '1y',
    '5Y': '5y',
    'MAX': 'max',
}


def get_cached(key: str):
    if key in cache:
        data, ts = cache[key]
        if datetime.now() - ts < CACHE_DURATION:
            return data
    return None


def fetch_ohlcv(ticker: str, period: str) -> dict:
    import yfinance as yf
    yf_period = YF_PERIOD_MAP.get(period, '3mo')
    hist = yf.Ticker(ticker).history(period=yf_period)
    hist = hist.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    if hist.empty:
        raise ValueError(f"No data returned for {ticker}")
    timestamps = [int(ts.timestamp()) for ts in hist.index]
    return {
        'timestamps': timestamps,
        'open':   [round(v, 4) for v in hist['Open'].tolist()],
        'high':   [round(v, 4) for v in hist['High'].tolist()],
        'low':    [round(v, 4) for v in hist['Low'].tolist()],
        'close':  [round(v, 4) for v in hist['Close'].tolist()],
        'volume': [int(v) for v in hist['Volume'].tolist()],
        'period': period,
    }


def fetch_fundamentals(ticker: str) -> dict:
    import yfinance as yf
    import math
    info = yf.Ticker(ticker).info

    def safe(key):
        v = info.get(key)
        if v is None:
            return None
        try:
            return None if math.isnan(float(v)) else v
        except (TypeError, ValueError):
            return None

    return {
        'pe':            safe('trailingPE'),
        'forwardPE':     safe('forwardPE'),
        'pb':            safe('priceToBook'),
        'debtToEquity':  safe('debtToEquity'),
        'netMargin':     safe('profitMargins'),
        'revenueGrowth': safe('revenueGrowth'),
        'roe':           safe('returnOnEquity'),
        'dividendYield': safe('dividendYield'),
    }


def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route('/api/stock/<ticker>')
    def get_stock(ticker):
        period = request.args.get('period', '3M').upper()
        cache_key = f"{ticker}_{period}"

        cached = get_cached(cache_key)
        if cached:
            print(f"Cache hit: {ticker} ({period})")
            return jsonify(cached)

        print(f"Fetching {ticker} ({period}) via yfinance...")
        try:
            data = fetch_ohlcv(ticker, period)
        except ValueError as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return jsonify({'error': str(e)}), 500

        cache[cache_key] = (data, datetime.now())
        print(f"Fetched {ticker}: {len(data['timestamps'])} trading days")
        return jsonify(data)

    @app.route('/api/fundamentals/<ticker>')
    def get_fundamentals(ticker):
        cache_key = f"fundamentals_{ticker}"
        cached = get_cached(cache_key)
        if cached:
            print(f"Cache hit: fundamentals {ticker}")
            return jsonify(cached)

        print(f"Fetching fundamentals for {ticker}...")
        try:
            data = fetch_fundamentals(ticker)
        except Exception as e:
            print(f"Error fetching fundamentals for {ticker}: {e}")
            return jsonify({'error': str(e)}), 500

        cache[cache_key] = (data, datetime.now())
        return jsonify(data)

    @app.route('/api/forecast/<ticker>')
    def get_forecast(ticker):
        ticker = ticker.upper()
        n_paths = int(request.args.get('n_paths', 6))
        cache_key = f"forecast_{ticker}_{n_paths}"

        cached = get_cached(cache_key)
        if cached:
            print(f"Cache hit: forecast {ticker}")
            return jsonify(cached)

        print(f"Generating LSTM forecast for {ticker} ({n_paths} paths)...")
        try:
            data = fetch_forecast(ticker, n_paths=n_paths)
        except (ValueError, FileNotFoundError) as e:
            return jsonify({'error': str(e)}), 404
        except RuntimeError as e:
            return jsonify({'error': str(e)}), 503
        except Exception as e:
            print(f"Forecast error for {ticker}: {e}")
            return jsonify({'error': str(e)}), 500

        cache[cache_key] = (data, datetime.now())
        return jsonify(data)

    @app.route('/api/test')
    def test():
        try:
            data = fetch_ohlcv('IBM', '1M')
            return jsonify({
                'status': 'success',
                'message': 'yfinance is working',
                'sample_date': datetime.fromtimestamp(data['timestamps'][-1]).strftime('%Y-%m-%d'),
            })
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @app.route('/api/cache/clear')
    def clear_cache():
        count = len(cache)
        cache.clear()
        return jsonify({'message': f'Cache cleared ({count} entries removed)'})

    @app.route('/api/cache/status')
    def cache_status():
        return jsonify({
            'cached_items': list(cache.keys()),
            'count': len(cache),
        })

    return app


def run_server(port: int = 5050, debug: bool = False):
    import logging
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    app = create_app()
    print(f"yfinance proxy server running on http://localhost:{port}")
    print(f"Test: http://localhost:{port}/api/test")
    print("No API key required — powered by yfinance")
    app.run(port=port, debug=debug)


if __name__ == '__main__':
    import os
    run_server(port=int(os.environ.get('FLASK_PORT', 5050)))
