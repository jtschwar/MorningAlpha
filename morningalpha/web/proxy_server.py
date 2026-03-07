#!/usr/bin/env python3
"""
yfinance proxy server — replaces Alpha Vantage for stock detail OHLCV data.
No API key required. Run via `alpha launch`.
"""
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS

# In-memory cache: key -> (data, timestamp)
cache: dict = {}
CACHE_DURATION = timedelta(hours=4)

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
