#!/usr/bin/env python3
"""
Alpha Vantage proxy server - Better for free tier
Get free API key: https://www.alphavantage.co/support/#api-key
"""
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta

# Global cache
cache = {}
CACHE_DURATION = timedelta(hours=4)


# Lazy import - only load requests when needed
def _lazy_imports():
    import requests
    return requests


def get_cached_data(ticker):
    if ticker in cache:
        data, timestamp = cache[ticker]
        if datetime.now() - timestamp < CACHE_DURATION:
            return data
    return None


def get_api_key():
    """
    Get the API key from the keys module.
    
    Returns:
        API key string or None if not found
    """
    try:
        from morningalpha import keys
        return keys.get_alpha_vantage_key()
    except ImportError:
        # Fallback to environment variable if keys module not available
        return os.environ.get('ALPHA_VANTAGE_API_KEY')


def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/api/stock/<ticker>')
    def get_stock(ticker):
        requests = _lazy_imports()
        api_key = get_api_key()
        period = request.args.get('period', '3M')
        
        # Check cache
        cached = get_cached_data(ticker)
        if cached:
            print(f"✅ Cache hit for {ticker}")
            return jsonify(cached)
        
        try:
            # Alpha Vantage API call
            url = f'https://www.alphavantage.co/query'
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': ticker,
                'apikey': api_key,
                'outputsize': 'full'
            }
            
            print(f"📊 Fetching {ticker} from Alpha Vantage...")
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            # Check for errors
            if 'Error Message' in data:
                return jsonify({'error': f'Invalid ticker: {ticker}'}), 404
            
            if 'Note' in data:
                return jsonify({
                    'error': 'API rate limit (25 calls/day). Please wait or use demo mode.',
                    'suggestion': 'Rate limit resets daily. Try again tomorrow.'
                }), 429
            
            if 'Information' in data:
                return jsonify({
                    'error': 'API limit reached',
                    'message': data['Information']
                }), 429
            
            time_series = data.get('Time Series (Daily)', {})
            
            if not time_series:
                return jsonify({
                    'error': 'No data available',
                    'raw_response': data
                }), 404
            
            # Get last 90 days of data
            dates = sorted(time_series.keys(), reverse=True)[:90]
            dates = sorted(dates)  # Re-sort chronologically
            
            result = {
                'timestamps': [int(datetime.strptime(d, '%Y-%m-%d').timestamp()) for d in dates],
                'open': [float(time_series[d]['1. open']) for d in dates],
                'high': [float(time_series[d]['2. high']) for d in dates],
                'low': [float(time_series[d]['3. low']) for d in dates],
                'close': [float(time_series[d]['4. close']) for d in dates],
                'volume': [int(time_series[d]['5. volume']) for d in dates]
            }
            
            # Cache the result
            cache[ticker] = (result, datetime.now())
            
            print(f"✅ Successfully fetched {ticker} ({len(result['timestamps'])} days)")
            return jsonify(result)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/test')
    def test():
        """Test endpoint"""
        requests = _lazy_imports()
        api_key = get_api_key()
        
        try:
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': 'IBM',
                'apikey': api_key
            }
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                return jsonify({
                    'status': 'success',
                    'message': 'API key is working!',
                    'sample_date': list(data['Time Series (Daily)'].keys())[0]
                })
            else:
                return jsonify({
                    'status': 'error',
                    'response': data
                }), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/cache/clear')
    def clear_cache():
        cache.clear()
        return jsonify({'message': f'Cache cleared'})
    
    @app.route('/api/cache/status')
    def cache_status():
        return jsonify({
            'cached_stocks': list(cache.keys()),
            'count': len(cache)
        })
    
    return app


def run_server(port=5050, debug=False):
    """
    Run the Flask server.
    
    Args:
        port: Port to run the server on
        debug: Enable debug mode
    """
    # Get API key
    api_key = get_api_key()
    
    if not api_key:
        print("⚠️  WARNING: No API key found!")
        print("🔑 Get one free at: https://www.alphavantage.co/support/#api-key")
        print("💡 Tip: Use 'alpha-serve' command to configure your API key")
        return
    
    # Suppress Flask development server warning
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    
    app = create_app()
    
    print("🚀 Alpha Vantage Proxy Server")
    print(f"🔑 API Key: {api_key[:10]}...")
    print(f"📊 Server: http://localhost:{port}")
    print(f"🧪 Test: http://localhost:{port}/api/test")
    print("\n⚠️  Free tier: 25 API calls per day")
    print("💡 Tip: Data is cached for 4 hours to save API calls")
    
    app.run(port=port, debug=debug)


if __name__ == '__main__':
    # For standalone usage
    FLASK_PORT = int(os.environ.get('FLASK_PORT', 5050))
    run_server(port=FLASK_PORT, debug=False)