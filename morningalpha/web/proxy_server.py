#!/usr/bin/env python3
"""
Alpha Vantage proxy server - Enhanced with flexible period support
Get free API key: https://www.alphavantage.co/support/#api-key
"""
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta

# Global cache
cache = {}
CACHE_DURATION = timedelta(hours=4)

# Period to days mapping for data fetching
PERIOD_DAYS = {
    '1M': 30,
    '3M': 90,
    '6M': 180,
    '1Y': 365,
    '5Y': 1825,
    'MAX': 5000  # Alpha Vantage full history
}


def _lazy_imports():
    """Lazy import requests when needed."""
    import requests
    return requests


def get_cached_data(cache_key):
    """Get data from cache if available and not expired."""
    if cache_key in cache:
        data, timestamp = cache[cache_key]
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
        
        # Check if API key exists
        if not api_key:
            return jsonify({
                'error': 'No API key configured. Please run "morningalpha launch" to set up your Alpha Vantage API key.',
                'suggestion': 'Get a free API key at https://www.alphavantage.co/support/#api-key'
            }), 400
        
        # Create cache key with period
        cache_key = f"{ticker}_{period}"
        
        # Check cache
        cached = get_cached_data(cache_key)
        if cached:
            print(f"✅ Cache hit for {ticker} ({period})")
            return jsonify(cached)
        
        try:
            # Alpha Vantage API call
            # Note: Free tier only supports 'compact' (last 100 data points)
            # 'full' requires premium subscription
            url = f'https://www.alphavantage.co/query'
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': ticker,
                'apikey': api_key,
                'outputsize': 'compact'  # Free tier: last 100 data points
            }
            
            print(f"📊 Fetching {ticker} ({period}) from Alpha Vantage...")
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            # Debug: print response to see what Alpha Vantage is returning
            print(f"🔍 Alpha Vantage response keys: {list(data.keys())}")
            
            # Check for errors - be more specific
            if 'Error Message' in data:
                error_msg = data['Error Message']
                print(f"❌ Alpha Vantage error: {error_msg}")
                return jsonify({
                    'error': f'Alpha Vantage error: {error_msg}',
                    'ticker': ticker
                }), 404
            
            # Check for rate limit - be specific about the "Note" field
            if 'Note' in data:
                note_msg = data['Note']
                print(f"⚠️ Rate limit detected: {note_msg}")
                # Only return rate limit if it actually mentions rate limit
                if 'rate limit' in note_msg.lower() or 'call' in note_msg.lower():
                    return jsonify({
                        'error': 'API rate limit (25 calls/day). Please wait or use demo mode.',
                        'suggestion': 'Rate limit resets daily. Try again tomorrow.',
                        'note': note_msg
                    }), 429
                else:
                    # Some other note - return it but don't call it a rate limit
                    return jsonify({
                        'error': f'API returned note: {note_msg}',
                        'raw_response': data
                    }), 400
            
            # Check for "Information" field - this is usually about API limits
            if 'Information' in data:
                info_msg = data['Information']
                print(f"⚠️ API information: {info_msg}")
                # Check if it's actually about rate limits
                if 'call' in info_msg.lower() or 'limit' in info_msg.lower():
                    return jsonify({
                        'error': 'API limit reached',
                        'message': info_msg
                    }), 429
                else:
                    # Other information - return as warning, not error
                    return jsonify({
                        'error': f'API returned information: {info_msg}',
                        'raw_response': data
                    }), 400
            
            time_series = data.get('Time Series (Daily)', {})
            
            if not time_series:
                return jsonify({
                    'error': 'No data available',
                    'raw_response': data
                }), 404
            
            # Calculate how many days we need
            days_needed = PERIOD_DAYS.get(period, 90)
            
            # Get the required number of days of data
            dates = sorted(time_series.keys(), reverse=True)[:days_needed]
            dates = sorted(dates)  # Re-sort chronologically
            
            result = {
                'timestamps': [int(datetime.strptime(d, '%Y-%m-%d').timestamp()) for d in dates],
                'open': [float(time_series[d]['1. open']) for d in dates],
                'high': [float(time_series[d]['2. high']) for d in dates],
                'low': [float(time_series[d]['3. low']) for d in dates],
                'close': [float(time_series[d]['4. close']) for d in dates],
                'volume': [int(time_series[d]['5. volume']) for d in dates],
                'period': period
            }
            
            # Cache the result
            cache[cache_key] = (result, datetime.now())
            
            print(f"✅ Successfully fetched {ticker} ({len(result['timestamps'])} days, {period})")
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
        """Clear all cached data"""
        count = len(cache)
        cache.clear()
        return jsonify({'message': f'Cache cleared ({count} entries removed)'})
    
    @app.route('/api/cache/status')
    def cache_status():
        """Get cache status"""
        return jsonify({
            'cached_items': [key for key in cache.keys()],
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
        print("💡 Tip: Use 'morningalpha launch' command to configure your API key")
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
    print("📅 Supported periods: 1M, 3M, 6M, 1Y, 5Y, MAX")
    
    app.run(port=port, debug=debug)


if __name__ == '__main__':
    # For standalone usage
    FLASK_PORT = int(os.environ.get('FLASK_PORT', 5050))
    run_server(port=FLASK_PORT, debug=False)