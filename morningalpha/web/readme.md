# Stock Gainers Dashboard with Dask

An interactive web dashboard for analyzing top stock gainers with detailed price charts, technical indicators, and multiple visualization types.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Data (CLI)

```bash
# Get top 200 gainers over 3 months from NASDAQ and S&P 500
python dask_stock_analyzer.py --metric 3m --top 200 --output stock_data.csv

# Get top 100 gainers YTD from NASDAQ only
python dask_stock_analyzer.py --universe nasdaq --metric ytd --top 100 --output stock_data.csv

# Use more workers for faster processing
python dask_stock_analyzer.py --workers 8 --metric 6m --output stock_data.csv
```

### 3. Launch the Dashboard

```bash
# Start local web server
python -m http.server 8000

# Open in browser: http://localhost:8000
```

### 4. Explore the Dashboard

1. **Load Data**: Click "Choose File" and select `stock_data.csv`
2. **Overview**: View statistics and multiple chart types (bar, scatter, treemap, sunburst)
3. **Drill Down**: Click any stock to see:
   - Candlestick price chart
   - Trading volume
   - Daily returns distribution
   - Technical indicators (MAs, volatility, max drawdown)

## 📊 Features

### Main Dashboard (index.html)
- **Multiple Chart Types**: 
  - Bar chart (default)
  - Scatter plot by exchange
  - Treemap visualization
  - Sunburst chart
- **Interactive Filtering**: Top N stocks, search by ticker
- **Statistics Cards**: Key metrics at a glance
- **Clickable Elements**: Click any chart point or table row to drill down

### Stock Detail Page (stock-detail.html)
- **Real-time Data**: Fetches historical prices from Yahoo Finance
- **Candlestick Charts**: OHLC price visualization
- **Volume Analysis**: Trading volume over time
- **Returns Distribution**: Histogram of daily returns
- **Technical Indicators**:
  - 20-day & 50-day moving averages
  - Volatility (standard deviation)
  - Maximum drawdown

### Dask Backend
- **Parallel Processing**: Concurrent stock data fetching
- **Scalable**: Adjust worker count for your system
- **Progress Tracking**: Real-time updates
- **CSV Output**: Smaller files, Excel-compatible

## 📁 Project Structure

```
.
├── index.html                    # Main dashboard
├── stock-detail.html             # Individual stock analysis
├── css/
│   └── styles.css                # Shared styles
├── js/
│   ├── charts.js                 # Chart creation utilities
│   └── data-loader.js            # CSV parsing & data utilities
├── dask_stock_analyzer.py        # Dask-powered data generator
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── stock_data.csv                # Generated data (after running analyzer)
```

## 🎯 Use Cases

1. **Market Research**: Identify high-performing stocks and trends
2. **Portfolio Analysis**: Analyze individual stock performance
3. **Educational Tool**: Learn about technical analysis and indicators
4. **Quick Screening**: Filter and rank stocks by multiple criteria

## 🌐 Deploying to GitHub Pages

### Option 1: Static Deployment (No Real-time Data)

```bash
git init
git add index.html stock-detail.html css/ js/ stock_data.csv
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/stock-dashboard.git
git push -u origin main
```

Enable GitHub Pages in Settings → Pages → Deploy from `main` branch.

**Note**: Stock detail page won't fetch real-time data on GitHub Pages due to CORS restrictions. See Option 2 for full functionality.

### Option 2: With Backend (Full Functionality)

For real-time stock data fetching, you need a backend proxy:

1. Deploy the frontend to GitHub Pages (as above)
2. Create a simple backend API (Flask, FastAPI, etc.) to proxy Yahoo Finance requests
3. Deploy backend to Heroku, Railway, or similar
4. Update `stock-detail.html` to point to your backend endpoint

Example backend endpoint:
```python
# app.py (Flask example)
from flask import Flask, jsonify, request
import yfinance as yf

app = Flask(__name__)

@app.route('/api/stock/<ticker>')
def get_stock(ticker):
    period = request.args.get('period', '3mo')
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return jsonify(data.to_dict())
```

## ⚡ Performance Tips

- **More Workers**: Use 8-16 workers on powerful machines
- **Smaller Universe**: Use `--universe nasdaq` for faster processing
- **Browser Cache**: Stock detail data is cached in localStorage
- **Batch Processing**: Dask processes 100 tickers per chunk

## 🔧 Troubleshooting

**Issue**: Yahoo Finance data not loading in stock detail page
- **Solution**: CORS restrictions prevent direct API calls from file:// or GitHub Pages. Use a local server (`python -m http.server`) or deploy a backend proxy.

**Issue**: "Could not download NASDAQ symbol directory"
- **Solution**: Check internet connection, endpoints may be temporarily down

**Issue**: Charts not rendering
- **Solution**: Ensure Plotly CDN is accessible, check browser console for errors

**Issue**: CSV parsing errors
- **Solution**: Ensure CSV has proper headers and no malformed rows

## 🛠️ Customization

### Add New Chart Types

Edit `js/charts.js` and add a new function:

```javascript
function createMyChart(data, metadata, container) {
    const trace = {
        // Your Plotly configuration
    };
    Plotly.newPlot(container, [trace], layout);
}
```

Then add the option to the dropdown in `index.html`.

### Customize Styling

Edit `css/styles.css` to change:
- Color scheme (update gradient colors)
- Card layouts
- Chart dimensions
- Responsive breakpoints

### Add More Technical Indicators

Edit `stock-detail.html` and add calculations in the `calculateTechnicals()` function.

## 📊 Data Format

The CSV should have these columns:
```
Rank,Ticker,Name,Exchange,Return_3M_%
1,AAPL,Apple Inc.,NASDAQ,25.5
```

## 🤝 Contributing

Feel free to submit issues or pull requests for:
- New chart types
- Additional technical indicators
- Performance improvements
- UI/UX enhancements

## 📄 License

MIT License - free to use and modify.

## 🙏 Acknowledgments

- **Plotly.js**: Interactive charts
- **yfinance**: Stock data
- **Dask**: Parallel processing
- **Yahoo Finance**: Real-time stock data
