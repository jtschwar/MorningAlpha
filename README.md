# MorningAlpha
A professional-grade stock analysis application that combines Python-based data collection with interactive web visualization to deliver comprehensive market insights and real-time stock analysis.

## Overview

MorningAlpha transforms basic stock screening into a sophisticated analytical tool, providing meaningful financial insights through multiple chart types, technical indicators, and user-friendly interfaces. The system analyzes NASDAQ and S&P 500 stocks, generating both detailed CSV reports and interactive web dashboards for comprehensive market analysis.

## Features

- **Multi-Index Coverage**: Analyze stocks from NASDAQ and S&P 500
- **Reliable Data Integration**: Alpha Vantage API for consistent financial data
- **Interactive Visualizations**: 
  - Bar charts for performance comparison
  - Scatter plots for correlation analysis
  - Treemaps and sunburst charts for market composition
  - Candlestick charts for detailed price action
  - Volume analysis and returns distribution
- **Technical Indicators**: Comprehensive technical analysis tools
- **Enhanced User Experience**: Rich progress bars and interactive CLI
- **Excel-Compatible Output**: CSV format for easy data manipulation
- **Deployment-Ready**: Modular architecture suitable for production environments

## Installation
```bash
# Clone the repository
git clone https://github.com/jtschwar/morningalpha.git
cd morningalpha

# Install with pip
pip install -e .
```

## Usage

### Command Line Interface
```bash
# Run stock analysis
morningalpha analyze -bs 1024 -m 1m

# Start the web server
morningalpha launch
```

The CLI features lazy loading for fast startup times and rich progress indicators for better user experience.