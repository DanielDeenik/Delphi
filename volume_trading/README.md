# Volume Trading System

A simple system for analyzing volume patterns in stocks.

## Overview

The Volume Trading System is designed to track stocks, detect volume patterns, and generate trading signals based on volume inefficiencies. The system uses a modular architecture with the following components:

1. **Data Fetcher**: Retrieves stock data from Alpha Vantage
2. **Volume Analyzer**: Detects volume patterns and inefficiencies
3. **Data Storage**: Saves and loads data and analysis results
4. **Notebook Generator**: Creates interactive Jupyter notebooks for analysis

## Features

- **Volume Analysis**: Detects volume spikes and patterns
- **Signal Generation**: Generates trading signals based on volume patterns
- **Local Storage**: Saves data and analysis results locally
- **Summary Reports**: Provides summaries of analysis results
- **Interactive Notebooks**: Jupyter notebooks for each stock and a master dashboard

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/volume-trading.git
cd volume-trading

# Install the package
pip install -e .
```

## Configuration

1. Create a `.env` file with your API key:

```
ALPHA_VANTAGE_API_KEY=your_api_key
```

2. Configure the tracked stocks in `config/config.json`:

```json
{
  "tracked_stocks": {
    "buy": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    "short": ["BIDU", "NIO", "PINS", "SNAP", "COIN"]
  }
}
```

## Usage

### Fetch Data

```bash
# Fetch data for all tracked stocks
python -m volume_trading.run fetch

# Fetch data for specific stocks
python -m volume_trading.run fetch --tickers AAPL MSFT

# Force full refresh
python -m volume_trading.run fetch --force
```

### Analyze Data

```bash
# Analyze data for all tracked stocks
python -m volume_trading.run analyze

# Analyze data for specific stocks
python -m volume_trading.run analyze --tickers AAPL MSFT
```

### Print Summary

```bash
# Print master summary
python -m volume_trading.run summary

# Print summary for a specific stock
python -m volume_trading.run summary --ticker AAPL
```

### Run All Steps

```bash
# Fetch and analyze data for all tracked stocks
python -m volume_trading.run run

# Fetch and analyze data for specific stocks
python -m volume_trading.run run --tickers AAPL MSFT

# Force full refresh
python -m volume_trading.run run --force
```

### Generate and Use Notebooks

```bash
# Import data and generate notebooks
python -m volume_trading.run_all

# Force full refresh of data
python -m volume_trading.run_all --force-refresh

# Open the master dashboard notebook
jupyter notebook notebooks/master_dashboard.ipynb

# Open a specific stock notebook
jupyter notebook notebooks/AAPL_analysis.ipynb
```

## Project Structure

```
volume_trading/
├── __init__.py
├── config.py
├── run.py
├── run_all.py
├── import_all_data.py
├── generate_notebooks.py
│
├── core/
│   ├── __init__.py
│   ├── data_fetcher.py
│   ├── volume_analyzer.py
│   └── data_storage.py
│
└── notebooks/
    ├── __init__.py
    └── generator.py
```

## Next Steps

- Enhance visualization capabilities
- Implement sentiment analysis
- Add BigQuery integration for cloud storage
- Implement reinforcement learning models
- Add Discord bot integration for alerts

## License

This project is licensed under the MIT License - see the LICENSE file for details.
