# Volume Intelligence Trading System

An AI-powered trading system focused on volume inefficiencies, sentiment analysis, and reinforcement learning.

## Overview

The Volume Intelligence Trading System is designed to track 20 stocks, detect volume inefficiencies, and generate intelligent trade signals with traceable logic. The system uses a modular architecture with the following components:

1. **Data Ingestion**: Fetches market data from Alpha Vantage and stores it in BigQuery
2. **Volume Analysis**: Detects volume patterns and inefficiencies
3. **Sentiment Analysis**: Analyzes sentiment from news and social media
4. **Paper Trading**: Simulates trades with realistic costs and fees
5. **Reinforcement Learning**: (Coming soon) Trains RL agents for each ticker

## Features

- **Transparency**: No black box logic - every signal has a clear explanation
- **Volume Focus**: Detects supply/demand imbalances through volume analysis
- **Mosaic Approach**: Combines multiple data sources for better insights
- **Cost-Aware**: Accounts for spreads, financing costs, and other trading expenses
- **Performance Tracking**: Logs trades and tracks performance metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/tradingai/trading-ai.git
cd trading-ai

# Install the package
pip install -e .

# Install optional dependencies
pip install -e ".[notebook]"  # For notebook support
pip install -e ".[dev]"       # For development tools
```

## Configuration

1. Create a `.env` file with your API keys and configuration:

```
ALPHA_VANTAGE_API_KEY=your_api_key
GOOGLE_CLOUD_PROJECT=your_project_id
BIGQUERY_DATASET=trading_insights
```

2. Configure the tracked stocks in `config/tracked_stocks.json`:

```json
{
  "buy": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "ADBE", "ORCL", "ASML"],
  "short": ["BIDU", "NIO", "PINS", "SNAP", "COIN", "PLTR", "UBER", "LCID", "INTC", "XPEV"]
}
```

## Usage

### Data Ingestion

```bash
# Run data ingestion for all tracked stocks
python -m trading_ai.run ingest

# Force full data refresh
python -m trading_ai.run ingest --force-full
```

### Volume Analysis

```bash
# Run volume analysis for all tracked stocks
python -m trading_ai.run volume

# Run volume analysis for a specific stock
python -m trading_ai.run volume --ticker AAPL
```

### Sentiment Analysis

```bash
# Run sentiment analysis for all tracked stocks
python -m trading_ai.run sentiment

# Run sentiment analysis for a specific stock
python -m trading_ai.run sentiment --ticker AAPL
```

### Generate Master Summary

```bash
# Generate master summary for all tracked stocks
python -m trading_ai.run summary
```

### Paper Trading

```bash
# Execute a paper trade
python -m trading_ai.run trade --ticker AAPL --direction buy --entry-price 150.0 --capital-percentage 5.0 --stop-loss 145.0 --take-profit 160.0 --trigger-reason "Volume spike with positive price action" --notes "Strong support at 145"

# Close a paper trade
python -m trading_ai.run close --trade-id your_trade_id --exit-price 155.0 --exit-reason "Target reached" --notes "Closed at resistance"

# Get account summary
python -m trading_ai.run account
```

## Project Structure

```
trading_ai/
├── __init__.py
├── config.py
├── run.py
│
├── core/
│   ├── __init__.py
│   ├── alpha_client.py
│   ├── bigquery_io.py
│   ├── data_ingestion.py
│   ├── sentiment.py
│   └── volume_footprint.py
│
├── trading/
│   ├── __init__.py
│   ├── cost_calculator.py
│   ├── paper_trader.py
│   └── trade_logger.py
│
├── models/
│   ├── __init__.py
│   └── (coming soon)
│
├── dashboard/
│   ├── __init__.py
│   └── (coming soon)
│
└── discord/
    ├── __init__.py
    └── (coming soon)
```

## Google Colab Integration

The system is designed to work with Google Colab notebooks for interactive analysis. Notebooks will be provided for:

1. Master control notebook
2. Individual stock analysis notebooks (one per stock)

## Coming Soon

- Reinforcement Learning models for each ticker
- Discord bot for alerts and insights
- Interactive dashboard
- Automated notebook generation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
