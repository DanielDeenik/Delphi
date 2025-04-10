# Stock Volume Analysis System

This system provides a framework for analyzing volume patterns and inefficiencies in stocks using Google Colab notebooks and BigQuery for data storage.

## System Overview

The system consists of:

1. **Individual Stock Analysis Notebooks**: One notebook per stock that analyzes volume patterns and inefficiencies.
2. **Master Summary Notebook**: A central dashboard that aggregates signals from all individual notebooks.
3. **BigQuery Tables**: For storing stock price data and analysis results.
4. **Scripts**: For setting up the system and generating notebooks.

## Setup Instructions

### 1. Configure Tracked Stocks

Edit the `config/tracked_stocks.json` file to specify the stocks you want to track:

```json
{
  "buy": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "ADBE", "ORCL", "ASML"],
  "short": ["BIDU", "NIO", "PINS", "SNAP", "COIN", "PLTR", "UBER", "LCID", "INTC", "XPEV"]
}
```

### 2. Set Up BigQuery Tables

Run the following script to set up the necessary BigQuery tables:

```bash
python -m src.scripts.setup_bigquery_tables
```

This will create:
- A dataset called `trading_insights`
- Price tables for each stock: `stock_{TICKER}_prices`
- Analysis tables for each stock: `stock_{TICKER}_analysis`
- A master summary table: `master_summary`

### 3. Import Stock Data

Run the following script to import stock data from Alpha Vantage to BigQuery:

```bash
python -m src.scripts.import_stock_data
```

### 4. Generate Notebooks

Run the following scripts to generate the individual stock analysis notebooks and the master summary notebook:

```bash
python -m src.scripts.generate_notebooks
python -m src.scripts.generate_master_notebook
```

### 5. Upload Notebooks to Google Colab

Upload the generated notebooks to Google Colab:
- Upload all notebooks from the `notebooks/individual` directory
- Upload the `notebooks/master_summary.ipynb` file

## Using the System

### Individual Stock Analysis Notebooks

Each individual stock analysis notebook:
1. Connects to BigQuery to fetch the latest data for the stock
2. Performs volume analysis to identify spikes and patterns
3. Generates trading signals based on volume patterns
4. Saves analysis results back to BigQuery
5. Provides visualizations of volume patterns and signals

### Master Summary Notebook

The master summary notebook:
1. Connects to BigQuery to fetch the latest analysis results for all stocks
2. Provides a dashboard view of all signals
3. Ranks stocks by signal strength
4. Visualizes volume patterns across all stocks
5. Provides links to individual stock notebooks for detailed analysis

## Maintenance

To keep the system up to date:

1. Run the data import script daily to update stock prices:
   ```bash
   python -m src.scripts.import_stock_data
   ```

2. Run each individual notebook to update analysis results

3. Run the master summary notebook to get an updated dashboard

## Customization

You can customize the system by:

1. Modifying the volume analysis logic in the individual notebook template
2. Adding additional analysis techniques to the notebooks
3. Customizing the dashboard visualizations in the master notebook
4. Adding more stocks to the tracked stocks configuration

## Troubleshooting

If you encounter issues:

1. Check that your Google Cloud credentials are properly set up
2. Verify that the BigQuery tables exist and have the correct schema
3. Check that the Alpha Vantage API key is valid and has sufficient quota
4. Ensure that the notebooks have the necessary permissions to access BigQuery

## Next Steps

Future enhancements could include:

1. Automating the notebook execution using Google Cloud Scheduler
2. Adding email or Telegram notifications for strong signals
3. Integrating with trading platforms for automated trading
4. Adding sentiment analysis from news and social media
5. Incorporating additional technical indicators for signal generation
