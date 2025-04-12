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

### Automated Data Import

You have two options for keeping the data up to date:

1. **From the Master Notebook**:
   - The master summary notebook includes a section for data import
   - Simply run the `import_stock_data()` function to update all stocks

2. **Using a Scheduled Cron Job**:
   - Use the provided script for automated data import:
   ```bash
   python -m src.scripts.scheduled_data_import
   ```
   - Set up a cron job to run this script daily:
   ```
   # Example cron job (runs daily at 6:00 AM)
   0 6 * * * python /path/to/scheduled_data_import.py
   ```
   - Make sure to set the required environment variables:
     - `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID
     - `ALPHA_VANTAGE_API_KEY`: Your Alpha Vantage API key

### Dynamic Watchlist Management

You can update the tracked stocks in two ways:

1. **From the Master Notebook**:
   - Use the `update_tracked_stocks()` function
   - Example: `update_tracked_stocks(buy_stocks=['AAPL', 'MSFT', 'GOOGL'], short_stocks=['BIDU', 'NIO', 'SNAP'])`
   - This will:
     - Create a backup of the current configuration
     - Update the tracked stocks
     - Generate new notebooks for the updated stocks

2. **Using the Command-Line Script**:
   ```bash
   python -m src.scripts.update_tracked_stocks --buy AAPL MSFT GOOGL --short BIDU NIO SNAP
   ```

### Analysis Updates

1. Run each individual notebook to update analysis results
2. Run the master summary notebook to get an updated dashboard

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
