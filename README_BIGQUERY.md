# Oracle of Delphi - BigQuery Integration

This document provides instructions for importing VIX, SPY, and PLTR data from Alpha Vantage to BigQuery and analyzing it.

## Prerequisites

1. **Alpha Vantage API Key**: Get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. **Google Cloud Account**: You need a Google Cloud account with BigQuery enabled.
3. **Google Cloud Project**: Create a project in Google Cloud Console.
4. **Google Cloud SDK**: Install and configure the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
5. **Python Libraries**: Install the required Python libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn google-cloud-bigquery requests
   ```

## Setup

### 1. Set Up Google Cloud Authentication

1. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   ```

2. Set your project ID:
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

3. Set up application default credentials:
   ```bash
   gcloud auth application-default login
   ```

## Step 2: Import Data to BigQuery

1. Open `import_to_bigquery.py` and update the following variables:
   - `PROJECT_ID`: Your Google Cloud project ID
   - `DATASET_ID`: Your BigQuery dataset ID (default: "market_data")
   - `TABLE_ID`: Your BigQuery table ID (default: "time_series")

2. Run the import script:
   ```bash
   python import_to_bigquery.py
   ```

3. When prompted, enter your Alpha Vantage API key.

4. The script will:
   - Set up the BigQuery dataset and table if they don't exist
   - Fetch data for VIX, SPY, and PLTR from Alpha Vantage
   - Process the data
   - Load it into BigQuery

## Step 3: Analyze the Data

1. Open `analyze_bigquery_data.py` and update the following variables:
   - `PROJECT_ID`: Your Google Cloud project ID
   - `DATASET_ID`: Your BigQuery dataset ID (default: "market_data")
   - `TABLE_ID`: Your BigQuery table ID (default: "time_series")

2. Run the analysis script:
   ```bash
   python analyze_bigquery_data.py
   ```

3. The script will:
   - Fetch data from BigQuery
   - Analyze volume patterns
   - Analyze correlations between symbols
   - Generate plots in the "plots" directory
   - Print analysis summaries

## Understanding the Analysis

### Volume Analysis

The volume analysis includes:
- Volume spikes (volume > 2x 20-day average)
- Volume drops (volume < 0.5x 20-day average)
- Price-volume divergences:
  - Price up, volume down
  - Price down, volume up
- On-Balance Volume (OBV) analysis:
  - OBV bearish divergence (price up, OBV down)
  - OBV bullish divergence (price down, OBV up)

### Correlation Analysis

The correlation analysis includes:
- Price correlations
- Return correlations
- Volume correlations
- Key correlations:
  - VIX-SPY correlation
  - VIX-PLTR correlation
  - SPY-PLTR correlation

## Plots

The analysis generates the following plots in the "plots" directory:
- Volume plots for each symbol
- OBV plots for each symbol
- Return correlation heatmap
- Normalized price comparison

## Troubleshooting

### Authentication Issues

If you encounter authentication issues:
1. Verify that you've run `gcloud auth application-default login`
2. Check that your Google Cloud project has BigQuery API enabled
3. Ensure you have the necessary permissions to create datasets and tables

### Alpha Vantage API Issues

If you encounter issues with the Alpha Vantage API:
1. Verify that your API key is correct
2. Check the API rate limits (5 calls per minute for free tier)
3. Try running the script again after waiting a few minutes

### BigQuery Issues

If you encounter issues with BigQuery:
1. Verify that your project ID, dataset ID, and table ID are correct
2. Check that you have the necessary permissions
3. Ensure that the BigQuery API is enabled for your project

## Customization

You can customize the scripts to:
- Import different symbols by modifying the `SYMBOLS_DICT` variable
- Change the date range by modifying the `from_date` and `to_date` parameters
- Adjust the analysis parameters in the `analyze_volume_patterns` and `analyze_correlations` functions

## Next Steps

1. **Create a Google Colab Notebook**: For interactive analysis of the data
2. **Set Up Scheduled Imports**: Use Cloud Scheduler to automate data imports
3. **Implement Advanced Analytics**: Use BigQuery ML for advanced analytics
4. **Create Dashboards**: Use Data Studio to visualize the data
