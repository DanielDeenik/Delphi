# Delphi Trading Intelligence - Notebooks

This document explains how to use the notebook generation and management features of the Delphi Trading Intelligence System.

## Overview

The Delphi Trading Intelligence System includes a notebook generation and management system that allows you to:

1. Generate Google Colab notebooks for volume inefficiency analysis
2. Upload notebooks to Google Colab
3. Embed notebooks in the web dashboard
4. Open notebooks directly in your browser

## Notebook Types

The system generates two types of notebooks:

1. **Master Summary Notebook**: A summary of volume inefficiency analysis for all tracked stocks
2. **Individual Stock Notebooks**: Detailed analysis for each tracked stock

## Usage

### Command-Line Interface

The system provides a command-line interface for generating and managing notebooks:

```bash
# Generate notebooks for all tracked tickers
python -m trading_ai.cli.notebook_cli --generate

# Generate notebooks for specific tickers
python -m trading_ai.cli.notebook_cli --generate --tickers AAPL MSFT GOOGL

# Generate and upload notebooks
python -m trading_ai.cli.notebook_cli --generate --upload

# Open the master notebook in your browser
python -m trading_ai.cli.notebook_cli --open

# Open a specific ticker notebook in your browser
python -m trading_ai.cli.notebook_cli --open --ticker AAPL
```

### Web Dashboard

The web dashboard includes a dedicated page for accessing notebooks:

1. Start the dashboard:
   ```bash
   python -m trading_ai.cli.dashboard_cli
   ```

2. Open your browser and navigate to `http://localhost:6000/colab`

3. Select a notebook from the list to view it

4. To view all notebooks in tabs, click on "View All Notebooks" at the bottom of the list or navigate to `http://localhost:6000/colab/all`

## Notebook Templates

The system uses templates to generate notebooks:

- `notebooks/master_summary_template.ipynb`: Template for the master summary notebook
- `notebooks/volume_analysis_template.ipynb`: Template for individual stock notebooks

You can customize these templates to add your own analysis or visualizations.

## Configuration

Notebook URLs are stored in a configuration file:

- `config/notebook_urls.json`: Maps tickers to Google Colab URLs

This file is automatically created and updated when you generate and upload notebooks.

## Batch Scripts

For convenience, the system includes batch scripts for common tasks:

- `scripts/generate_notebooks.bat` (Windows) or `scripts/generate_notebooks.sh` (Unix): Generate and upload notebooks
- `run_app_with_notebooks.bat` (Windows) or `run_app_with_notebooks.sh` (Unix): Launch the dashboard and open all notebooks in tabs

## Integration with Google Colab

The notebooks are designed to work with Google Colab and include:

1. Authentication with Google Cloud
2. Connection to BigQuery for data access
3. Interactive visualizations with Plotly
4. Volume inefficiency analysis

## Troubleshooting

If you encounter issues with notebook generation or access:

1. Check that the `notebooks` directory exists and is writable
2. Ensure that the `config` directory exists and is writable
3. Verify that you have the necessary permissions to access Google Colab and BigQuery
4. Check the logs for error messages

For more information, refer to the main README.md file.
