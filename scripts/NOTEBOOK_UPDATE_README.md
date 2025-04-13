# Notebook Update Scripts

This directory contains scripts for updating the Google Colab notebooks with fresh data.

## Overview

The notebook update scripts provide a convenient way to:

1. Import data for individual tickers or all tickers
2. Generate notebooks for all tickers
3. Upload notebooks to Google Colab

## Individual Ticker Scripts

The `ticker_imports` directory contains scripts for importing data for individual tickers:

- `import_aapl.py`: Import data for AAPL
- `import_msft.py`: Import data for MSFT
- `import_googl.py`: Import data for GOOGL
- etc.

Each script can be run with various options to control the import process. See the [README](ticker_imports/README.md) in the `ticker_imports` directory for more information.

## Master Import Script

The `ticker_imports/import_master.py` script imports data for all tickers. This is useful for updating the data for the Master Summary notebook.

## Notebook Generation Script

The `generate_notebooks.py` script generates notebooks for all tickers. This script:

1. Creates a notebook for each ticker based on a template
2. Creates a master summary notebook
3. Uploads the notebooks to Google Colab (if requested)

## All-in-One Update Script

The `update_all_notebooks.py` script performs all the necessary steps to update the notebooks:

1. Imports data for all tickers
2. Generates notebooks for all tickers
3. Uploads notebooks to Google Colab

This is the easiest way to update all notebooks with fresh data.

## Usage

### Windows

To update all notebooks with fresh data:

```batch
update_all_notebooks.bat
```

To import data for a specific ticker:

```batch
ticker_imports\import_aapl.bat
```

To import data for all tickers:

```batch
ticker_imports\import_master.bat
```

### Unix/Linux/macOS

To update all notebooks with fresh data:

```bash
./update_all_notebooks.sh
```

To import data for a specific ticker:

```bash
./ticker_imports/import_aapl.sh
```

To import data for all tickers:

```bash
./ticker_imports/import_master.sh
```

## Integration with Google Colab

These scripts are designed to work with the Google Colab notebooks in the Delphi Trading Intelligence System. The notebooks are embedded in the web dashboard and can be accessed at:

- Dashboard: http://localhost:3000
- Analysis Notebooks: http://localhost:3000/colab
- All Notebooks: http://localhost:3000/colab/all

## Troubleshooting

If you encounter issues with the scripts:

1. Check the logs in the `logs` directory
2. Make sure your Alpha Vantage API key is valid
3. Make sure your Google Cloud credentials are valid
4. Try running the scripts with the `--force-full` option to force a full data refresh
