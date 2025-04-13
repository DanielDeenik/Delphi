# Ticker Import Scripts

This directory contains scripts for importing data for individual tickers. Each script is designed to update the data for a specific Google Colab notebook.

## Overview

The ticker import scripts provide a convenient way to update the data for individual ticker notebooks. Each script:

1. Imports data for a specific ticker from Alpha Vantage to BigQuery
2. Provides options for controlling the import process
3. Logs the import process to a file in the `logs` directory

## Usage

### Windows

To import data for a specific ticker:

```batch
import_aapl.bat
```

To import data for all tickers (Master notebook):

```batch
import_master.bat
```

### Unix/Linux/macOS

To import data for a specific ticker:

```bash
./import_aapl.sh
```

To import data for all tickers (Master notebook):

```bash
./import_master.sh
```

## Options

Each script supports the following options:

### Individual Ticker Scripts

```
--force-full        Force a full data refresh
--days DAYS         Number of days of data to import (default: 365)
--repair-missing    Check and repair missing dates
--check-missing     Check for missing dates without importing
```

### Master Script

```
--force-full        Force a full data refresh
--days DAYS         Number of days of data to import (default: 90)
--repair-missing    Check and repair missing dates
--batch-size SIZE   Number of tickers to process in each batch (default: 5)
--max-workers N     Maximum number of worker threads (default: 3)
--report            Generate import status report
--retry-failed      Retry failed imports
```

## Examples

### Import 90 days of data for AAPL

```bash
./import_aapl.sh --days 90
```

### Force a full refresh for MSFT

```bash
./import_msft.sh --force-full
```

### Check for missing dates for GOOGL

```bash
./import_googl.sh --check-missing
```

### Import data for all tickers with 3 workers

```bash
./import_master.sh --max-workers 3
```

### Generate an import status report

```bash
./import_master.sh --report
```

## Integration with Google Colab

These scripts are designed to be used with the Google Colab notebooks in the Delphi Trading Intelligence System. Each script updates the data for a specific notebook:

- `import_master.py`: Updates data for the Master Summary notebook
- `import_aapl.py`: Updates data for the AAPL Analysis notebook
- `import_msft.py`: Updates data for the MSFT Analysis notebook
- etc.

## Generating Scripts

If you need to regenerate the scripts (e.g., after adding new tickers to the configuration), run:

```bash
# Windows
..\generate_ticker_scripts.bat

# Unix/Linux/macOS
../generate_ticker_scripts.sh
```

This will generate scripts for all tickers in the configuration.
