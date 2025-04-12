# Delphi Unified Import System

This document explains the unified import system for the Delphi trading intelligence platform. The system provides a streamlined, efficient way to import time series data from Alpha Vantage to BigQuery.

## Overview

The unified import system leverages the existing `trading_ai` modules to provide a cohesive, robust data import solution. It integrates the following components:

- **Alpha Vantage Client**: Fetches data from Alpha Vantage API
- **BigQuery Storage**: Stores data in Google BigQuery
- **Data Validation**: Ensures data quality and consistency
- **Error Handling**: Provides robust error handling and retry logic
- **Missing Data Detection**: Identifies and repairs missing data

## Quick Start

To run the unified import system:

### Windows

```batch
run_unified_import.bat
```

### Unix/Linux/macOS

```bash
./run_unified_import.sh
```

This will import data for all tracked stocks with default settings.

## Usage

### Basic Import

To import data for all tracked stocks:

```bash
python unified_import.py
```

### Import a Specific Ticker

To import data for a specific ticker:

```bash
python unified_import.py --ticker AAPL
```

### Force Full Refresh

To force a full data refresh:

```bash
python unified_import.py --force-full
```

### Retry Failed Imports

To retry failed imports:

```bash
python unified_import.py --retry-failed
```

### Repair Missing Data

To check and repair missing data:

```bash
python unified_import.py --repair-missing
```

### Check for Missing Dates

To check for missing dates without importing:

```bash
python unified_import.py --check-missing
```

### Generate Import Report

To generate a report of import status:

```bash
python unified_import.py --report
```

## Advanced Options

### Batch Size

To set the batch size for processing tickers:

```bash
python unified_import.py --batch-size 5
```

### Days to Import

To set the number of days of data to import:

```bash
python unified_import.py --days 365
```

### Maximum Worker Threads

To set the maximum number of worker threads:

```bash
python unified_import.py --max-workers 3
```

### Import from File

To import tickers listed in a file (one per line):

```bash
python unified_import.py --tickers-file tickers.txt
```

## Integration with Delphi Platform

The unified import system integrates seamlessly with the Delphi platform:

- **Configuration**: Uses `trading_ai.config.config_manager` for configuration
- **Data Source**: Uses `trading_ai.core.alpha_client.AlphaVantageClient` for fetching data
- **Storage Service**: Uses `trading_ai.core.bigquery_io.BigQueryStorage` for storing data

## Architecture

The unified import system follows a clean, modular architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Unified Import System                            │
├─────────────────────────────────────────────────────────────────────┤
│                           Core Components                            │
├───────────────┬───────────────┬───────────────┬───────────────┬─────┤
│ ConfigManager │ AlphaVantage  │ BigQuery      │ UnifiedImporter│ CLI │
│ (Config)      │ (Data Source) │ (Storage)     │ (Orchestrator)│     │
└───────────────┴───────────────┴───────────────┴───────────────┴─────┘
```

## Troubleshooting

### API Rate Limits

If you encounter API rate limit errors, try the following:

1. Reduce the batch size: `--batch-size 3`
2. Increase the wait time between batches (built-in)

### Data Quality Issues

If you encounter data quality issues, try the following:

1. Check for missing dates: `--check-missing`
2. Repair missing data: `--repair-missing`
3. Force a full refresh: `--force-full`

### Authentication Issues

If you encounter authentication issues:

1. Ensure you have run `gcloud auth application-default login`
2. Check that your Alpha Vantage API key is set in the configuration
3. Verify that your Google Cloud project ID is set in the configuration

## Conclusion

The unified import system provides a streamlined, efficient way to import time series data from Alpha Vantage to BigQuery. It leverages the existing `trading_ai` modules to provide a cohesive, robust data import solution.
