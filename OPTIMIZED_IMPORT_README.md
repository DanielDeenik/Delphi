# Delphi Optimized Import System

This document explains the optimized import system for the Delphi platform. The system provides a unified, efficient way to import data from Alpha Vantage to BigQuery.

## Overview

The optimized import system uses the following components:

- **Alpha Vantage Client**: Fetches data from Alpha Vantage API
- **BigQuery Storage**: Stores data in Google BigQuery
- **Import Manager**: Orchestrates the import process with robust error handling and retry logic
- **Data Repair Tool**: Fixes inconsistencies in BigQuery data
- **Storage Optimizer**: Optimizes BigQuery storage for time series data

## Quick Start

To run the optimized import system:

```bash
python run_optimized_import.py
```

This will import data for all tracked stocks with default settings.

## Usage

### Basic Import

To import data for all tracked stocks:

```bash
python run_optimized_import.py
```

### Import a Specific Ticker

To import data for a specific ticker:

```bash
python run_optimized_import.py --ticker AAPL
```

### Force Full Refresh

To force a full data refresh:

```bash
python run_optimized_import.py --force-full
```

### Retry Failed Imports

To retry failed imports:

```bash
python run_optimized_import.py --retry-failed
```

### Data Repair

To repair data inconsistencies:

```bash
python run_optimized_import.py --repair
```

### Storage Optimization

To optimize BigQuery storage:

```bash
python run_optimized_import.py --optimize
```

## Advanced Options

### Batch Size

To set the batch size for processing tickers:

```bash
python run_optimized_import.py --batch-size 5
```

### Wait Time

To set the wait time between batches in seconds:

```bash
python run_optimized_import.py --wait-time 60
```

### Maximum Batches

To set the maximum number of batches to process:

```bash
python run_optimized_import.py --max-batches 3
```

## Integration with Delphi Platform

The optimized import system integrates with the Delphi platform using:

- **Configuration**: Uses `trading_ai.config.config_manager` for configuration
- **Data Source**: Uses `trading_ai.core.alpha_client.AlphaVantageClient` for fetching data
- **Storage Service**: Uses `trading_ai.core.bigquery_io.BigQueryStorage` for storing data

## Architecture

The optimized import system follows a clean, modular architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Optimized BigQuery Import System                 │
├─────────────────────────────────────────────────────────────────────┤
│                           Core Components                            │
├───────────────┬───────────────┬───────────────┬───────────────┬─────┤
│ ConfigManager │ DataSource    │ StorageService│ ImportManager │ CLI │
│ (Single)      │ (AlphaVantage)│ (BigQuery)    │ (Orchestrator)│     │
└───────────────┴───────────────┴───────────────┴───────────────┴─────┘
```

## Troubleshooting

### API Rate Limits

If you encounter API rate limit errors, try the following:

1. Reduce the batch size: `--batch-size 3`
2. Increase the wait time between batches: `--wait-time 60`

### Data Quality Issues

If you encounter data quality issues, try the following:

1. Run data repair: `--repair`
2. Force a full refresh: `--force-full`

### Storage Issues

If you encounter storage issues, try the following:

1. Optimize storage: `--optimize`
2. Compress historical data: `--compress-history`

## Conclusion

The optimized import system provides a unified, efficient way to import data from Alpha Vantage to BigQuery. It integrates with the Delphi platform to provide a seamless experience.
