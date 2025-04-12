# Delphi Integrated Import System

This document explains how to use the integrated import system for the Delphi platform. The system provides a unified interface for importing data to BigQuery, repairing data inconsistencies, and optimizing storage.

## Overview

The integrated import system uses the existing components of the Delphi platform:

- **Import Manager**: Handles data imports with robust error handling and retry logic
- **Data Repair Tool**: Fixes inconsistencies in BigQuery data
- **Storage Optimizer**: Optimizes BigQuery storage for time series data

## Quick Start

### Windows

```batch
run_delphi_import.bat
```

### Unix/Linux/Mac

```bash
./run_delphi_import.sh
```

## Usage

### Basic Import

To import data for all tracked stocks:

```bash
python scripts/run_integrated_import.py
```

### Import a Specific Ticker

To import data for a specific ticker:

```bash
python scripts/run_integrated_import.py --ticker AAPL
```

### Force Full Refresh

To force a full data refresh:

```bash
python scripts/run_integrated_import.py --force-full
```

### Retry Failed Imports

To retry failed imports:

```bash
python scripts/run_integrated_import.py --retry-failed
```

### Data Repair

To repair data inconsistencies:

```bash
python scripts/run_integrated_import.py --repair
```

### Check Data Quality

To check data quality without repairing:

```bash
python scripts/run_integrated_import.py --check-quality
```

### Storage Optimization

To optimize BigQuery storage:

```bash
python scripts/run_integrated_import.py --optimize
```

### Optimize Table Schemas

To optimize table schemas:

```bash
python scripts/run_integrated_import.py --optimize-schema
```

### Compress Historical Data

To compress historical data:

```bash
python scripts/run_integrated_import.py --compress-history
```

### Generate Import Report

To generate an import report:

```bash
python scripts/run_integrated_import.py --report
```

## Advanced Options

### Batch Size

To set the batch size for processing tickers:

```bash
python scripts/run_integrated_import.py --batch-size 5
```

### Retries

To set the maximum number of retries for failed imports:

```bash
python scripts/run_integrated_import.py --retries 5
```

### Retry Delay

To set the delay between retries in seconds:

```bash
python scripts/run_integrated_import.py --retry-delay 10
```

### Compress Days

To set the number of days for historical data compression:

```bash
python scripts/run_integrated_import.py --compress-history --compress-days 180
```

## Combining Options

You can combine multiple options to customize the import process:

```bash
python scripts/run_integrated_import.py --ticker AAPL --force-full --repair --optimize-schema
```

## Troubleshooting

### API Rate Limits

If you encounter API rate limit errors, try the following:

1. Reduce the batch size: `--batch-size 3`
2. Increase the retry delay: `--retry-delay 10`

### Data Quality Issues

If you encounter data quality issues, try the following:

1. Check data quality: `--check-quality`
2. Repair data: `--repair`
3. Force a full refresh: `--force-full`

### Storage Issues

If you encounter storage issues, try the following:

1. Optimize table schemas: `--optimize-schema`
2. Compress historical data: `--compress-history`

## Integration with Existing Codebase

The integrated import system uses the following components from the existing codebase:

- **Configuration**: Uses `trading_ai.config.config_manager` for configuration
- **Data Source**: Uses `trading_ai.core.alpha_client.AlphaVantageClient` for fetching data
- **Storage Service**: Uses `trading_ai.core.bigquery_io.BigQueryStorage` for storing data
- **Volume Analysis**: Uses `trading_ai.core.volume_footprint.VolumeAnalyzer` for analyzing volume

## Conclusion

The integrated import system provides a unified interface for importing data to BigQuery, repairing data inconsistencies, and optimizing storage. It integrates with the existing codebase to provide a seamless experience.
