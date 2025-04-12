# Delphi Robust Data Management System

This document explains the robust data management system for the Delphi platform. The system provides comprehensive tools for importing, validating, repairing, and reconciling time series data in BigQuery.

## Overview

The robust data management system consists of the following components:

1. **Import Manager**: Handles data imports with robust error handling and retry logic
2. **Data Repair Tool**: Fixes inconsistencies in BigQuery data
3. **Data Reconciliation Tool**: Checks for missing dates and validates data quality
4. **Storage Optimizer**: Optimizes BigQuery storage for time series data

## Key Features

### Robust Error Handling

- Comprehensive error handling with detailed logging
- Automatic retries for failed imports with configurable parameters
- Status tracking for each ticker

### Data Validation

- Validates data before import to ensure quality
- Checks for required columns
- Validates price ranges (high >= low, etc.)
- Checks for null values

### Data Repair

- Fixes inconsistencies in BigQuery data
- Fills missing values using interpolation
- Removes duplicate dates
- Corrects data types

### Data Reconciliation

- Checks for missing dates in time series data
- Compares with trading calendars to identify missing trading days
- Validates data quality with detailed metrics
- Generates reconciliation reports
- Identifies tickers needing reimport

### Reimport Capabilities

- Automatically reimports data for tickers with missing dates
- Handles partial data updates
- Preserves existing data when appropriate

### Storage Optimization

- Optimizes BigQuery storage for time series data
- Partitions tables by date for faster queries
- Clusters tables by relevant fields for faster queries
- Compresses historical data to reduce storage costs

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

### Data Reconciliation

To reconcile data and check for missing dates:

```bash
python run_optimized_import.py --reconcile
```

### Reimport Missing Data

To reimport data for tickers with missing dates:

```bash
python run_optimized_import.py --reconcile --reimport-missing
```

### Storage Optimization

To optimize BigQuery storage:

```bash
python run_optimized_import.py --optimize
```

## Advanced Usage

### Combining Options

You can combine multiple options to customize the data management process:

```bash
python run_optimized_import.py --ticker AAPL --force-full --repair --reconcile --reimport-missing
```

### Batch Processing

To set the batch size for processing tickers:

```bash
python run_optimized_import.py --batch-size 5
```

### Retry Configuration

To configure retry behavior:

```bash
python run_optimized_import.py --retries 5 --retry-delay 10
```

## Data Reconciliation Reports

The data reconciliation tool generates detailed reports that include:

- Missing dates for each ticker
- Data quality metrics
- Tickers needing reimport

Reports are saved in the `reports` directory with a timestamp in the filename.

## Troubleshooting

### Missing Data

If you encounter missing data issues:

1. Run data reconciliation: `--reconcile`
2. Reimport missing data: `--reimport-missing`
3. Check reconciliation reports in the `reports` directory

### Data Quality Issues

If you encounter data quality issues:

1. Run data repair: `--repair`
2. Force a full refresh: `--force-full`
3. Check data quality metrics in reconciliation reports

### Storage Issues

If you encounter storage issues:

1. Optimize storage: `--optimize`
2. Compress historical data: `--compress-history`

## Conclusion

The robust data management system provides comprehensive tools for importing, validating, repairing, and reconciling time series data in BigQuery. It ensures data completeness, consistency, and quality for reliable financial analysis.
