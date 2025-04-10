# Time Series Storage with InfluxDB

This document provides information on how to set up and use InfluxDB for storing time series data from Alpha Vantage in the Delphi project.

## Overview

Delphi uses InfluxDB as a free, high-performance time series database for storing market data. InfluxDB is specifically designed for time series data and provides excellent performance for this use case. The implementation also includes a migration path to Google BigQuery for when you need to scale up.

## Setup

### Option 1: Using Docker Compose (Recommended)

The easiest way to set up InfluxDB is using Docker Compose:

```bash
# Start InfluxDB
docker-compose up -d influxdb

# Wait a few seconds for InfluxDB to initialize
sleep 5

# Run the setup script to configure environment variables
python scripts/setup_influxdb.py
```

### Option 2: Manual Setup

If you prefer to set up InfluxDB manually:

1. Install InfluxDB following the [official documentation](https://docs.influxdata.com/influxdb/v2.7/install/)
2. Start InfluxDB
3. Run the setup script:

```bash
python scripts/setup_influxdb.py --host <your-influxdb-host> --port <your-influxdb-port>
```

## Configuration

The setup script will create a `.env` file with the necessary configuration variables:

```
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your-token
INFLUXDB_ORG=delphi
INFLUXDB_BUCKET=market_data
INFLUXDB_RETENTION_PERIOD=30d
```

You can modify these values as needed.

## Usage

The InfluxDB storage service is automatically used by the Alpha Vantage client. When you fetch data using the client, it will be stored in InfluxDB:

```python
from src.data.alpha_vantage_client import AlphaVantageClient

# Initialize the client
client = AlphaVantageClient()

# Fetch data (will be automatically stored in InfluxDB)
data = client.fetch_daily_adjusted('AAPL')
```

## Migrating to BigQuery

When you're ready to migrate to Google BigQuery for more scalable storage, you can use the migration script:

```bash
# Set up Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
export GOOGLE_CLOUD_PROJECT=your-project-id

# Run the migration script
python scripts/migrate_to_bigquery.py --symbols AAPL,MSFT,GOOG
```

## Visualization

You can visualize your time series data using Grafana. To set up Grafana:

1. Uncomment the Grafana service in `docker-compose.yml`
2. Start the services:

```bash
docker-compose up -d
```

3. Access Grafana at http://localhost:3000 (default credentials: admin/adminpassword)
4. Add InfluxDB as a data source:
   - URL: http://influxdb:8086
   - Organization: delphi
   - Token: (from your .env file)
   - Default Bucket: market_data

## Troubleshooting

### Connection Issues

If you're having trouble connecting to InfluxDB, check:

1. Is InfluxDB running? `docker ps | grep influxdb`
2. Are the environment variables set correctly? Check your `.env` file
3. Can you access the InfluxDB UI at http://localhost:8086?

### Data Not Being Stored

If data isn't being stored correctly:

1. Check the logs for error messages: `docker logs delphi-influxdb`
2. Verify your token has write permissions
3. Make sure the bucket exists

## Additional Resources

- [InfluxDB Documentation](https://docs.influxdata.com/influxdb/v2.7/)
- [Flux Query Language](https://docs.influxdata.com/flux/v0.x/)
- [Google BigQuery Documentation](https://cloud.google.com/bigquery/docs)
