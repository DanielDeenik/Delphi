# Time Series Storage with SQLite

This document provides information on how to use SQLite for storing time series data from Alpha Vantage in the Delphi project.

## Overview

Delphi uses SQLite as a free, lightweight time series database for storing market data. SQLite is built into Python and requires no additional installation or server setup. The implementation also includes a migration path to Google BigQuery for when you need to scale up.

## Features

- **Zero Configuration**: SQLite is built into Python, so no additional setup is required
- **Portable**: The database is stored as a single file in the `data` directory
- **Reliable**: SQLite is known for its reliability and durability
- **Migration Path**: Easy migration to BigQuery when you need to scale

## Usage

The SQLite storage service is automatically used by the Alpha Vantage client. When you fetch data using the client, it will be stored in SQLite:

```python
from src.data.alpha_vantage_client import AlphaVantageClient

# Initialize the client
client = AlphaVantageClient()

# Fetch data (will be automatically stored in SQLite)
data = client.fetch_daily_adjusted('AAPL')
```

## Database Location

The SQLite database is stored in the `data/market_data.db` file in your project directory. This file is automatically created when you first use the storage service.

## Schema

The SQLite database uses the following schema:

```sql
CREATE TABLE market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TIMESTAMP NOT NULL,
    data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
)
```

- `symbol`: The trading symbol (e.g., 'AAPL')
- `date`: The timestamp for the data point
- `data`: The market data stored as JSON
- `created_at`: When the record was created

## Migrating to BigQuery

When you're ready to migrate to Google BigQuery for more scalable storage, you can use the migration script:

```bash
# Set up Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
export GOOGLE_CLOUD_PROJECT=your-project-id

# Run the migration script
python scripts/migrate_to_bigquery.py --symbols AAPL,MSFT,GOOG
```

## Querying Data Directly

If you need to query the SQLite database directly, you can use the `sqlite3` command-line tool or any SQLite client:

```bash
# Using the sqlite3 command-line tool
sqlite3 data/market_data.db

# Example query
sqlite> SELECT symbol, date, json_extract(data, '$.Close') as close_price 
        FROM market_data 
        WHERE symbol = 'AAPL' 
        ORDER BY date DESC 
        LIMIT 10;
```

## Backup and Maintenance

### Backing Up the Database

To back up the SQLite database, simply copy the `data/market_data.db` file:

```bash
cp data/market_data.db data/market_data_backup_$(date +%Y%m%d).db
```

### Optimizing the Database

To optimize the SQLite database after heavy use:

```bash
sqlite3 data/market_data.db "VACUUM;"
```

## Troubleshooting

### Database Locked

If you encounter a "database is locked" error, it means another process is using the database. Wait for the other process to finish or restart your application.

### Slow Queries

If queries are becoming slow, consider:

1. Adding additional indexes:
   ```sql
   CREATE INDEX idx_market_data_symbol ON market_data (symbol);
   ```

2. Running the VACUUM command to optimize the database:
   ```sql
   VACUUM;
   ```

3. If performance issues persist, it might be time to migrate to BigQuery.

## Additional Resources

- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [Python sqlite3 Module](https://docs.python.org/3/library/sqlite3.html)
- [Google BigQuery Documentation](https://cloud.google.com/bigquery/docs)
