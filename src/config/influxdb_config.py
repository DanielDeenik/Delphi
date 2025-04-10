"""InfluxDB configuration settings."""
import os

INFLUXDB_CONFIG = {
    # InfluxDB connection settings
    'url': os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
    'token': os.getenv('INFLUXDB_TOKEN', ''),
    'org': os.getenv('INFLUXDB_ORG', 'delphi'),
    'bucket': os.getenv('INFLUXDB_BUCKET', 'market_data'),
    
    # Data retention settings
    'retention_period': os.getenv('INFLUXDB_RETENTION_PERIOD', '30d'),
    
    # Migration settings
    'auto_export_to_bigquery': os.getenv('AUTO_EXPORT_TO_BIGQUERY', 'false').lower() == 'true',
    'export_interval_days': int(os.getenv('EXPORT_INTERVAL_DAYS', '30')),
}
