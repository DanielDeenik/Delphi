# Delphi - Financial Analysis Platform

Delphi is a comprehensive platform for financial data analysis, with a focus on volume analysis and correlation detection.

## Features

- Import data from Alpha Vantage and Yahoo Finance
- Store data in Google BigQuery
- Analyze volume patterns and correlations
- Generate visualizations
- Command-line interface for data import and analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/delphi.git
cd delphi

# Install the package
pip install -e .

# Install optional dependencies
pip install -e ".[yfinance]"
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file in the root directory with the following variables:

```
# Google Cloud
GOOGLE_CLOUD_PROJECT=your-project-id
BIGQUERY_DATASET=market_data
BIGQUERY_TABLE=time_series
BIGQUERY_LOCATION=US

# Alpha Vantage
ALPHA_VANTAGE_API_KEY=your-api-key
```

## Usage

### Command-line Interface

Import data:

```bash
# Import data from Yahoo Finance
delphi-import --source yfinance --symbols SPY PLTR ^VIX

# Import data from Alpha Vantage
delphi-import --source alpha_vantage --symbols SPY PLTR ^VIX
```

Analyze data:

```bash
# Analyze data for the last 90 days
delphi-analyze --days 90 --symbols SPY PLTR ^VIX
```

### Python API

```python
from delphi.data import YFinanceClient, BigQueryImporter
from delphi.models import VolumeAnalyzer, CorrelationAnalyzer
from delphi.utils.config import load_env

# Load environment variables
load_env()

# Create data source
data_source = YFinanceClient()

# Create importer
importer = BigQueryImporter()

# Import data
results = importer.import_data(
    data_source=data_source,
    symbols=["SPY", "PLTR", "^VIX"],
    period="2y"
)

# Create analyzers
volume_analyzer = VolumeAnalyzer()
correlation_analyzer = CorrelationAnalyzer()

# Analyze data
volume_results = volume_analyzer.analyze(data["SPY"])
correlation_results = correlation_analyzer.analyze(data)
```

## Project Structure

```
delphi/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── data_source.py
│   ├── alpha_vantage_client.py
│   ├── yfinance_client.py
│   └── bigquery_importer.py
├── models/
│   ├── __init__.py
│   ├── volume_analyzer.py
│   ├── correlation_analyzer.py
│   └── unified_analyzer.py
├── services/
│   ├── __init__.py
│   ├── storage_service.py
│   ├── bigquery_service.py
│   └── service_factory.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   └── parallel.py
├── visualization/
│   ├── __init__.py
│   ├── volume_plots.py
│   └── correlation_plots.py
└── cli/
    ├── __init__.py
    ├── import_command.py
    └── analyze_command.py
```

## License

MIT