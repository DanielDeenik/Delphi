# Delphi - Cloud-Native Trading Intelligence System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Delphi is a cloud-native, AI-powered trading intelligence system that leverages Alpha Vantage data, BigQuery, and machine learning models to provide actionable trading insights.

## Features

- **Robust Data Import**: Import time series data from Alpha Vantage with error handling and retry logic
- **Cloud Storage**: Store and query data efficiently in Google BigQuery
- **Volume Analysis**: Detect volume inefficiencies and trading opportunities
- **Time Series Analysis**: Apply statistical and machine learning models to financial time series
- **Interactive Dashboards**: Analyze data through Google Colab notebooks
- **Command-line Interface**: Manage data import and analysis through a simple CLI
- **Scalable Architecture**: Designed for cloud deployment with Docker and GKE

## Installation

### Prerequisites

- Python 3.8 or higher
- Google Cloud account with BigQuery enabled
- Alpha Vantage API key

### Quick Install

```bash
# Clone the repository
git clone https://github.com/delphitrading/delphi.git
cd delphi

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and dependencies
pip install -e .

# Install optional dependencies
pip install -e ".[docs]"  # For documentation
pip install -e ".[dev]"   # For development
pip install -e ".[yfinance]"  # For Yahoo Finance support
```

### Docker Installation

```bash
# Build the Docker image
docker build -t delphi .

# Run the container
docker run -it --rm \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/config/credentials.json \
  delphi
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# Google Cloud
GOOGLE_CLOUD_PROJECT=your-project-id
BIGQUERY_DATASET=trading_insights
BIGQUERY_LOCATION=US

# Alpha Vantage
ALPHA_VANTAGE_API_KEY=your-api-key
```

### Google Cloud Authentication

```bash
# Authenticate with Google Cloud
gcloud auth application-default login

# Set the project ID
gcloud config set project your-project-id
```

## Usage

### Time Series Import

Import time series data from Alpha Vantage to BigQuery:

```bash
# Import data for all configured tickers
python run_time_series_import.py

# Import data for specific tickers
python run_time_series_import.py --ticker AAPL
python run_time_series_import.py --tickers-file tickers.txt

# Force a full data refresh
python run_time_series_import.py --force-full

# Check for missing dates
python run_time_series_import.py --check-missing

# Repair missing data
python run_time_series_import.py --repair-missing

# Generate import report
python run_time_series_import.py --report
```

Or use the batch scripts:

```bash
# Windows
run_import.bat --ticker AAPL

# Unix/Linux/macOS
./run_import.sh --ticker AAPL
```

### Command-line Interface

The package provides command-line tools for common tasks:

```bash
# Import data
delphi-import --tickers AAPL MSFT GOOGL --days 365

# Analyze data
delphi-analyze --ticker AAPL --days 90

# Launch dashboard on port 3000
delphi-dashboard --port 3000
```

### Launching the Application

You can launch the application using the cross-platform launcher:

```bash
# Cross-platform Python launcher (recommended)
python launch_delphi.py

# Simple wrapper scripts
# Windows
launch.bat

# Unix/Linux/macOS
./launch.sh
```

Alternatively, you can use the platform-specific scripts:

```bash
# Windows
launch_delphi.bat

# Unix/Linux/macOS
./launch_delphi.sh
```

The launcher will:
1. Set up the necessary environment
2. Start the Flask application on port 3000
3. Open the browser to the multi-tab Colab view

You can access the application at:

- Dashboard: http://localhost:3000
- Analysis Notebooks: http://localhost:3000/colab
- All Notebooks (21 tabs): http://localhost:3000/colab/all
- API Documentation: http://localhost:3000/api/docs

For Windows users, you can create a desktop shortcut by running:

```batch
create_desktop_shortcut.bat
```

For more information, see [LAUNCHER_README.md](LAUNCHER_README.md).

### Python API

```python
from trading_ai.config import config_manager
from trading_ai.core.alpha_client import AlphaVantageClient
from trading_ai.core.bigquery_io import BigQueryStorage
from trading_ai.analysis.volume_analyzer import VolumeAnalyzer

# Initialize clients
alpha_client = AlphaVantageClient()
bigquery_storage = BigQueryStorage()

# Import data
ticker = "AAPL"
price_df = alpha_client.fetch_daily(ticker, outputsize="full")
bigquery_storage.store_stock_prices(ticker, price_df)

# Analyze data
volume_analyzer = VolumeAnalyzer()
results = volume_analyzer.analyze_ticker(ticker, days=90)

# Store analysis results
bigquery_storage.store_volume_analysis(ticker, results)
```

### Jupyter Notebooks

The system includes Google Colab notebooks for interactive analysis:

1. **Individual Stock Analysis**: One notebook per stock for detailed volume analysis
2. **Master Summary**: Central watchlist with key metrics for all tracked stocks

## Project Structure

```
.
├── trading_ai/                  # Main package
│   ├── __init__.py             # Package initialization
│   ├── config/                 # Configuration management
│   │   ├── __init__.py
│   │   └── config_manager.py   # Configuration loading and validation
│   ├── core/                   # Core functionality
│   │   ├── __init__.py
│   │   ├── alpha_client.py     # Alpha Vantage API client
│   │   └── bigquery_io.py      # BigQuery storage interface
│   ├── analysis/               # Analysis modules
│   │   ├── __init__.py
│   │   ├── volume_analyzer.py  # Volume analysis
│   │   └── time_series.py      # Time series analysis
│   ├── visualization/          # Visualization tools
│   │   ├── __init__.py
│   │   ├── volume_plots.py     # Volume visualization
│   │   └── dashboard.py        # Dashboard generation
│   └── cli/                    # Command-line interface
│       ├── __init__.py
│       ├── import_cli.py       # Import command
│       └── analyze_cli.py      # Analysis command
├── notebooks/                  # Jupyter notebooks
│   ├── master_summary.ipynb    # Master summary notebook
│   └── stocks/                 # Individual stock notebooks
│       ├── AAPL.ipynb
│       └── ...
├── scripts/                    # Utility scripts
│   ├── run_import.bat          # Windows import script
│   └── run_import.sh           # Unix import script
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_alpha_client.py
│   └── test_bigquery_io.py
├── docs/                       # Documentation
│   ├── index.md
│   └── api/
├── run_time_series_import.py   # Main import script
├── setup.py                    # Package setup
├── requirements.txt            # Dependencies
├── .env.example                # Example environment variables
├── .gitignore                  # Git ignore file
├── LICENSE                     # License file
└── README.md                   # This file
```

## Architecture

The system follows a modular, layered architecture:

1. **Data Layer**: Alpha Vantage API client and BigQuery storage
2. **Analysis Layer**: Volume analysis and time series processing
3. **Visualization Layer**: Plotting and dashboard generation
4. **Interface Layer**: CLI and Python API

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Alpha Vantage](https://www.alphavantage.co/) for providing financial data APIs
- [Google BigQuery](https://cloud.google.com/bigquery) for scalable data storage and analysis
- [Pandas](https://pandas.pydata.org/) for data manipulation and analysis