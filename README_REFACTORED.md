# Delphi - Financial Analysis Platform (Refactored)

This is a refactored version of the Delphi financial analysis platform, with a focus on clean code, maintainability, and performance.

## Project Structure

The refactored codebase follows a clean, modular structure:

```
delphi/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── sources/
│   │   │   ├── __init__.py
│   │   │   ├── alpha_vantage.py
│   │   │   ├── yfinance.py
│   │   │   └── base.py
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   ├── bigquery.py
│   │   │   ├── sqlite.py
│   │   │   └── base.py
│   │   └── repository.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── volume/
│   │   │   ├── __init__.py
│   │   │   ├── analyzer.py
│   │   │   ├── strategies.py
│   │   │   └── patterns.py
│   │   ├── correlation/
│   │   │   ├── __init__.py
│   │   │   └── analyzer.py
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── lstm.py
│   │   │   ├── hmm.py
│   │   │   └── autoencoder.py
│   │   └── rag/
│   │       ├── __init__.py
│   │       ├── analyzer.py
│   │       └── memory.py
│   └── services/
│       ├── __init__.py
│       ├── scheduler.py
│       ├── trading.py
│       └── sentiment.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   └── parallel.py
├── visualization/
│   ├── __init__.py
│   ├── plots.py
│   └── notebooks.py
├── cli/
│   ├── __init__.py
│   ├── main.py
│   ├── import_command.py
│   └── analyze_command.py
└── api/
    ├── __init__.py
    ├── app.py
    └── routes.py
```

## Design Patterns

The refactored codebase uses several design patterns to improve maintainability and extensibility:

### Factory Pattern

Used for creating instances of data sources, storage services, and analyzers:

```python
class ModelFactory:
    @staticmethod
    def create_analyzer(analyzer_type: str, **kwargs) -> Analyzer:
        if analyzer_type == "volume":
            return VolumeAnalyzer(**kwargs)
        elif analyzer_type == "correlation":
            return CorrelationAnalyzer(**kwargs)
        else:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")
```

### Strategy Pattern

Used for analysis algorithms:

```python
class VolumeAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, data):
        pass

class SimpleVolumeStrategy(VolumeAnalysisStrategy):
    def analyze(self, data):
        # Simple volume analysis
        pass

class MLVolumeStrategy(VolumeAnalysisStrategy):
    def analyze(self, data):
        # ML-based volume analysis
        pass
```

### Repository Pattern

Used for data access:

```python
class MarketDataRepository:
    def __init__(self, data_source, storage_service):
        self.data_source = data_source
        self.storage_service = storage_service
    
    def get_stock_data(self, symbol, start_date, end_date):
        return self.storage_service.get_stock_prices(symbol, start_date, end_date)
```

### Dependency Injection

Used for better testability:

```python
class VolumeAnalyzer:
    def __init__(self, strategy):
        self.strategy = strategy
    
    def analyze(self, data):
        return self.strategy.analyze(data)
```

## Key Components

### Data Sources

- `DataSource`: Base class for all data sources
- `AlphaVantageClient`: Client for Alpha Vantage API
- `YFinanceClient`: Client for Yahoo Finance API

### Storage Services

- `StorageService`: Base class for all storage services
- `BigQueryStorage`: Storage service for Google BigQuery
- `SQLiteStorage`: Storage service for SQLite

### Models

- `Analyzer`: Base class for all analyzers
- `VolumeAnalyzer`: Analyzer for volume patterns
- `CorrelationAnalyzer`: Analyzer for correlations

### Services

- `MarketDataRepository`: Repository for accessing market data
- `SchedulerService`: Service for scheduling tasks
- `TradingService`: Service for trading operations

## Usage

### Command Line Interface

```bash
# Import data
python -m delphi.cli.main_refactored import --symbols AAPL MSFT GOOGL

# Analyze data
python -m delphi.cli.main_refactored analyze --symbols AAPL MSFT GOOGL --days 90
```

### Python API

```python
from delphi.core.data.sources import create_data_source
from delphi.core.data.storage import create_storage_service
from delphi.core.data.repository import MarketDataRepository
from delphi.core.models.volume.analyzer import VolumeAnalyzer
from delphi.core.models.volume.strategies import SimpleVolumeStrategy

# Create data source
data_source = create_data_source("alpha_vantage", api_key="YOUR_API_KEY")

# Create storage service
storage_service = create_storage_service("bigquery", project_id="YOUR_PROJECT_ID")

# Create repository
repository = MarketDataRepository(data_source, storage_service)

# Get data
df = repository.get_stock_data("AAPL")

# Create analyzer
strategy = SimpleVolumeStrategy()
analyzer = VolumeAnalyzer(strategy=strategy)

# Analyze data
results = analyzer.analyze(df)
```

## Configuration

Configuration is managed through the `ConfigManager` class in `delphi.utils.config`. The configuration is loaded from JSON files in the `config` directory:

- `system_config.json`: System-wide configuration
- `tracked_stocks.json`: Configuration for tracked stocks

Environment variables can also be used for configuration, and they take precedence over configuration files.

## Error Handling and Logging

The refactored codebase uses a consistent approach to error handling and logging:

```python
try:
    # Operation
    pass
except Exception as e:
    logger.error(f"Error in operation: {str(e)}")
    # Handle error appropriately
```

Logging is configured with appropriate levels:

```python
logger = logging.getLogger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message", exc_info=True)
```

## Testing

The refactored codebase is designed to be easily testable, with dependency injection and clear interfaces. Unit tests can be run with:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Use the existing design patterns and coding style
2. Add unit tests for new functionality
3. Update documentation as needed
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
