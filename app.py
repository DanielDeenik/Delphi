#!/usr/bin/env python3
"""
Delphi Trading Intelligence Dashboard

This is the main entry point for the Flask dashboard application with embedded Google Colab.
"""
import json
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from flask import Flask, jsonify, request, Response
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Import from trading_ai module
from trading_ai.config import config_manager
from trading_ai.config.notebook_config import notebook_config
from trading_ai.core.alpha_client import AlphaVantageClient
from trading_ai.core.bigquery_io import BigQueryStorage

# Initialize Flask app
app = Flask(__name__,
           static_folder='static',
           template_folder='templates')

# Initialize clients
alpha_client = AlphaVantageClient()
bigquery_storage = BigQueryStorage()

# Create directories for templates and static files
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Custom CSS for the application
custom_css = """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #424242;
    }
    .positive {
        color: #4CAF50;
    }
    .negative {
        color: #F44336;
    }
    .colab-wrapper {
        width: 100%;
        height: 800px;
        border: none;
        margin-top: 20px;
    }
</style>
"""

def load_import_status():
    """Load import status from file."""
    status_file = Path("status/import_status.json")
    if status_file.exists():
        try:
            with open(status_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading status file: {str(e)}")

    return {}

def get_ticker_list():
    """Get list of available tickers."""
    # First try to get from import status
    import_status = load_import_status()
    if import_status:
        return sorted([ticker for ticker in import_status.keys()])

    # Fallback to config
    return config_manager.get_all_tickers()

def load_ticker_data(ticker, days=90):
    """Load ticker data from BigQuery."""
    try:
        # Get data from BigQuery
        df = bigquery_storage.get_stock_prices(ticker, days=days)

        if df.empty:
            logger.error(f"No data found for {ticker} in BigQuery")
            return None

        # Sort by date
        df = df.sort_values('date')

        return df
    except Exception as e:
        logger.error(f"Error loading data for {ticker}: {str(e)}")
        return None

def calculate_metrics(df):
    """Calculate key metrics for a ticker."""
    if df is None or df.empty:
        return {}

    # Get latest price
    latest_price = df['close'].iloc[-1]

    # Calculate daily change
    daily_change = df['close'].iloc[-1] - df['close'].iloc[-2]
    daily_change_pct = (daily_change / df['close'].iloc[-2]) * 100

    # Calculate 30-day change
    if len(df) >= 30:
        price_30d_ago = df['close'].iloc[-30]
        change_30d = latest_price - price_30d_ago
        change_30d_pct = (change_30d / price_30d_ago) * 100
    else:
        change_30d = None
        change_30d_pct = None

    # Calculate average volume
    avg_volume = df['volume'].mean()

    # Calculate volume ratio (latest volume / average volume)
    volume_ratio = df['volume'].iloc[-1] / avg_volume

    # Calculate volatility (standard deviation of daily returns)
    daily_returns = df['close'].pct_change().dropna()
    volatility = daily_returns.std() * 100

    return {
        'latest_price': latest_price,
        'daily_change': daily_change,
        'daily_change_pct': daily_change_pct,
        'change_30d': change_30d,
        'change_30d_pct': change_30d_pct,
        'avg_volume': avg_volume,
        'latest_volume': df['volume'].iloc[-1],
        'volume_ratio': volume_ratio,
        'volatility': volatility
    }

def plot_price_chart(df, ticker):
    """Create a price chart with volume."""
    if df is None or df.empty:
        return None

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['close'],
            name='Close Price',
            line=dict(color='#1E88E5', width=2)
        )
    )

    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='Volume',
            marker=dict(color='rgba(30, 136, 229, 0.3)'),
            opacity=0.5,
            yaxis='y2'
        )
    )

    # Set layout
    fig.update_layout(
        title=f'{ticker} Price and Volume',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price', side='left', showgrid=False),
        yaxis2=dict(
            title='Volume',
            side='right',
            overlaying='y',
            showgrid=False
        ),
        legend=dict(x=0, y=1.1, orientation='h'),
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='x unified'
    )

    return fig

def plot_volume_analysis(df, ticker):
    """Create a volume analysis chart."""
    if df is None or df.empty:
        return None

    # Calculate volume moving average
    df['volume_ma'] = df['volume'].rolling(window=20).mean()

    # Calculate volume ratio
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Create figure
    fig = go.Figure()

    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='Volume',
            marker=dict(color='rgba(30, 136, 229, 0.5)')
        )
    )

    # Add volume moving average
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['volume_ma'],
            name='20-day MA',
            line=dict(color='#FF9800', width=2)
        )
    )

    # Set layout
    fig.update_layout(
        title=f'{ticker} Volume Analysis',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Volume'),
        legend=dict(x=0, y=1.1, orientation='h'),
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='x unified'
    )

    return fig

# Create template for index page
index_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Delphi Trading Intelligence</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="/static/styles.css">
    {custom_css}
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">
                <img src="https://img.icons8.com/color/48/000000/oracle-of-delphi.png" width="30" height="30" class="d-inline-block align-top" alt="">
                Delphi Trading Intelligence
            </a>
            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ml-auto">
                    <li class="nav-item active">
                        <a class="nav-link" href="/">Dashboard</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/colab">Analysis Notebooks</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/api/docs">API</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <h1 class="main-header">Delphi Trading Intelligence</h1>

        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Select Ticker</h5>
                        <form action="/" method="GET">
                            <div class="form-group">
                                <select name="ticker" class="form-control" onchange="this.form.submit()">
                                    {ticker_options}
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Time Range</label>
                                <div class="form-check">
                                    <input class="form-check-input" type="radio" name="time_range" value="30" {checked_30}>
                                    <label class="form-check-label">30 Days</label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input" type="radio" name="time_range" value="90" {checked_90}>
                                    <label class="form-check-label">90 Days</label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input" type="radio" name="time_range" value="365" {checked_365}>
                                    <label class="form-check-label">1 Year</label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input" type="radio" name="time_range" value="1825" {checked_1825}>
                                    <label class="form-check-label">5 Years</label>
                                </div>
                                <button type="submit" class="btn btn-primary btn-sm mt-2">Update</button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
            <div class="col-md-9">
                {content}
            </div>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</body>
</html>
"""

# Create template for Colab page
colab_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Delphi Trading Intelligence - Analysis Notebooks</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="/static/styles.css">
    {custom_css}
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">
                <img src="https://img.icons8.com/color/48/000000/oracle-of-delphi.png" width="30" height="30" class="d-inline-block align-top" alt="">
                Delphi Trading Intelligence
            </a>
            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ml-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Dashboard</a>
                    </li>
                    <li class="nav-item active">
                        <a class="nav-link" href="/colab">Analysis Notebooks</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/api/docs">API</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <h1 class="main-header">Analysis Notebooks</h1>

        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Select Notebook</h5>
                        <div class="list-group">
                            <a href="/colab/master" class="list-group-item list-group-item-action {master_active}">Master Summary</a>
                            {ticker_links}
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-9">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">{notebook_title}</h5>
                        <iframe class="colab-wrapper" src="{colab_url}" frameborder="0" allowfullscreen></iframe>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
"""

# Save templates to files
with open("templates/index.html", "w") as f:
    f.write(index_template)

with open("templates/colab.html", "w") as f:
    f.write(colab_template)

# Flask routes
@app.route('/')
def index():
    """Main dashboard page."""
    # Get query parameters
    ticker = request.args.get('ticker', '')
    time_range = request.args.get('time_range', '90')

    # Convert time range to days
    days = int(time_range)

    # Get available tickers
    tickers = get_ticker_list()

    # If no ticker is selected, use the first one
    if not ticker and tickers:
        ticker = tickers[0]

    # Create ticker options HTML
    ticker_options = ''
    for t in tickers:
        selected = 'selected' if t == ticker else ''
        ticker_options += f'<option value="{t}" {selected}>{t}</option>'

    # Set checked status for time range radio buttons
    checked_30 = 'checked' if time_range == '30' else ''
    checked_90 = 'checked' if time_range == '90' else ''
    checked_365 = 'checked' if time_range == '365' else ''
    checked_1825 = 'checked' if time_range == '1825' else ''

    # Load data
    content = ''
    if ticker:
        df = load_ticker_data(ticker, days=days)

        if df is not None and not df.empty:
            # Calculate metrics
            metrics = calculate_metrics(df)

            # Create metrics HTML
            metrics_html = f'''
            <div class="row">
                <div class="col-md-3">
                    <div class="card">
                        <div class="card-body">
                            <div class="metric-label">Latest Price</div>
                            <div class="metric-value">${metrics["latest_price"]:.2f}</div>
                            <div class="{"positive" if metrics["daily_change"] >= 0 else "negative"}">
                                {"+" if metrics["daily_change"] >= 0 else ""}{metrics["daily_change"]:.2f}
                                ({"+" if metrics["daily_change"] >= 0 else ""}{metrics["daily_change_pct"]:.2f}%)
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card">
                        <div class="card-body">
                            <div class="metric-label">30-Day Change</div>
                            {f'<div class="metric-value {"positive" if metrics["change_30d"] >= 0 else "negative"}">{"+" if metrics["change_30d"] >= 0 else ""}{metrics["change_30d_pct"]:.2f}%</div><div class="{"positive" if metrics["change_30d"] >= 0 else "negative"}">{"+" if metrics["change_30d"] >= 0 else ""}{metrics["change_30d"]:.2f}</div>' if metrics["change_30d"] is not None else '<div class="metric-value">N/A</div>'}
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card">
                        <div class="card-body">
                            <div class="metric-label">Volume</div>
                            <div class="metric-value">{metrics["latest_volume"]:,.0f}</div>
                            <div class="{"positive" if metrics["volume_ratio"] >= 1 else "negative"}">
                                {"+" if metrics["volume_ratio"] >= 1 else ""}{metrics["volume_ratio"]:.2f}x avg
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card">
                        <div class="card-body">
                            <div class="metric-label">Volatility</div>
                            <div class="metric-value">{metrics["volatility"]:.2f}%</div>
                            <div>Daily std dev</div>
                        </div>
                    </div>
                </div>
            </div>
            '''

            # Create price chart
            price_chart = plot_price_chart(df, ticker)
            price_chart_html = ''
            if price_chart:
                price_chart_json = price_chart.to_json()
                price_chart_html = f'''
                <div class="card mt-4">
                    <div class="card-body">
                        <h5 class="card-title">Price Chart</h5>
                        <div id="price-chart" style="height: 500px;"></div>
                        <script>
                            var price_data = {price_chart_json};
                            Plotly.newPlot('price-chart', price_data.data, price_data.layout);
                        </script>
                    </div>
                </div>
                '''

            # Create volume chart
            volume_chart = plot_volume_analysis(df, ticker)
            volume_chart_html = ''
            if volume_chart:
                volume_chart_json = volume_chart.to_json()
                volume_chart_html = f'''
                <div class="card mt-4">
                    <div class="card-body">
                        <h5 class="card-title">Volume Analysis</h5>
                        <div id="volume-chart" style="height: 400px;"></div>
                        <script>
                            var volume_data = {volume_chart_json};
                            Plotly.newPlot('volume-chart', volume_data.data, volume_data.layout);
                        </script>
                    </div>
                </div>
                '''

            # Create data table
            table_data = df.sort_values('date', ascending=False).head(10).to_html(classes='table table-striped', index=False)
            table_html = f'''
            <div class="card mt-4">
                <div class="card-body">
                    <h5 class="card-title">Raw Data</h5>
                    {table_data}
                    <a href="/api/data/{ticker}?days={days}" class="btn btn-primary mt-2">Download CSV</a>
                </div>
            </div>
            '''

            # Combine all content
            content = metrics_html + price_chart_html + volume_chart_html + table_html
        else:
            content = f'''
            <div class="alert alert-warning">
                <h4>No data available for {ticker}</h4>
                <p>Please check if the data has been imported.</p>
                <h5>Import Data</h5>
                <pre>python run_time_series_import.py --ticker {ticker} --force-full</pre>
            </div>
            '''

    # Render template
    html = index_template.format(
        custom_css=custom_css,
        ticker_options=ticker_options,
        checked_30=checked_30,
        checked_90=checked_90,
        checked_365=checked_365,
        checked_1825=checked_1825,
        content=content
    )

    return html

@app.route('/colab')
@app.route('/colab/<ticker>')
def colab(ticker=None):
    """Google Colab notebook page."""
    # Get available tickers
    tickers = get_ticker_list()

    # Default to master notebook
    if not ticker:
        ticker = 'master'

    # Set active status
    master_active = 'active' if ticker == 'master' else ''

    # Create ticker links HTML
    ticker_links = ''
    for t in tickers:
        active = 'active' if t == ticker else ''
        ticker_links += f'<a href="/colab/{t}" class="list-group-item list-group-item-action {active}">{t}</a>'

    # Set notebook title and URL
    if ticker == 'master':
        notebook_title = 'Master Summary Notebook - Volume Inefficiency Analysis'
        colab_url = notebook_config.get_notebook_url('master')
    else:
        notebook_title = f'{ticker} Volume Inefficiency Analysis'
        colab_url = notebook_config.get_notebook_url(ticker)

    # Render template
    html = colab_template.format(
        custom_css=custom_css,
        master_active=master_active,
        ticker_links=ticker_links,
        notebook_title=notebook_title,
        colab_url=colab_url
    )

    return html

@app.route('/api/data/<ticker>')
def get_data(ticker):
    """API endpoint to get ticker data as CSV."""
    days = request.args.get('days', 90, type=int)

    # Load data
    df = load_ticker_data(ticker, days=days)

    if df is not None and not df.empty:
        # Convert to CSV
        csv_data = df.to_csv(index=False)

        # Create response
        response = Response(
            csv_data,
            mimetype='text/csv',
            headers={
                "Content-Disposition": f"attachment; filename={ticker}_data.csv"
            }
        )

        return response
    else:
        return jsonify({"error": f"No data available for {ticker}"}), 404

@app.route('/api/docs')
def api_docs():
    """API documentation page."""
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Delphi API Documentation</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        {custom_css}
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
            <div class="container">
                <a class="navbar-brand" href="/">
                    <img src="https://img.icons8.com/color/48/000000/oracle-of-delphi.png" width="30" height="30" class="d-inline-block align-top" alt="">
                    Delphi Trading Intelligence
                </a>
                <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav ml-auto">
                        <li class="nav-item">
                            <a class="nav-link" href="/">Dashboard</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/colab">Analysis Notebooks</a>
                        </li>
                        <li class="nav-item active">
                            <a class="nav-link" href="/api/docs">API</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>
        <div class="container mt-4">
            <h1 class="main-header">API Documentation</h1>
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">GET /api/data/&lt;ticker&gt;</h5>
                    <p>Get historical price data for a ticker as CSV.</p>
                    <h6>Parameters:</h6>
                    <ul>
                        <li><code>days</code> (optional): Number of days of data to retrieve (default: 90)</li>
                    </ul>
                    <h6>Example:</h6>
                    <pre><code>GET /api/data/AAPL?days=30</code></pre>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == "__main__":
    # Create necessary directories
    Path("templates").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)

    # Run the Flask app
    app.run(host='0.0.0.0', port=8080, debug=True)
