"""
Notebook generator for the Volume Trading System.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

from volume_trading.config import config
from volume_trading.core.data_storage import DataStorage

# Configure logging
logger = logging.getLogger(__name__)

class NotebookGenerator:
    """Generator for Jupyter notebooks."""
    
    def __init__(self, notebooks_dir: Optional[Path] = None):
        """Initialize the notebook generator.
        
        Args:
            notebooks_dir: Directory for storing notebooks
        """
        self.notebooks_dir = notebooks_dir or Path("notebooks")
        
        # Create directories if they don't exist
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.storage = DataStorage()
    
    def generate_stock_notebook(self, ticker: str) -> Path:
        """Generate a notebook for a specific stock.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Path to the generated notebook
        """
        try:
            # Get direction
            tracked_stocks = config.get_tracked_stocks()
            direction = None
            for dir_type, tickers in tracked_stocks.items():
                if ticker in tickers:
                    direction = dir_type
                    break
            
            if direction is None:
                logger.warning(f"Ticker {ticker} not found in tracked stocks")
                direction = "unknown"
            
            # Create notebook
            nb = new_notebook()
            
            # Add title
            nb.cells.append(new_markdown_cell(f"# {ticker} Volume Analysis\n\n"
                                             f"Direction: {direction.upper()}\n\n"
                                             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
            
            # Add setup cell
            nb.cells.append(new_code_cell(
                "# Setup\n"
                "import pandas as pd\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "from datetime import datetime, timedelta\n\n"
                "# Set plot style\n"
                "sns.set(style='darkgrid')\n"
                "plt.rcParams['figure.figsize'] = (14, 8)"
            ))
            
            # Add data loading cell
            nb.cells.append(new_markdown_cell("## Load Data"))
            nb.cells.append(new_code_cell(
                f"# Load data for {ticker}\n"
                "from volume_trading.core.data_storage import DataStorage\n\n"
                "storage = DataStorage()\n"
                f"df = storage.load_price_data('{ticker}')\n\n"
                "# Display basic info\n"
                "print(f\"Loaded {len(df)} rows of data\")\n"
                "print(f\"Date range: {df.index.min()} to {df.index.max()}\")\n"
                "df.head()"
            ))
            
            # Add volume analysis cell
            nb.cells.append(new_markdown_cell("## Volume Analysis"))
            nb.cells.append(new_code_cell(
                "# Load analysis results\n"
                f"analysis_df = storage.load_analysis_results('{ticker}')\n\n"
                "# Display volume metrics\n"
                "print(f\"Volume spikes: {analysis_df['is_volume_spike'].sum()}\")\n"
                "analysis_df[analysis_df['is_volume_spike']].head()"
            ))
            
            # Add visualization cell
            nb.cells.append(new_markdown_cell("## Visualization"))
            nb.cells.append(new_code_cell(
                "# Plot price and volume\n"
                "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)\n\n"
                "# Plot price\n"
                "ax1.plot(analysis_df.index, analysis_df['close'], label='Close Price')\n"
                f"ax1.set_title('{ticker} Price', fontsize=16)\n"
                "ax1.set_ylabel('Price ($)', fontsize=14)\n"
                "ax1.legend()\n"
                "ax1.grid(True)\n\n"
                "# Plot volume\n"
                "ax2.bar(analysis_df.index, analysis_df['volume'], color='blue', alpha=0.5, label='Volume')\n"
                "ax2.plot(analysis_df.index, analysis_df['volume_ma20'], color='red', label='20-day MA')\n\n"
                "# Highlight volume spikes\n"
                "spike_dates = analysis_df[analysis_df['is_volume_spike']].index\n"
                "spike_volumes = analysis_df.loc[spike_dates, 'volume']\n"
                "ax2.scatter(spike_dates, spike_volumes, color='red', s=50, label='Volume Spike')\n\n"
                f"ax2.set_title('{ticker} Volume', fontsize=16)\n"
                "ax2.set_ylabel('Volume', fontsize=14)\n"
                "ax2.set_xlabel('Date', fontsize=14)\n"
                "ax2.legend()\n"
                "ax2.grid(True)\n\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ))
            
            # Add signals cell
            nb.cells.append(new_markdown_cell("## Trading Signals"))
            nb.cells.append(new_code_cell(
                "# Display signals\n"
                "signals = analysis_df[analysis_df['signal'] != 'NEUTRAL']\n"
                "print(f\"Found {len(signals)} signals\")\n\n"
                "# Display recent signals\n"
                "recent_signals = signals.head(5)\n"
                "for idx, row in recent_signals.iterrows():\n"
                "    print(f\"Date: {idx.strftime('%Y-%m-%d')}\")\n"
                "    print(f\"Signal: {row['signal']}\")\n"
                "    print(f\"Strength: {row['signal_strength']:.2f}\")\n"
                "    print(f\"Notes: {row['notes']}\")\n"
                "    print(\"---\")"
            ))
            
            # Add forward returns analysis cell
            nb.cells.append(new_markdown_cell("## Forward Returns Analysis"))
            nb.cells.append(new_code_cell(
                "# Calculate forward returns\n"
                "def calculate_forward_returns(df, days=[1, 3, 5, 10]):\n"
                "    result = df.copy()\n"
                "    for day in days:\n"
                "        result[f'return_{day}d'] = result['close'].pct_change(periods=day).shift(-day) * 100\n"
                "    return result\n\n"
                "# Calculate forward returns\n"
                "returns_df = calculate_forward_returns(analysis_df)\n\n"
                "# Analyze returns after volume spikes\n"
                "spike_returns = returns_df[returns_df['is_volume_spike']].copy()\n\n"
                "# Display average returns after spikes\n"
                "print(\"Average returns after volume spikes:\")\n"
                "for day in [1, 3, 5, 10]:\n"
                "    avg_return = spike_returns[f'return_{day}d'].mean()\n"
                "    print(f\"{day}-day return: {avg_return:.2f}%\")\n\n"
                "# Display recent volume spikes\n"
                "recent_spikes = spike_returns.head(5)\n"
                "recent_spikes[['close', 'volume', 'volume_z_score', 'return_1d', 'return_3d', 'return_5d']]"
            ))
            
            # Add summary cell
            nb.cells.append(new_markdown_cell("## Summary"))
            nb.cells.append(new_code_cell(
                "# Load summary\n"
                f"summary = storage.load_summary('{ticker}')\n\n"
                "# Display summary\n"
                "print(f\"Summary for {ticker}:\")\n"
                "for key, value in summary.items():\n"
                "    print(f\"{key}: {value}\")"
            ))
            
            # Save notebook
            notebook_path = self.notebooks_dir / f"{ticker}_analysis.ipynb"
            with open(notebook_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
            
            logger.info(f"Generated notebook for {ticker}: {notebook_path}")
            return notebook_path
            
        except Exception as e:
            logger.error(f"Error generating notebook for {ticker}: {str(e)}")
            return None
    
    def generate_master_notebook(self) -> Path:
        """Generate a master notebook for all stocks.
        
        Returns:
            Path to the generated notebook
        """
        try:
            # Create notebook
            nb = new_notebook()
            
            # Add title
            nb.cells.append(new_markdown_cell("# Volume Trading System - Master Dashboard\n\n"
                                             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
            
            # Add setup cell
            nb.cells.append(new_code_cell(
                "# Setup\n"
                "import pandas as pd\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "from datetime import datetime, timedelta\n"
                "import ipywidgets as widgets\n"
                "from IPython.display import display, HTML\n\n"
                "# Set plot style\n"
                "sns.set(style='darkgrid')\n"
                "plt.rcParams['figure.figsize'] = (14, 8)"
            ))
            
            # Add data loading cell
            nb.cells.append(new_markdown_cell("## Load Master Summary"))
            nb.cells.append(new_code_cell(
                "# Load master summary\n"
                "from volume_trading.core.data_storage import DataStorage\n\n"
                "storage = DataStorage()\n"
                "master_summary = storage.load_master_summary()\n\n"
                "# Get summaries\n"
                "summaries = master_summary.get('summaries', [])\n"
                "print(f\"Loaded {len(summaries)} stock summaries\")\n"
                "print(f\"Timestamp: {master_summary.get('timestamp', '')}\")"
            ))
            
            # Add dashboard cell
            nb.cells.append(new_markdown_cell("## Dashboard"))
            nb.cells.append(new_code_cell(
                "# Create dashboard\n"
                "def create_dashboard(summaries):\n"
                "    # Convert to DataFrame\n"
                "    df = pd.DataFrame(summaries)\n"
                "    \n"
                "    # Create HTML table\n"
                "    html = \"<h2>Stock Dashboard</h2>\"\n"
                "    html += \"<style>\"\n"
                "    html += \"table { border-collapse: collapse; width: 100%; }\"\n"
                "    html += \"th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }\"\n"
                "    html += \"th { background-color: #f2f2f2; }\"\n"
                "    html += \"tr:hover { background-color: #f5f5f5; }\"\n"
                "    html += \".buy { color: green; }\"\n"
                "    html += \".short { color: red; }\"\n"
                "    html += \".neutral { color: gray; }\"\n"
                "    html += \"</style>\"\n"
                "    \n"
                "    # Create table\n"
                "    html += \"<table>\"\n"
                "    html += \"<tr><th>Ticker</th><th>Direction</th><th>Close</th><th>Volume Z-Score</th><th>Signal</th><th>Strength</th><th>Notes</th></tr>\"\n"
                "    \n"
                "    # Sort by signal strength\n"
                "    sorted_df = df.sort_values('signal_strength', ascending=False)\n"
                "    \n"
                "    # Add rows\n"
                "    for _, row in sorted_df.iterrows():\n"
                "        direction_class = row['direction'].lower() if row['direction'] else 'neutral'\n"
                "        signal_class = 'neutral' if row['latest_signal'] == 'NEUTRAL' else direction_class\n"
                "        \n"
                "        html += f\"<tr>\"\n"
                "        html += f\"<td><strong>{row['ticker']}</strong></td>\"\n"
                "        html += f\"<td class='{direction_class}'>{row['direction']}</td>\"\n"
                "        html += f\"<td>${row['latest_close']:.2f}</td>\"\n"
                "        html += f\"<td>{row['latest_volume_z_score']:.2f}</td>\"\n"
                "        html += f\"<td class='{signal_class}'>{row['latest_signal']}</td>\"\n"
                "        html += f\"<td>{row['signal_strength']:.2f}</td>\"\n"
                "        html += f\"<td>{row['notes']}</td>\"\n"
                "        html += f\"</tr>\"\n"
                "    \n"
                "    html += \"</table>\"\n"
                "    \n"
                "    return HTML(html)\n\n"
                "# Display dashboard\n"
                "dashboard = create_dashboard(summaries)\n"
                "display(dashboard)"
            ))
            
            # Add signal summary cell
            nb.cells.append(new_markdown_cell("## Signal Summary"))
            nb.cells.append(new_code_cell(
                "# Create signal summary\n"
                "def create_signal_summary(summaries):\n"
                "    # Convert to DataFrame\n"
                "    df = pd.DataFrame(summaries)\n"
                "    \n"
                "    # Filter non-neutral signals\n"
                "    signals_df = df[df['latest_signal'] != 'NEUTRAL']\n"
                "    \n"
                "    # Group by direction and signal\n"
                "    if not signals_df.empty:\n"
                "        grouped = signals_df.groupby(['direction', 'latest_signal']).size().reset_index(name='count')\n"
                "        \n"
                "        # Plot\n"
                "        plt.figure(figsize=(10, 6))\n"
                "        ax = sns.barplot(x='latest_signal', y='count', hue='direction', data=grouped)\n"
                "        plt.title('Signal Summary', fontsize=16)\n"
                "        plt.xlabel('Signal', fontsize=14)\n"
                "        plt.ylabel('Count', fontsize=14)\n"
                "        plt.xticks(rotation=45)\n"
                "        plt.tight_layout()\n"
                "        plt.show()\n"
                "    else:\n"
                "        print(\"No signals found\")\n\n"
                "# Display signal summary\n"
                "create_signal_summary(summaries)"
            ))
            
            # Add volume spikes cell
            nb.cells.append(new_markdown_cell("## Volume Spikes"))
            nb.cells.append(new_code_cell(
                "# Create volume spikes summary\n"
                "def create_volume_spikes_summary(summaries):\n"
                "    # Convert to DataFrame\n"
                "    df = pd.DataFrame(summaries)\n"
                "    \n"
                "    # Filter volume spikes\n"
                "    spikes_df = df[df['is_volume_spike'] == True]\n"
                "    \n"
                "    if not spikes_df.empty:\n"
                "        # Sort by volume Z-score\n"
                "        sorted_df = spikes_df.sort_values('latest_volume_z_score', ascending=False)\n"
                "        \n"
                "        # Plot\n"
                "        plt.figure(figsize=(12, 6))\n"
                "        ax = sns.barplot(x='ticker', y='latest_volume_z_score', hue='direction', data=sorted_df)\n"
                "        plt.title('Volume Spikes', fontsize=16)\n"
                "        plt.xlabel('Ticker', fontsize=14)\n"
                "        plt.ylabel('Volume Z-Score', fontsize=14)\n"
                "        plt.xticks(rotation=45)\n"
                "        plt.tight_layout()\n"
                "        plt.show()\n"
                "        \n"
                "        # Display top spikes\n"
                "        print(\"Top Volume Spikes:\")\n"
                "        for _, row in sorted_df.head(5).iterrows():\n"
                "            print(f\"{row['ticker']}: Z-Score = {row['latest_volume_z_score']:.2f}, Signal = {row['latest_signal']}\")\n"
                "    else:\n"
                "        print(\"No volume spikes found\")\n\n"
                "# Display volume spikes summary\n"
                "create_volume_spikes_summary(summaries)"
            ))
            
            # Add stock selector cell
            nb.cells.append(new_markdown_cell("## Stock Selector"))
            nb.cells.append(new_code_cell(
                "# Create stock selector\n"
                "def create_stock_selector(summaries):\n"
                "    # Get tickers\n"
                "    tickers = [summary['ticker'] for summary in summaries]\n"
                "    \n"
                "    # Create dropdown\n"
                "    dropdown = widgets.Dropdown(\n"
                "        options=tickers,\n"
                "        description='Ticker:',\n"
                "        style={'description_width': 'initial'}\n"
                "    )\n"
                "    \n"
                "    # Create button\n"
                "    button = widgets.Button(\n"
                "        description='Load Notebook',\n"
                "        button_style='primary',\n"
                "        tooltip='Load notebook for selected ticker'\n"
                "    )\n"
                "    \n"
                "    # Create output\n"
                "    output = widgets.Output()\n"
                "    \n"
                "    # Define button click handler\n"
                "    def on_button_click(b):\n"
                "        with output:\n"
                "            output.clear_output()\n"
                "            ticker = dropdown.value\n"
                "            print(f\"Loading notebook for {ticker}...\")\n"
                "            print(f\"Please run the following cell to open the notebook:\")\n"
                "            print(f\"!jupyter notebook notebooks/{ticker}_analysis.ipynb\")\n"
                "    \n"
                "    # Register button click handler\n"
                "    button.on_click(on_button_click)\n"
                "    \n"
                "    # Display widgets\n"
                "    display(widgets.HBox([dropdown, button]))\n"
                "    display(output)\n\n"
                "# Display stock selector\n"
                "create_stock_selector(summaries)"
            ))
            
            # Add data refresh cell
            nb.cells.append(new_markdown_cell("## Data Refresh"))
            nb.cells.append(new_code_cell(
                "# Refresh data\n"
                "def refresh_data():\n"
                "    from volume_trading.import_all_data import import_all_data\n"
                "    import_all_data(force_refresh=False)\n"
                "    print(\"Data refresh completed. Please restart the notebook to see updated data.\")\n\n"
                "# Create refresh button\n"
                "refresh_button = widgets.Button(\n"
                "    description='Refresh Data',\n"
                "    button_style='danger',\n"
                "    tooltip='Refresh data for all stocks'\n"
                ")\n\n"
                "# Create output\n"
                "refresh_output = widgets.Output()\n\n"
                "# Define button click handler\n"
                "def on_refresh_click(b):\n"
                "    with refresh_output:\n"
                "        refresh_output.clear_output()\n"
                "        print(\"Refreshing data...\")\n"
                "        refresh_data()\n\n"
                "# Register button click handler\n"
                "refresh_button.on_click(on_refresh_click)\n\n"
                "# Display widgets\n"
                "display(refresh_button)\n"
                "display(refresh_output)"
            ))
            
            # Save notebook
            notebook_path = self.notebooks_dir / "master_dashboard.ipynb"
            with open(notebook_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
            
            logger.info(f"Generated master notebook: {notebook_path}")
            return notebook_path
            
        except Exception as e:
            logger.error(f"Error generating master notebook: {str(e)}")
            return None
    
    def generate_all_notebooks(self) -> List[Path]:
        """Generate notebooks for all tracked stocks and a master notebook.
        
        Returns:
            List of paths to the generated notebooks
        """
        try:
            # Get all tracked tickers
            tickers = config.get_all_tickers()
            
            # Generate notebooks for each ticker
            notebook_paths = []
            for ticker in tickers:
                path = self.generate_stock_notebook(ticker)
                if path:
                    notebook_paths.append(path)
            
            # Generate master notebook
            master_path = self.generate_master_notebook()
            if master_path:
                notebook_paths.append(master_path)
            
            logger.info(f"Generated {len(notebook_paths)} notebooks")
            return notebook_paths
            
        except Exception as e:
            logger.error(f"Error generating notebooks: {str(e)}")
            return []
