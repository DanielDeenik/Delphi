#!/bin/bash
# Delphi Trading Intelligence System Launcher
# This script automatically launches the Delphi application and opens the browser

echo "======================================================"
echo "   Delphi Trading Intelligence System Launcher"
echo "======================================================"
echo

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in the PATH."
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

echo "[1/4] Setting up environment..."
# Create necessary directories
mkdir -p logs status templates static config

# Check if config file exists
if [ ! -f "config/config.json" ]; then
    echo "[INFO] Creating default configuration..."
    cat > config/config.json << EOL
{
  "alpha_vantage": {
    "api_key": "IAS7UEKOT0HZW0MY",
    "base_url": "https://www.alphavantage.co/query"
  },
  "google_cloud": {
    "project_id": "delphi-449908",
    "dataset": "stock_data"
  },
  "tickers": [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
    "TSLA", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "UNH", "HD",
    "BAC", "PFE", "CSCO", "DIS", "VZ"
  ]
}
EOL
fi

echo "[2/4] Starting Delphi application on port 6000..."
# Start the Flask application in the background
python -m trading_ai.cli.dashboard_cli --port 6000 > logs/app_$(date +%Y%m%d).log 2>&1 &
APP_PID=$!

# Save the PID to a file for later cleanup
echo $APP_PID > .app.pid

echo "[3/4] Waiting for application to start..."
# Wait for the application to start
sleep 5

echo "[4/4] Opening browser..."
# Open the browser to the multi-tab Colab view
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    open http://localhost:6000/colab/all
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Linux
    xdg-open http://localhost:6000/colab/all
fi

echo
echo "======================================================"
echo "   Delphi Trading Intelligence System is running"
echo "   Dashboard URL: http://localhost:6000"
echo "   Notebooks URL: http://localhost:6000/colab"
echo "   All Notebooks: http://localhost:6000/colab/all"
echo "======================================================"
echo
echo "Press Ctrl+C to stop the application"
echo

# Handle cleanup on exit
cleanup() {
    echo
    echo "Stopping Delphi application..."
    if [ -f .app.pid ]; then
        kill $(cat .app.pid) 2>/dev/null
        rm .app.pid
    fi
    echo "Delphi application stopped."
    exit 0
}

# Set up trap for cleanup
trap cleanup INT TERM

# Keep the script running
while true; do
    sleep 1
done
