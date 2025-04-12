#!/bin/bash
# Launch the Delphi Trading Intelligence Dashboard on port 6000

echo "Starting Delphi Trading Intelligence Dashboard on port 6000..."
python -m trading_ai.cli.dashboard_cli --port=6000 --browser

echo "Press Ctrl+C to exit..."
