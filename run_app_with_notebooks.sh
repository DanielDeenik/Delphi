#!/bin/bash
# Launch the Delphi Trading Intelligence Dashboard on port 3000 and open all notebooks

echo "Starting Delphi Trading Intelligence Dashboard on port 3000..."
python -m trading_ai.cli.dashboard_cli --port=3000 --browser &

echo "Opening all notebooks view..."
sleep 3
xdg-open http://localhost:3000/colab/all || open http://localhost:3000/colab/all

echo "Press Ctrl+C to exit..."
wait
