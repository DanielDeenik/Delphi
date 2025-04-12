#!/bin/bash
# Generate and upload notebooks for all tracked tickers

echo "Generating and uploading notebooks..."
python -m trading_ai.cli.notebook_cli --generate --upload

echo "Done!"
