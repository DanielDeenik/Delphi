#!/bin/bash
# Shell script for running the integrated import system on Unix/Linux/Mac
# This script integrates with the existing codebase

echo "Running Delphi Integrated Import System"
echo "====================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Python not found in PATH. Please install Python or add it to your PATH."
    exit 1
fi

# Pass all arguments to the Python script
python scripts/run_integrated_import.py "$@"

# Check exit code
if [ $? -ne 0 ]; then
    echo "Import process failed with exit code $?"
    exit $?
fi

echo "Import process completed successfully"
exit 0
