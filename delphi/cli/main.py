"""
Main CLI Entry Point

This module provides the main entry point for the CLI.
"""

import sys
import argparse
import logging

from .import_command import import_data
from .analyze_command import analyze_data
from ..utils.logger import get_logger

logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the CLI.
    """
    # Create parser
    parser = argparse.ArgumentParser(description="Delphi - Financial Analysis Platform")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import data")
    import_parser.add_argument(
        "--source",
        choices=["alpha_vantage", "yfinance"],
        default="yfinance",
        help="Data source (default: yfinance)"
    )
    import_parser.add_argument(
        "--period",
        default="2y",
        help="Period to import (default: 2y)"
    )
    import_parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to import (default: from config)"
    )
    import_parser.add_argument(
        "--project-id",
        help="Google Cloud project ID (default: from config)"
    )
    import_parser.add_argument(
        "--dataset-id",
        help="BigQuery dataset ID (default: from config)"
    )
    import_parser.add_argument(
        "--table-id",
        help="BigQuery table ID (default: from config)"
    )
    import_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of workers for parallel processing (default: 4)"
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze data")
    analyze_parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days to analyze (default: 90)"
    )
    analyze_parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to analyze (default: from config)"
    )
    analyze_parser.add_argument(
        "--project-id",
        help="Google Cloud project ID (default: from config)"
    )
    analyze_parser.add_argument(
        "--dataset-id",
        help="BigQuery dataset ID (default: from config)"
    )
    analyze_parser.add_argument(
        "--table-id",
        help="BigQuery table ID (default: from config)"
    )
    analyze_parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to save plots to (default: plots)"
    )
    analyze_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of workers for parallel processing (default: 4)"
    )
    
    # Common arguments
    for subparser in [import_parser, analyze_parser]:
        subparser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level (default: INFO)"
        )
        subparser.add_argument(
            "--log-file",
            help="Log file (default: None)"
        )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logger
    logger = get_logger(__name__, getattr(logging, args.log_level.upper(), logging.INFO), args.log_file)
    
    # Run command
    if args.command == "import":
        return import_data()
    elif args.command == "analyze":
        return analyze_data()
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
