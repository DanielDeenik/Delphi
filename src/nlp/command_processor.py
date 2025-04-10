"""
Natural Language Command Processor for Oracle of Delphi.

This module processes natural language commands for importing and analyzing market data.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class CommandProcessor:
    """Processes natural language commands for the Oracle of Delphi application."""
    
    def __init__(self):
        # Command patterns for importing data
        self.import_patterns = [
            r"import\s+(?:data\s+for\s+)?(?:symbol[s]?\s+)?([A-Z]+(?:,\s*[A-Z]+)*)",
            r"get\s+(?:data\s+for\s+)?(?:symbol[s]?\s+)?([A-Z]+(?:,\s*[A-Z]+)*)",
            r"fetch\s+(?:data\s+for\s+)?(?:symbol[s]?\s+)?([A-Z]+(?:,\s*[A-Z]+)*)",
            r"load\s+(?:data\s+for\s+)?(?:symbol[s]?\s+)?([A-Z]+(?:,\s*[A-Z]+)*)",
            r"add\s+(?:data\s+for\s+)?(?:symbol[s]?\s+)?([A-Z]+(?:,\s*[A-Z]+)*)"
        ]
        
        # Command patterns for analyzing data
        self.analyze_patterns = [
            r"analyze\s+(?:data\s+for\s+)?(?:symbol[s]?\s+)?([A-Z]+(?:,\s*[A-Z]+)*)",
            r"show\s+(?:data\s+for\s+)?(?:symbol[s]?\s+)?([A-Z]+(?:,\s*[A-Z]+)*)",
            r"display\s+(?:data\s+for\s+)?(?:symbol[s]?\s+)?([A-Z]+(?:,\s*[A-Z]+)*)",
            r"chart\s+(?:data\s+for\s+)?(?:symbol[s]?\s+)?([A-Z]+(?:,\s*[A-Z]+)*)"
        ]
        
        # Time period patterns
        self.time_patterns = [
            (r"(?:for|over|last)\s+(\d+)\s+days?", self._parse_days),
            (r"(?:for|over|last)\s+(\d+)\s+weeks?", self._parse_weeks),
            (r"(?:for|over|last)\s+(\d+)\s+months?", self._parse_months),
            (r"(?:for|over|last)\s+(\d+)\s+years?", self._parse_years),
            (r"(?:since|from)\s+(\d{4}-\d{2}-\d{2})", self._parse_since_date),
            (r"(?:from)\s+(\d{4}-\d{2}-\d{2})\s+(?:to)\s+(\d{4}-\d{2}-\d{2})", self._parse_date_range)
        ]
    
    def _parse_days(self, match) -> Tuple[datetime, datetime]:
        days = int(match.group(1))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return start_date, end_date
    
    def _parse_weeks(self, match) -> Tuple[datetime, datetime]:
        weeks = int(match.group(1))
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=weeks)
        return start_date, end_date
    
    def _parse_months(self, match) -> Tuple[datetime, datetime]:
        months = int(match.group(1))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30*months)
        return start_date, end_date
    
    def _parse_years(self, match) -> Tuple[datetime, datetime]:
        years = int(match.group(1))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)
        return start_date, end_date
    
    def _parse_since_date(self, match) -> Tuple[datetime, datetime]:
        start_date = datetime.strptime(match.group(1), "%Y-%m-%d")
        end_date = datetime.now()
        return start_date, end_date
    
    def _parse_date_range(self, match) -> Tuple[datetime, datetime]:
        start_date = datetime.strptime(match.group(1), "%Y-%m-%d")
        end_date = datetime.strptime(match.group(2), "%Y-%m-%d")
        return start_date, end_date
    
    def process_command(self, command: str) -> Dict[str, Any]:
        """
        Process a natural language command.
        
        Args:
            command: The natural language command to process
            
        Returns:
            A dictionary with the parsed command details
        """
        command = command.strip()
        result = {
            'action': None,
            'symbols': [],
            'start_date': datetime.now() - timedelta(days=90),  # Default: last 90 days
            'end_date': datetime.now(),
            'original_command': command
        }
        
        # Check for import command
        for pattern in self.import_patterns:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                result['action'] = 'import'
                symbols_str = match.group(1)
                result['symbols'] = [s.strip() for s in symbols_str.split(',')]
                break
        
        # Check for analyze command
        if not result['action']:
            for pattern in self.analyze_patterns:
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    result['action'] = 'analyze'
                    symbols_str = match.group(1)
                    result['symbols'] = [s.strip() for s in symbols_str.split(',')]
                    break
        
        # If no action was found, return empty result
        if not result['action']:
            return result
        
        # Check for time period
        for pattern, parser in self.time_patterns:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                result['start_date'], result['end_date'] = parser(match)
                break
        
        return result
    
    def get_suggestions(self, partial_command: str) -> List[str]:
        """
        Get command suggestions based on partial input.
        
        Args:
            partial_command: The partial command to get suggestions for
            
        Returns:
            A list of command suggestions
        """
        suggestions = []
        
        # Common symbols
        common_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
        
        # If the command is empty or very short, suggest basic commands
        if len(partial_command) < 3:
            suggestions = [
                "import data for AAPL",
                "analyze data for MSFT",
                "show data for GOOGL over 30 days",
                "get data for TSLA, AAPL, MSFT"
            ]
        else:
            # Check if it looks like an import command
            if any(keyword in partial_command.lower() for keyword in ["import", "get", "fetch", "load", "add"]):
                # If no symbol is specified yet
                if not re.search(r"[A-Z]{2,}", partial_command):
                    for symbol in common_symbols:
                        suggestions.append(f"{partial_command} {symbol}")
                # If it looks complete, suggest adding time period
                else:
                    suggestions.append(f"{partial_command} for 30 days")
                    suggestions.append(f"{partial_command} for 3 months")
                    suggestions.append(f"{partial_command} since 2023-01-01")
            
            # Check if it looks like an analyze command
            elif any(keyword in partial_command.lower() for keyword in ["analyze", "show", "display", "chart"]):
                # If no symbol is specified yet
                if not re.search(r"[A-Z]{2,}", partial_command):
                    for symbol in common_symbols:
                        suggestions.append(f"{partial_command} {symbol}")
                # If it looks complete, suggest adding time period
                else:
                    suggestions.append(f"{partial_command} for 30 days")
                    suggestions.append(f"{partial_command} for 3 months")
                    suggestions.append(f"{partial_command} since 2023-01-01")
        
        return suggestions
