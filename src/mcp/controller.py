"""
Controller component of the MCP architecture for Oracle of Delphi.

This module handles business logic and coordinates between the model and presenter.
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from .model import MarketDataModel
from src.nlp.command_processor import CommandProcessor

logger = logging.getLogger(__name__)

class MarketDataController:
    """Controller for market data operations."""
    
    def __init__(self, api_key: str):
        self.model = MarketDataModel()
        self.nlp_processor = CommandProcessor()
        self.api_key = api_key
        self.base_url = 'https://www.alphavantage.co/query'
    
    def process_command(self, command: str) -> Dict[str, Any]:
        """
        Process a natural language command.
        
        Args:
            command: The natural language command to process
            
        Returns:
            A dictionary with the result of the command
        """
        # Parse the command
        parsed = self.nlp_processor.process_command(command)
        
        # If no action was found, return error
        if not parsed['action']:
            return {
                'success': False,
                'message': f"Could not understand command: '{command}'",
                'suggestions': self.nlp_processor.get_suggestions(command)
            }
        
        # If no symbols were found, return error
        if not parsed['symbols']:
            return {
                'success': False,
                'message': "No symbols specified in command",
                'suggestions': self.nlp_processor.get_suggestions(command)
            }
        
        # Process the command based on the action
        if parsed['action'] == 'import':
            return self._handle_import(parsed)
        elif parsed['action'] == 'analyze':
            return self._handle_analyze(parsed)
        else:
            return {
                'success': False,
                'message': f"Unknown action: {parsed['action']}",
                'suggestions': self.nlp_processor.get_suggestions(command)
            }
    
    def _handle_import(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Handle import command."""
        results = {
            'success': True,
            'action': 'import',
            'symbols': parsed['symbols'],
            'results': {},
            'message': ""
        }
        
        for symbol in parsed['symbols']:
            try:
                # Fetch data from Alpha Vantage
                data = self._fetch_from_alpha_vantage(symbol)
                
                if data is None or data.empty:
                    results['results'][symbol] = {
                        'success': False,
                        'message': f"Failed to fetch data for {symbol}"
                    }
                    continue
                
                # Store data in the model
                success = self.model.store_market_data(symbol, data)
                
                results['results'][symbol] = {
                    'success': success,
                    'rows': len(data) if success else 0,
                    'message': f"Successfully imported {len(data)} rows for {symbol}" if success else f"Failed to store data for {symbol}"
                }
                
            except Exception as e:
                logger.error(f"Error importing data for {symbol}: {str(e)}")
                results['results'][symbol] = {
                    'success': False,
                    'message': f"Error: {str(e)}"
                }
        
        # Generate overall message
        success_count = sum(1 for r in results['results'].values() if r['success'])
        total_count = len(results['results'])
        
        if success_count == total_count:
            results['message'] = f"Successfully imported data for all {total_count} symbols"
        elif success_count == 0:
            results['success'] = False
            results['message'] = f"Failed to import data for all {total_count} symbols"
        else:
            results['message'] = f"Imported data for {success_count} out of {total_count} symbols"
        
        return results
    
    def _handle_analyze(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analyze command."""
        results = {
            'success': True,
            'action': 'analyze',
            'symbols': parsed['symbols'],
            'start_date': parsed['start_date'],
            'end_date': parsed['end_date'],
            'results': {},
            'message': ""
        }
        
        for symbol in parsed['symbols']:
            try:
                # Get data from the model
                data = self.model.get_market_data(symbol, parsed['start_date'], parsed['end_date'])
                
                if data.empty:
                    results['results'][symbol] = {
                        'success': False,
                        'message': f"No data found for {symbol} in the specified date range"
                    }
                    continue
                
                # Calculate some basic statistics
                latest_price = data['Close'].iloc[0] if 'Close' in data.columns else None
                avg_price = data['Close'].mean() if 'Close' in data.columns else None
                avg_volume = data['Volume'].mean() if 'Volume' in data.columns else None
                
                results['results'][symbol] = {
                    'success': True,
                    'rows': len(data),
                    'data': data.to_dict(orient='records'),
                    'stats': {
                        'latest_price': latest_price,
                        'avg_price': avg_price,
                        'avg_volume': avg_volume
                    },
                    'message': f"Retrieved {len(data)} rows for {symbol}"
                }
                
            except Exception as e:
                logger.error(f"Error analyzing data for {symbol}: {str(e)}")
                results['results'][symbol] = {
                    'success': False,
                    'message': f"Error: {str(e)}"
                }
        
        # Generate overall message
        success_count = sum(1 for r in results['results'].values() if r['success'])
        total_count = len(results['results'])
        
        if success_count == total_count:
            results['message'] = f"Successfully analyzed data for all {total_count} symbols"
        elif success_count == 0:
            results['success'] = False
            results['message'] = f"Failed to analyze data for all {total_count} symbols"
        else:
            results['message'] = f"Analyzed data for {success_count} out of {total_count} symbols"
        
        return results
    
    def _fetch_from_alpha_vantage(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage API."""
        try:
            # Fetch from Alpha Vantage
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': 'full',
                'apikey': self.api_key
            }

            logger.info(f"Fetching data for {symbol} from Alpha Vantage...")
            response = requests.get(self.base_url, params=params)
            data = response.json()

            if 'Time Series (Daily)' not in data:
                error_msg = data.get('Note', data.get('Error Message', 'Unknown error'))
                logger.error(f"Error fetching data for {symbol}: {error_msg}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')

            # Map column names
            column_mapping = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. adjusted close': 'Adjusted_Close',
                '6. volume': 'Volume',
                '7. dividend amount': 'Dividend',
                '8. split coefficient': 'Split_Coefficient'
            }

            # Check for missing columns
            missing_columns = [col for col in column_mapping.keys() if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing columns in API response for {symbol}: {missing_columns}")
                # Continue anyway, just map the columns that exist
                column_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}

            # Rename columns
            df = df.rename(columns=column_mapping)

            # Convert index to datetime
            df.index = pd.to_datetime(df.index)

            # Convert all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Forward fill any NaN values
            df = df.ffill()

            logger.info(f"Successfully fetched data for {symbol}. Shape: {df.shape}")
            
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """Get a list of all available symbols in the database."""
        return self.model.get_available_symbols()
