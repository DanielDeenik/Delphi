#!/usr/bin/env python
"""
Simple Flask application for displaying market data from SQLite.

This is a simplified version of the Oracle of Delphi application
that only requires Flask, pandas, and requests.
"""

from flask import Flask, render_template, jsonify, request
import logging
import os
import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

class SimpleSQLiteService:
    """Simple service for retrieving market data from SQLite."""
    
    def __init__(self):
        self.db_path = os.path.join('data', 'market_data.db')
        if not os.path.exists(self.db_path):
            logger.error(f"SQLite database not found at {self.db_path}")
            logger.info("Please run simple_import.py first to import data")
    
    def get_market_data(self, symbol, start_date=None, end_date=None):
        """Retrieve market data from SQLite."""
        try:
            # Set default date range if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=90)
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query data
            cursor.execute('''
            SELECT date, data
            FROM market_data
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date DESC
            ''', (symbol, start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                logger.warning(f"No data found for {symbol} in the specified date range")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data_list = []
            for date_str, data_json in rows:
                row_data = json.loads(data_json)
                row_data['date'] = pd.to_datetime(date_str)
                data_list.append(row_data)
            
            df = pd.DataFrame(data_list)
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving market data from SQLite: {str(e)}")
            return pd.DataFrame()

# Initialize services
storage_service = SimpleSQLiteService()

@app.route('/')
def index():
    return render_template('simple_index.html')

@app.route('/api/market_data/<symbol>')
def get_market_data(symbol):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        data = storage_service.get_market_data(symbol=symbol, start_date=start_date, end_date=end_date)
        
        if data.empty:
            return jsonify({"error": f"No data found for {symbol}"}), 404
            
        return jsonify(data.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/symbols')
def get_symbols():
    try:
        # Connect to database
        conn = sqlite3.connect(storage_service.db_path)
        cursor = conn.cursor()
        
        # Query distinct symbols
        cursor.execute('SELECT DISTINCT symbol FROM market_data')
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return jsonify(symbols)
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
