"""
Flask application for Oracle of Delphi with screenshot analysis capabilities.

This application allows users to upload screenshots from financial platforms,
extract tickers, and analyze them using neural network models.
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
import sqlite3

# Import custom modules
from src.ocr.image_processor import ImageProcessor
from src.models.neural_network_analyzer import NeuralNetworkAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Maximum file size (5MB)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Initialize services
image_processor = ImageProcessor()
neural_network = NeuralNetworkAnalyzer()

# SQLite database path
DB_PATH = os.path.join('data', 'market_data.db')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_market_data(symbol, start_date=None, end_date=None):
    """Retrieve market data from SQLite."""
    try:
        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=90)
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
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

def get_available_symbols():
    """Get a list of all available symbols in the database."""
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Query distinct symbols
        cursor.execute('SELECT DISTINCT symbol FROM market_data')
        symbols = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return symbols
    except Exception as e:
        logger.error(f"Error getting available symbols: {str(e)}")
        return []

@app.route('/')
def index():
    """Render the main page."""
    return render_template('screenshot_analyzer.html')

@app.route('/api/symbols')
def api_symbols():
    """API endpoint to get available symbols."""
    symbols = get_available_symbols()
    return jsonify(symbols)

@app.route('/api/upload_screenshot', methods=['POST'])
def upload_screenshot():
    """Handle screenshot upload and processing."""
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if the file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save the file
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        result = image_processor.process_image(filepath)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing screenshot: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze_tickers', methods=['POST'])
def analyze_tickers():
    """Analyze tickers using neural network models."""
    data = request.json
    
    if not data or 'tickers' not in data:
        return jsonify({'error': 'No tickers provided'}), 400
    
    tickers = data['tickers']
    models = data.get('models', ['price_prediction', 'trend_analysis', 'volatility_forecast', 'volume_analysis', 'sentiment_analysis'])
    
    results = {}
    
    for ticker in tickers:
        # Get market data for the ticker
        market_data = get_market_data(ticker)
        
        if market_data.empty:
            results[ticker] = {'error': f"No data available for {ticker}"}
            continue
        
        # Analyze the ticker
        analysis = neural_network.analyze_ticker(ticker, market_data, models)
        results[ticker] = analysis
    
    return jsonify(results)

@app.route('/api/market_data/<symbol>')
def api_market_data(symbol):
    """API endpoint to get market data for a symbol."""
    try:
        days = int(request.args.get('days', 90))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = get_market_data(symbol, start_date, end_date)
        
        if data.empty:
            return jsonify({'error': f"No data found for {symbol}"}), 404
            
        return jsonify(data.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
