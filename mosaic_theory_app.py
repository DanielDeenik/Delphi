"""
Mosaic Theory Evaluation API

This Flask application provides an API for evaluating stocks using the Mosaic Theory approach.
It also includes a simple web interface for uploading screenshots and analyzing extracted tickers.
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Any
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd

# Import custom modules
from src.ocr.image_processor import ImageProcessor
from src.analysis.mosaic_theory_evaluator import MosaicTheoryEvaluator
from src.data.alpha_vantage_batch_importer import AlphaVantageBatchImporter

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
evaluator = MosaicTheoryEvaluator()

# Get API key from environment
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
if not api_key:
    logger.warning("ALPHA_VANTAGE_API_KEY environment variable not set. Import functionality will be limited.")
    importer = None
else:
    importer = AlphaVantageBatchImporter(api_key)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page."""
    return render_template('mosaic_theory.html')

@app.route('/api/evaluate/<symbol>')
def evaluate_stock(symbol):
    """API endpoint to evaluate a single stock."""
    try:
        days = int(request.args.get('days', 90))
        evaluation = evaluator.evaluate_stock(symbol, days)
        return jsonify(evaluation)
    except Exception as e:
        logger.error(f"Error evaluating {symbol}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluate_multiple', methods=['POST'])
def evaluate_multiple():
    """API endpoint to evaluate multiple stocks."""
    try:
        data = request.json
        
        if not data or 'symbols' not in data:
            return jsonify({"error": "No symbols provided"}), 400
        
        symbols = data['symbols']
        days = data.get('days', 90)
        
        results = evaluator.evaluate_multiple(symbols, days)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error evaluating stocks: {str(e)}")
        return jsonify({"error": str(e)}), 500

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

@app.route('/api/import_tickers', methods=['POST'])
def import_tickers():
    """Import data for tickers."""
    try:
        data = request.json
        
        if not data or 'tickers' not in data:
            return jsonify({"error": "No tickers provided"}), 400
        
        tickers = data['tickers']
        
        if not importer:
            return jsonify({"error": "AlphaVantage API key not configured"}), 500
        
        # Add tickers to the import queue
        importer.add_tickers(tickers, priority=10)
        
        # Import a batch
        results = importer.import_batch(len(tickers))
        
        return jsonify({
            "message": f"Imported {sum(1 for v in results.values() if v)}/{len(results)} tickers",
            "results": {k: "Success" if v else "Failed" for k, v in results.items()}
        })
    except Exception as e:
        logger.error(f"Error importing tickers: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze_from_screenshot', methods=['POST'])
def analyze_from_screenshot():
    """Process a screenshot, extract tickers, import data, and evaluate."""
    try:
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
        
        # Save the file
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        ocr_result = image_processor.process_image(filepath)
        
        if 'error' in ocr_result:
            return jsonify({'error': ocr_result['error']}), 500
        
        # Extract tickers
        tickers = [symbol for symbol, _ in ocr_result['tickers']]
        
        if not tickers:
            return jsonify({'error': 'No tickers found in the screenshot'}), 404
        
        # Import data if importer is available
        import_results = {}
        if importer:
            importer.add_tickers(tickers, priority=10)
            import_results = importer.import_batch(len(tickers))
        
        # Evaluate tickers
        evaluations = evaluator.evaluate_multiple(tickers)
        
        # Combine results
        result = {
            'ocr_result': ocr_result,
            'import_results': {k: "Success" if v else "Failed" for k, v in import_results.items()},
            'evaluations': evaluations
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error analyzing from screenshot: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
