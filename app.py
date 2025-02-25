from flask import Flask, render_template, jsonify, request
from src.models.volume_analyzer import VolumeAnalyzer
from src.data.alpha_vantage_client import AlphaVantageClient
from src.services.sentiment_analysis_service import SentimentAnalysisService
from src.services.timeseries_storage_service import TimeSeriesStorageService
import logging
from datetime import datetime, timedelta
import os
import pandas as pd
from functools import lru_cache
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json


app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
alpha_vantage = AlphaVantageClient()
volume_analyzer = VolumeAnalyzer()
sentiment_service = SentimentAnalysisService()
storage_service = TimeSeriesStorageService()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/market_data/<symbol>')
def get_market_data(symbol):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        data = storage_service.get_market_data(symbol=symbol, start_date=start_date, end_date=end_date)
        if data.empty:
            data = alpha_vantage.fetch_daily_adjusted(symbol)
        return jsonify(data.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sentiment/<symbol>')
def get_sentiment(symbol):
    try:
        sentiment = sentiment_service.get_sentiment_analysis(symbol)
        return jsonify(sentiment)
    except Exception as e:
        logger.error(f"Error getting sentiment: {e}")
        return jsonify({"error": str(e)}), 500

#This section is adapted from the original code.  The caching mechanisms are preserved.  However, they are no longer Streamlit caches, but rather standard Python caches.  The functions are adapted to be used with the Flask app.
@lru_cache(maxsize=128) #Adjust maxsize as needed
def fetch_market_data_flask(symbol: str, start_date: datetime, end_date: datetime):
    try:
        data = storage_service.get_market_data(symbol=symbol, start_date=start_date, end_date=end_date)
        if data.empty:
            data = alpha_vantage.fetch_daily_adjusted(symbol)
        return data
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return None

@lru_cache(maxsize=128) #Adjust maxsize as needed
def get_sentiment_analysis_flask(symbol: str):
    try:
        return sentiment_service.get_sentiment_analysis(symbol)
    except Exception as e:
        logger.error(f"Error getting sentiment analysis: {e}")
        return None

@lru_cache(maxsize=128) #Adjust maxsize as needed
def fetch_watchlist_data_flask(symbols):
    """Fetch data for watchlist symbols"""
    data = {}
    for symbol in symbols:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            df = fetch_market_data_flask(symbol, start_date, end_date)
            if df is not None and not df.empty:
                latest_price = df['Close'].iloc[-1]
                price_change = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
                data[symbol] = {
                    'price': latest_price,
                    'change': price_change
                }
        except Exception as e:
            logger.error(f"Error fetching watchlist data for {symbol}: {e}")
    return data



if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=3000)