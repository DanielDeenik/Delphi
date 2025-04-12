"""
Data ingestion module for the Volume Intelligence Trading System.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import asyncio
import time
import os

from trading_ai.config import config_manager
from trading_ai.core.alpha_client import AlphaVantageClient
from trading_ai.core.bigquery_io import BigQueryStorage
from trading_ai.core.volume_footprint import VolumeAnalyzer
from trading_ai.core.sentiment import SentimentAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

class DataIngestionManager:
    """Manager for data ingestion and processing."""
    
    def __init__(self):
        """Initialize the data ingestion manager."""
        self.alpha_client = AlphaVantageClient()
        self.bigquery_storage = BigQueryStorage()
        self.volume_analyzer = VolumeAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    async def ingest_stock_data(self, ticker: str, force_full: bool = False) -> bool:
        """Ingest stock data for a ticker.
        
        Args:
            ticker: Stock symbol
            force_full: Whether to force a full data refresh
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Ingesting data for {ticker}...")
            
            # Determine output size based on existing data
            outputsize = 'full' if force_full else 'compact'
            
            # Fetch daily data
            df = await self.alpha_client.fetch_daily_data(ticker, outputsize)
            
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return False
            
            # Store in BigQuery
            success = self.bigquery_storage.store_stock_prices(ticker, df)
            
            if success:
                logger.info(f"Successfully ingested data for {ticker}")
            else:
                logger.warning(f"Failed to store data for {ticker}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error ingesting data for {ticker}: {str(e)}")
            return False
    
    async def ingest_all_stocks(self, force_full: bool = False) -> Dict[str, bool]:
        """Ingest data for all tracked stocks.
        
        Args:
            force_full: Whether to force a full data refresh
            
        Returns:
            Dictionary mapping tickers to success status
        """
        try:
            # Get all tracked tickers
            all_tickers = config_manager.get_all_tickers()
            
            # Initialize results
            results = {}
            
            # Process each ticker
            for ticker in all_tickers:
                success = await self.ingest_stock_data(ticker, force_full)
                results[ticker] = success
                
                # Add a delay to respect API rate limits
                await asyncio.sleep(12)  # 5 requests per minute = 12 seconds between requests
            
            # Log summary
            success_count = sum(1 for success in results.values() if success)
            logger.info(f"Ingestion completed: {success_count}/{len(all_tickers)} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Error ingesting all stocks: {str(e)}")
            return {}
    
    async def process_volume_analysis(self, ticker: str) -> bool:
        """Process volume analysis for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing volume analysis for {ticker}...")
            
            # Get stock data from BigQuery
            df = self.bigquery_storage.get_stock_prices(ticker)
            
            if df.empty:
                logger.warning(f"No data available for {ticker}")
                return False
            
            # Calculate volume metrics
            df = self.volume_analyzer.calculate_volume_metrics(df)
            
            # Detect volume inefficiencies
            df = self.volume_analyzer.detect_volume_inefficiencies(df)
            
            # Store volume analysis results
            success = self.bigquery_storage.store_volume_analysis(ticker, df)
            
            if success:
                logger.info(f"Successfully processed volume analysis for {ticker}")
            else:
                logger.warning(f"Failed to store volume analysis for {ticker}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing volume analysis for {ticker}: {str(e)}")
            return False
    
    async def process_all_volume_analysis(self) -> Dict[str, bool]:
        """Process volume analysis for all tracked stocks.
        
        Returns:
            Dictionary mapping tickers to success status
        """
        try:
            # Get all tracked tickers
            all_tickers = config_manager.get_all_tickers()
            
            # Initialize results
            results = {}
            
            # Process each ticker
            for ticker in all_tickers:
                success = await self.process_volume_analysis(ticker)
                results[ticker] = success
            
            # Log summary
            success_count = sum(1 for success in results.values() if success)
            logger.info(f"Volume analysis completed: {success_count}/{len(all_tickers)} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing all volume analysis: {str(e)}")
            return {}
    
    async def process_sentiment_analysis(self, ticker: str) -> Dict[str, Any]:
        """Process sentiment analysis for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with sentiment results
        """
        try:
            logger.info(f"Processing sentiment analysis for {ticker}...")
            
            # Fetch news sentiment
            news_df = self.sentiment_analyzer.fetch_news_sentiment(ticker)
            
            # Fetch social sentiment
            social_df = self.sentiment_analyzer.fetch_social_sentiment(ticker)
            
            # Calculate aggregate sentiment
            aggregate_sentiment = self.sentiment_analyzer.calculate_aggregate_sentiment(news_df, social_df)
            
            # Return results
            results = {
                'ticker': ticker,
                'news_sentiment': news_df,
                'social_sentiment': social_df,
                'aggregate_sentiment': aggregate_sentiment
            }
            
            logger.info(f"Successfully processed sentiment analysis for {ticker}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing sentiment analysis for {ticker}: {str(e)}")
            return {'ticker': ticker, 'error': str(e)}
    
    async def process_all_sentiment_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Process sentiment analysis for all tracked stocks.
        
        Returns:
            Dictionary mapping tickers to sentiment results
        """
        try:
            # Get all tracked tickers
            all_tickers = config_manager.get_all_tickers()
            
            # Initialize results
            results = {}
            
            # Process each ticker
            for ticker in all_tickers:
                sentiment_results = await self.process_sentiment_analysis(ticker)
                results[ticker] = sentiment_results
            
            # Log summary
            success_count = sum(1 for result in results.values() if 'error' not in result)
            logger.info(f"Sentiment analysis completed: {success_count}/{len(all_tickers)} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing all sentiment analysis: {str(e)}")
            return {}
    
    async def generate_master_summary(self) -> pd.DataFrame:
        """Generate master summary for all tracked stocks.
        
        Returns:
            DataFrame with master summary
        """
        try:
            logger.info("Generating master summary...")
            
            # Get all tracked tickers
            all_tickers = config_manager.get_all_tickers()
            
            # Initialize results list
            summary_data = []
            
            # Process each ticker
            for ticker in all_tickers:
                try:
                    # Get direction
                    direction = config_manager.get_ticker_direction(ticker)
                    
                    # Get volume analysis
                    volume_df = self.bigquery_storage.get_volume_analysis(ticker, days=30)
                    
                    if volume_df.empty:
                        logger.warning(f"No volume analysis available for {ticker}")
                        continue
                    
                    # Get latest data point
                    latest = volume_df.iloc[0]  # Assuming sorted by date desc
                    
                    # Get sentiment
                    sentiment_results = await self.process_sentiment_analysis(ticker)
                    aggregate_sentiment = sentiment_results.get('aggregate_sentiment', {})
                    
                    # Calculate stop loss and take profit
                    price_df = self.bigquery_storage.get_stock_prices(ticker, days=30)
                    if not price_df.empty:
                        # Calculate volume profile
                        volume_profile_df = self.volume_analyzer.calculate_volume_profile(price_df)
                        
                        # Calculate support/resistance
                        levels = self.volume_analyzer.calculate_support_resistance(price_df, volume_profile_df)
                        
                        # Get stop loss and take profit based on direction
                        if direction == 'buy':
                            stop_loss = levels.get('buy_stop_loss')
                            take_profit = levels.get('buy_take_profit')
                        else:  # short
                            stop_loss = levels.get('short_stop_loss')
                            take_profit = levels.get('short_take_profit')
                        
                        # Calculate risk/reward ratio
                        current_price = levels.get('current_price')
                        if current_price and stop_loss and take_profit:
                            if direction == 'buy':
                                risk = current_price - stop_loss
                                reward = take_profit - current_price
                            else:  # short
                                risk = stop_loss - current_price
                                reward = current_price - take_profit
                            
                            risk_reward_ratio = reward / risk if risk > 0 else 0
                        else:
                            risk_reward_ratio = 0
                    else:
                        stop_loss = None
                        take_profit = None
                        risk_reward_ratio = 0
                    
                    # Create summary entry
                    summary_entry = {
                        'date': latest['date'],
                        'symbol': ticker,
                        'direction': direction,
                        'close': latest.get('close', 0),
                        'volume': latest.get('volume', 0),
                        'relative_volume': latest.get('relative_volume_20d', 0),
                        'volume_z_score': latest.get('volume_z_score', 0),
                        'is_volume_spike': latest.get('is_volume_spike', False),
                        'spike_strength': latest.get('spike_strength', 0),
                        'price_change_pct': latest.get('price_change_pct', 0),
                        'signal': latest.get('signal', 'NEUTRAL'),
                        'confidence': latest.get('confidence', 0),
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk_reward_ratio': risk_reward_ratio,
                        'notes': latest.get('notes', ''),
                        'sentiment_compound': aggregate_sentiment.get('compound', 0),
                        'sentiment_positive_ratio': aggregate_sentiment.get('positive_ratio', 0),
                        'sentiment_negative_ratio': aggregate_sentiment.get('negative_ratio', 0),
                        'notebook_url': f"https://colab.research.google.com/drive/your-notebook-id-for-{ticker}",
                        'timestamp': datetime.now()
                    }
                    
                    summary_data.append(summary_entry)
                    
                except Exception as e:
                    logger.error(f"Error processing summary for {ticker}: {str(e)}")
            
            # Create DataFrame
            summary_df = pd.DataFrame(summary_data)
            
            # Store in BigQuery
            if not summary_df.empty:
                success = self.bigquery_storage.store_master_summary(summary_df)
                if success:
                    logger.info(f"Successfully stored master summary for {len(summary_df)} stocks")
                else:
                    logger.warning("Failed to store master summary")
            
            logger.info("Master summary generation completed")
            return summary_df
            
        except Exception as e:
            logger.error(f"Error generating master summary: {str(e)}")
            return pd.DataFrame()
    
    async def run_full_ingestion(self) -> bool:
        """Run full data ingestion and processing pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting full ingestion pipeline...")
            
            # Step 1: Initialize BigQuery tables
            self.bigquery_storage.initialize_tables()
            
            # Step 2: Ingest stock data
            await self.ingest_all_stocks()
            
            # Step 3: Process volume analysis
            await self.process_all_volume_analysis()
            
            # Step 4: Generate master summary
            await self.generate_master_summary()
            
            logger.info("Full ingestion pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error running full ingestion pipeline: {str(e)}")
            return False
