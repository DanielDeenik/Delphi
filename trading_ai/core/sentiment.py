"""
Sentiment analysis module for the Volume Intelligence Trading System.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import re
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import nltk

# Configure logging
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyzer for market sentiment from various sources."""
    
    def __init__(self, use_transformers: bool = False):
        """Initialize the sentiment analyzer.
        
        Args:
            use_transformers: Whether to use the transformers model (more accurate but slower)
        """
        self.use_transformers = use_transformers
        self.vader_analyzer = None
        self.transformers_analyzer = None
        
        # Initialize analyzers
        self._initialize_analyzers()
    
    def _initialize_analyzers(self) -> bool:
        """Initialize sentiment analyzers."""
        try:
            # Initialize VADER
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon')
            
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize transformers model if requested
            if self.use_transformers:
                model_name = "finiteautomata/bertweet-base-sentiment-analysis"
                self.transformers_analyzer = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name
                )
            
            logger.info("Sentiment analyzers initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzers: {str(e)}")
            return False
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            if not text:
                return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Use transformers if available
            if self.use_transformers and self.transformers_analyzer:
                try:
                    # Truncate text if too long
                    if len(cleaned_text) > 512:
                        cleaned_text = cleaned_text[:512]
                    
                    result = self.transformers_analyzer(cleaned_text)[0]
                    label = result['label']
                    score = result['score']
                    
                    if label == 'POS':
                        return {'compound': score, 'positive': score, 'negative': 0.0, 'neutral': 1.0 - score}
                    elif label == 'NEG':
                        return {'compound': -score, 'positive': 0.0, 'negative': score, 'neutral': 1.0 - score}
                    else:  # NEU
                        return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': score}
                
                except Exception as e:
                    logger.warning(f"Error using transformers analyzer: {str(e)}. Falling back to VADER.")
            
            # Use VADER
            if self.vader_analyzer:
                scores = self.vader_analyzer.polarity_scores(cleaned_text)
                return scores
            
            # Fallback to neutral sentiment
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to string if not already
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment of multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment scores
        """
        return [self.analyze_text(text) for text in texts]
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Analyze sentiment of texts in a DataFrame column.
        
        Args:
            df: DataFrame containing texts
            text_column: Name of the column containing texts
            
        Returns:
            DataFrame with added sentiment columns
        """
        try:
            if df.empty or text_column not in df.columns:
                logger.warning(f"DataFrame is empty or does not contain column {text_column}")
                return df
            
            # Make a copy to avoid modifying the original
            result_df = df.copy()
            
            # Apply sentiment analysis to each text
            sentiments = result_df[text_column].apply(self.analyze_text)
            
            # Extract sentiment scores
            result_df['sentiment_compound'] = sentiments.apply(lambda x: x['compound'])
            result_df['sentiment_positive'] = sentiments.apply(lambda x: x['positive'])
            result_df['sentiment_negative'] = sentiments.apply(lambda x: x['negative'])
            result_df['sentiment_neutral'] = sentiments.apply(lambda x: x['neutral'])
            
            # Add sentiment label
            result_df['sentiment_label'] = result_df['sentiment_compound'].apply(
                lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
            )
            
            logger.info(f"Successfully analyzed sentiment for {len(result_df)} texts")
            return result_df
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment in DataFrame: {str(e)}")
            return df
    
    def fetch_news_sentiment(self, ticker: str, days: int = 7) -> pd.DataFrame:
        """Fetch and analyze news sentiment for a ticker.
        
        Args:
            ticker: Stock symbol
            days: Number of days of news to fetch
            
        Returns:
            DataFrame with news and sentiment scores
        """
        try:
            # This is a placeholder - in a real implementation, you would:
            # 1. Fetch news from a news API (e.g., Alpha Vantage News API, NewsAPI, etc.)
            # 2. Analyze sentiment of each news article
            # 3. Return a DataFrame with the results
            
            # For now, we'll return a dummy DataFrame
            today = datetime.now()
            dates = [today - timedelta(days=i) for i in range(days)]
            
            # Create dummy news data
            news_data = []
            for date in dates:
                # Generate 1-3 news items per day
                for _ in range(np.random.randint(1, 4)):
                    sentiment = np.random.uniform(-1, 1)
                    
                    # Generate dummy headline based on sentiment
                    if sentiment > 0.3:
                        headline = f"{ticker} surges on positive earnings report"
                    elif sentiment < -0.3:
                        headline = f"{ticker} drops amid market concerns"
                    else:
                        headline = f"{ticker} remains stable as investors await news"
                    
                    news_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'ticker': ticker,
                        'headline': headline,
                        'url': f"https://example.com/news/{ticker}/{date.strftime('%Y%m%d')}",
                        'source': np.random.choice(['Bloomberg', 'Reuters', 'CNBC', 'WSJ']),
                        'sentiment_compound': sentiment
                    })
            
            # Create DataFrame
            df = pd.DataFrame(news_data)
            
            # Add sentiment labels
            df['sentiment_label'] = df['sentiment_compound'].apply(
                lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
            )
            
            # Sort by date (newest first)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
            
            logger.info(f"Successfully fetched and analyzed {len(df)} news items for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching news sentiment for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_social_sentiment(self, ticker: str, days: int = 7) -> pd.DataFrame:
        """Fetch and analyze social media sentiment for a ticker.
        
        Args:
            ticker: Stock symbol
            days: Number of days of social media posts to fetch
            
        Returns:
            DataFrame with social media posts and sentiment scores
        """
        try:
            # This is a placeholder - in a real implementation, you would:
            # 1. Fetch social media posts from APIs (e.g., Twitter API, Reddit API, etc.)
            # 2. Analyze sentiment of each post
            # 3. Return a DataFrame with the results
            
            # For now, we'll return a dummy DataFrame
            today = datetime.now()
            dates = [today - timedelta(days=i) for i in range(days)]
            
            # Create dummy social media data
            social_data = []
            for date in dates:
                # Generate 3-10 posts per day
                for _ in range(np.random.randint(3, 11)):
                    sentiment = np.random.uniform(-1, 1)
                    
                    # Generate dummy post based on sentiment
                    if sentiment > 0.3:
                        post = f"Just bought more ${ticker}! Looking bullish! 🚀"
                    elif sentiment < -0.3:
                        post = f"Selling my ${ticker} shares. Not looking good. 📉"
                    else:
                        post = f"What do you think about ${ticker} current price action?"
                    
                    social_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'ticker': ticker,
                        'platform': np.random.choice(['Twitter', 'Reddit', 'StockTwits']),
                        'post': post,
                        'user': f"user_{np.random.randint(1000, 9999)}",
                        'likes': np.random.randint(0, 100),
                        'sentiment_compound': sentiment
                    })
            
            # Create DataFrame
            df = pd.DataFrame(social_data)
            
            # Add sentiment labels
            df['sentiment_label'] = df['sentiment_compound'].apply(
                lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
            )
            
            # Sort by date (newest first)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
            
            logger.info(f"Successfully fetched and analyzed {len(df)} social media posts for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching social sentiment for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_aggregate_sentiment(self, news_df: pd.DataFrame, social_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate aggregate sentiment from news and social media.
        
        Args:
            news_df: DataFrame with news sentiment
            social_df: DataFrame with social media sentiment
            
        Returns:
            Dictionary with aggregate sentiment scores
        """
        try:
            if news_df.empty and social_df.empty:
                logger.warning("Both news and social DataFrames are empty")
                return {'compound': 0.0, 'positive_ratio': 0.0, 'negative_ratio': 0.0, 'neutral_ratio': 0.0}
            
            # Combine sentiment scores
            all_sentiments = []
            
            if not news_df.empty and 'sentiment_compound' in news_df.columns:
                # Weight news sentiment (2x)
                news_sentiments = news_df['sentiment_compound'].tolist() * 2
                all_sentiments.extend(news_sentiments)
            
            if not social_df.empty and 'sentiment_compound' in social_df.columns:
                # Weight social sentiment (1x)
                social_sentiments = social_df['sentiment_compound'].tolist()
                all_sentiments.extend(social_sentiments)
            
            if not all_sentiments:
                return {'compound': 0.0, 'positive_ratio': 0.0, 'negative_ratio': 0.0, 'neutral_ratio': 0.0}
            
            # Calculate aggregate sentiment
            compound = np.mean(all_sentiments)
            
            # Calculate sentiment ratios
            positive_count = sum(1 for s in all_sentiments if s > 0.05)
            negative_count = sum(1 for s in all_sentiments if s < -0.05)
            neutral_count = sum(1 for s in all_sentiments if -0.05 <= s <= 0.05)
            total_count = len(all_sentiments)
            
            positive_ratio = positive_count / total_count if total_count > 0 else 0.0
            negative_ratio = negative_count / total_count if total_count > 0 else 0.0
            neutral_ratio = neutral_count / total_count if total_count > 0 else 0.0
            
            logger.info(f"Successfully calculated aggregate sentiment from {total_count} sources")
            return {
                'compound': compound,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'neutral_ratio': neutral_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating aggregate sentiment: {str(e)}")
            return {'compound': 0.0, 'positive_ratio': 0.0, 'negative_ratio': 0.0, 'neutral_ratio': 0.0}
