
import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from src.models.omni_parser import OmniParser

class TestOmniParser(unittest.TestCase):
    def setUp(self):
        self.parser = OmniParser()
        
    def test_social_media_scraping(self):
        """Test social media data extraction"""
        with patch('src.models.omni_parser.tweepy.API') as mock_twitter:
            mock_twitter.return_value.search_tweets.return_value = [
                Mock(text="Great product!", favorite_count=100)
            ]
            result = self.parser.analyze_twitter_sentiment("AAPL")
            self.assertIsInstance(result, dict)
            self.assertIn('sentiment_score', result)
            
    def test_earnings_call_analysis(self):
        """Test earnings call sentiment analysis"""
        sample_text = "We are very confident about our growth prospects"
        result = self.parser.analyze_earnings_call(sample_text)
        self.assertIsInstance(result, dict)
        self.assertIn('confidence_score', result)
        
    def test_product_demand_tracking(self):
        """Test product demand analysis"""
        with patch('src.models.omni_parser.requests.get') as mock_request:
            mock_request.return_value.json.return_value = {'rating': 4.5}
            result = self.parser.track_product_demand("AAPL")
            self.assertIsInstance(result, dict)
            self.assertIn('demand_score', result)

if __name__ == '__main__':
    unittest.main()
