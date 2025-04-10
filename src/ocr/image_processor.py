"""
Image processing module for extracting tickers from screenshots.

This module uses OCR to extract text from images and then identifies
stock tickers within the extracted text.
"""

import os
import re
import logging
import tempfile
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

# For OCR, we'll use pytesseract which is a wrapper for Tesseract OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("pytesseract not installed. OCR functionality will be limited.")

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Processes images to extract stock tickers."""
    
    def __init__(self):
        # Common stock exchanges for verification
        self.exchanges = ['NYSE', 'NASDAQ', 'AMEX']
        
        # Common words that might be confused as tickers
        self.common_words = set(['THE', 'AND', 'FOR', 'INC', 'LTD', 'LLC', 'CORP', 'CO'])
        
        # Known platform-specific patterns
        self.platform_patterns = {
            'danelfin': {
                'ticker_pattern': r'\b[A-Z]{1,5}\b',  # 1-5 uppercase letters
                'context_patterns': [
                    r'Score:\s*(\d+(?:\.\d+)?)',
                    r'Target:\s*\$?(\d+(?:\.\d+)?)',
                    r'Potential:\s*(\+?\d+(?:\.\d+)?%)'
                ]
            },
            'finchat': {
                'ticker_pattern': r'\$([A-Z]{1,5})\b',  # Tickers often prefixed with $
                'context_patterns': [
                    r'Price:\s*\$?(\d+(?:\.\d+)?)',
                    r'Change:\s*(\+?\-?\d+(?:\.\d+)?%)',
                    r'Volume:\s*(\d+(?:\.\d+)?[KMB]?)'
                ]
            },
            'generic': {
                'ticker_pattern': r'\b[A-Z]{1,5}\b',  # Generic pattern for tickers
                'context_patterns': []
            }
        }
        
        # Load list of valid tickers (this would be a more comprehensive list in production)
        self.valid_tickers = self._load_valid_tickers()
    
    def _load_valid_tickers(self) -> set:
        """Load a set of valid ticker symbols."""
        # In a real implementation, this would load from a database or API
        # For now, we'll use a small set of common tickers
        return {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC',
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'PYPL', 'SQ',
            'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'DISH', 'ROKU', 'SPOT',
            'PFE', 'JNJ', 'MRK', 'ABBV', 'BMY', 'LLY', 'AMGN', 'GILD', 'BIIB', 'REGN',
            'WMT', 'TGT', 'COST', 'HD', 'LOW', 'SBUX', 'MCD', 'YUM', 'CMG', 'DKNG',
            'F', 'GM', 'TSLA', 'TM', 'HMC', 'RACE', 'LCID', 'RIVN', 'NIO', 'LI'
        }
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image to extract tickers and context.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with extracted information
        """
        try:
            # Check if the image exists
            if not os.path.exists(image_path):
                return {'error': f"Image file not found: {image_path}"}
            
            # Extract text from image
            extracted_text = self._extract_text_from_image(image_path)
            if not extracted_text:
                return {'error': "No text could be extracted from the image"}
            
            # Detect platform
            platform = self._detect_platform(extracted_text)
            
            # Extract tickers based on platform
            tickers = self._extract_tickers(extracted_text, platform)
            
            # Extract context for each ticker
            ticker_context = self._extract_context(extracted_text, tickers, platform)
            
            return {
                'platform': platform,
                'tickers': tickers,
                'ticker_context': ticker_context,
                'extracted_text': extracted_text
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {'error': str(e)}
    
    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using OCR."""
        try:
            if not TESSERACT_AVAILABLE:
                logger.warning("Tesseract not available. Using mock OCR implementation.")
                # Mock implementation for testing without Tesseract
                return self._mock_ocr(image_path)
            
            # Open the image
            image = Image.open(image_path)
            
            # Preprocess the image for better OCR results
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Increase contrast
            # This is a simple contrast enhancement; more sophisticated methods could be used
            enhancer = Image.ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                temp_path = temp.name
                image.save(temp_path)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return ""
    
    def _mock_ocr(self, image_path: str) -> str:
        """Mock OCR implementation for testing without Tesseract."""
        # This would be replaced with actual OCR in production
        # For now, return some sample text based on the image filename
        filename = os.path.basename(image_path).lower()
        
        if 'danelfin' in filename:
            return """
            Top Stocks - Danelfin AI
            
            AAPL - Apple Inc.
            Score: 9.2/10
            Target: $198.45
            Potential: +15.3%
            
            MSFT - Microsoft Corp.
            Score: 8.7/10
            Target: $420.10
            Potential: +12.1%
            
            NVDA - NVIDIA Corp.
            Score: 9.5/10
            Target: $950.00
            Potential: +22.4%
            """
        elif 'finchat' in filename:
            return """
            Finchat.io - Market Movers
            
            $TSLA +5.2%
            Price: $245.67
            Volume: 32.5M
            
            $AMD +3.1%
            Price: $178.90
            Volume: 45.2M
            
            $META -1.2%
            Price: $472.30
            Volume: 18.7M
            """
        else:
            return """
            Stock Watchlist
            
            AAPL $172.50 +1.2%
            MSFT $375.20 -0.5%
            GOOGL $142.30 +0.8%
            AMZN $180.10 +2.1%
            TSLA $245.67 +5.2%
            """
    
    def _detect_platform(self, text: str) -> str:
        """Detect the platform from the extracted text."""
        text_lower = text.lower()
        
        if 'danelfin' in text_lower:
            return 'danelfin'
        elif 'finchat' in text_lower:
            return 'finchat'
        else:
            return 'generic'
    
    def _extract_tickers(self, text: str, platform: str) -> List[Tuple[str, float]]:
        """
        Extract ticker symbols from text.
        
        Returns:
            List of tuples (ticker, confidence)
        """
        # Get the appropriate pattern for the platform
        pattern = self.platform_patterns.get(platform, self.platform_patterns['generic'])
        ticker_pattern = pattern['ticker_pattern']
        
        # Find all potential tickers
        potential_tickers = re.findall(ticker_pattern, text)
        
        # Filter and score tickers
        verified_tickers = []
        for ticker in potential_tickers:
            # Remove $ prefix if present
            if ticker.startswith('$'):
                ticker = ticker[1:]
            
            # Skip if it's a common word
            if ticker in self.common_words:
                continue
            
            # Calculate confidence score
            confidence = self._calculate_confidence(ticker, text)
            
            # Only include if confidence is above threshold
            if confidence > 0.5:
                verified_tickers.append((ticker, confidence))
        
        # Sort by confidence (highest first)
        verified_tickers.sort(key=lambda x: x[1], reverse=True)
        
        return verified_tickers
    
    def _calculate_confidence(self, ticker: str, text: str) -> float:
        """Calculate confidence score for a ticker."""
        # Base confidence
        confidence = 0.5
        
        # Increase confidence if it's in our list of valid tickers
        if ticker in self.valid_tickers:
            confidence += 0.3
        
        # Increase confidence if it's followed by Inc, Corp, etc.
        if re.search(rf"{ticker}\s+(Inc|Corp|Corporation|Company)", text):
            confidence += 0.1
        
        # Increase confidence if it's preceded by $ or followed by :
        if re.search(rf"(\$|^|\s){ticker}(\s|:|$)", text):
            confidence += 0.1
        
        # Decrease confidence for very short tickers (more likely to be false positives)
        if len(ticker) == 1:
            confidence -= 0.2
        
        # Cap confidence at 1.0
        return min(confidence, 1.0)
    
    def _extract_context(self, text: str, tickers: List[Tuple[str, float]], platform: str) -> Dict[str, Dict[str, Any]]:
        """Extract context information for each ticker."""
        context = {}
        
        # Get platform-specific context patterns
        patterns = self.platform_patterns.get(platform, self.platform_patterns['generic'])
        context_patterns = patterns['context_patterns']
        
        for ticker, _ in tickers:
            # Find the section of text that likely contains information about this ticker
            ticker_section = self._find_ticker_section(text, ticker)
            
            ticker_context = {}
            
            # Extract context based on platform-specific patterns
            for pattern in context_patterns:
                matches = re.search(pattern, ticker_section)
                if matches:
                    # Extract the key from the pattern (e.g., "Score", "Target", etc.)
                    key = pattern.split(r'\s*')[0].replace('r', '').replace(':', '')
                    ticker_context[key] = matches.group(1)
            
            context[ticker] = ticker_context
        
        return context
    
    def _find_ticker_section(self, text: str, ticker: str) -> str:
        """Find the section of text that contains information about a ticker."""
        # Split text into lines
        lines = text.split('\n')
        
        # Find the line containing the ticker
        ticker_line_idx = -1
        for i, line in enumerate(lines):
            if ticker in line:
                ticker_line_idx = i
                break
        
        if ticker_line_idx == -1:
            return ""
        
        # Extract a few lines before and after the ticker line
        start_idx = max(0, ticker_line_idx - 2)
        end_idx = min(len(lines), ticker_line_idx + 5)
        
        return '\n'.join(lines[start_idx:end_idx])
