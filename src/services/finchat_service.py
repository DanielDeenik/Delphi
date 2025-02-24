
import aiohttp
import logging
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)

class FinChatService:
    def __init__(self):
        self.api_key = os.environ.get('FINCHAT_API_KEY')
        if not self.api_key:
            raise ValueError("FINCHAT_API_KEY environment variable not set")
        self.base_url = 'https://api.finchat.io/v1'

    async def get_market_sentiment(self, symbol: str) -> Dict:
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            async with session.get(f"{self.base_url}/market/sentiment/{symbol}", headers=headers) as response:
                return await response.json()

    async def get_social_metrics(self, symbol: str) -> Dict:
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            async with session.get(f"{self.base_url}/social/metrics/{symbol}", headers=headers) as response:
                return await response.json()
