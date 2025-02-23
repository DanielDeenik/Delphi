
from fastapi import APIRouter, HTTPException
from typing import Dict
from src.models.omni_parser import OmniParser

router = APIRouter()
parser = OmniParser()

@router.get("/analyze/{symbol}")
async def analyze_symbol(symbol: str) -> Dict:
    """Analyze a symbol using OmniParser"""
    try:
        twitter_data = parser.analyze_twitter_sentiment(symbol)
        demand_data = parser.track_product_demand(symbol)
        
        return {
            "symbol": symbol,
            "social_sentiment": twitter_data,
            "demand_metrics": demand_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/earnings")
async def analyze_earnings(text: str) -> Dict:
    """Analyze earnings call text"""
    try:
        return parser.analyze_earnings_call(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
