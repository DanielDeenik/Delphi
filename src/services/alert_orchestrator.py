
import logging
from typing import Dict, List
from datetime import datetime
import requests
from src.models.omni_parser import OmniParser
from src.models.real_world_validator import RealWorldValidator
from src.models.institutional_tracker import InstitutionalTracker
from src.utils.alerts import AlertSystem

logger = logging.getLogger(__name__)

class AlertOrchestrator:
    def __init__(self):
        self.omni_parser = OmniParser()
        self.validator = RealWorldValidator()
        self.inst_tracker = InstitutionalTracker()
        self.alert_system = AlertSystem()
        
    async def process_trend(self, trend_data: Dict) -> Dict:
        """Process trend through validation pipeline"""
        try:
            # Extract and rank trend
            trend_score = self.omni_parser.track_product_demand(trend_data['symbol'])
            
            # Validate with real-world data
            validation = self.validator.validate_trend(trend_data)
            
            # Check institutional flow
            inst_data = self.inst_tracker.analyze_institutional_flow(trend_data['symbol'])
            
            alert_data = {
                'timestamp': datetime.now(),
                'symbol': trend_data['symbol'],
                'trend_score': trend_score['demand_score'],
                'validation_score': validation['confidence'],
                'institutional_sentiment': inst_data['sentiment'],
                'alert_type': self._determine_alert_type(trend_score, validation, inst_data)
            }
            
            if self._should_send_alert(alert_data):
                await self._distribute_alert(alert_data)
                
            return alert_data
            
        except Exception as e:
            logger.error(f"Error processing trend: {str(e)}")
            return {}
            
    def _determine_alert_type(self, trend_score: Dict, validation: Dict, inst_data: Dict) -> str:
        """Determine alert type based on analysis"""
        if (trend_score['demand_score'] > 0.7 and 
            validation['confidence'] > 0.6 and 
            inst_data['sentiment'] > 0):
            return 'BUY_SIGNAL'
        elif (trend_score['demand_score'] < 0.3 and 
              validation['confidence'] > 0.6 and 
              inst_data['sentiment'] < 0):
            return 'SELL_SIGNAL'
        return 'WATCH'
        
    def _should_send_alert(self, alert_data: Dict) -> bool:
        """Determine if alert should be sent"""
        return (
            alert_data['trend_score'] > 0.7 and
            alert_data['validation_score'] > 0.6 and
            alert_data['alert_type'] in ['BUY_SIGNAL', 'SELL_SIGNAL']
        )
        
    async def _distribute_alert(self, alert_data: Dict):
        """Distribute alert to all configured channels"""
        message = self._format_alert_message(alert_data)
        
        # Send to all configured channels
        await self._send_slack_alert(message)
        await self._send_discord_alert(message)
        await self._send_telegram_alert(message)
        await self._send_email_alert(message)
        
    def _format_alert_message(self, alert_data: Dict) -> str:
        """Format alert message for distribution"""
        return f"""🚨 SOCIAL ARBITRAGE ALERT
Symbol: {alert_data['symbol']}
Type: {alert_data['alert_type']}
Trend Score: {alert_data['trend_score']:.2f}
Institutional Sentiment: {alert_data['institutional_sentiment']:.2f}
Timestamp: {alert_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"""
