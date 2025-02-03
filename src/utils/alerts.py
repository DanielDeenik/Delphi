from datetime import datetime

class AlertSystem:
    def __init__(self):
        self.alerts = []
        
    def generate_volume_alerts(self, current_data, historical_data):
        """Generate alerts based on volume analysis"""
        alerts = []
        
        # Volume Spike Alert
        current_volume = current_data['Volume'].iloc[-1]
        avg_volume = historical_data['Volume'].mean()
        if current_volume > 2 * avg_volume:
            alerts.append({
                'timestamp': datetime.now(),
                'type': 'VOLUME_SPIKE',
                'message': f'Unusual volume detected: {current_volume:.0f} vs avg {avg_volume:.0f}',
                'severity': 'HIGH'
            })
        
        # Low Volume Alert
        if current_volume < 0.5 * avg_volume:
            alerts.append({
                'timestamp': datetime.now(),
                'type': 'LOW_VOLUME',
                'message': 'Volume significantly below average',
                'severity': 'MEDIUM'
            })
        
        return alerts
    
    def clear_old_alerts(self, max_age_hours=24):
        """Clear alerts older than specified hours"""
        current_time = datetime.now()
        self.alerts = [
            alert for alert in self.alerts
            if (current_time - alert['timestamp']).total_seconds() / 3600 < max_age_hours
        ]
