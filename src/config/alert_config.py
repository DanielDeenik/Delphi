
import os

ALERT_CONFIG = {
    'min_trend_score': float(os.getenv('MIN_TREND_SCORE', '0.7')),
    'min_validation_score': float(os.getenv('MIN_VALIDATION_SCORE', '0.6')),
    'min_institutional_score': float(os.getenv('MIN_INSTITUTIONAL_SCORE', '0.5')),
    
    'slack': {
        'enabled': bool(os.getenv('SLACK_ENABLED', 'True')),
        'webhook_url': os.getenv('SLACK_WEBHOOK_URL', ''),
        'channel': os.getenv('SLACK_CHANNEL', '#trading-alerts')
    },
    
    'discord': {
        'enabled': bool(os.getenv('DISCORD_ENABLED', 'True')),
        'webhook_url': os.getenv('DISCORD_WEBHOOK_URL', ''),
        'channel_id': os.getenv('DISCORD_CHANNEL_ID', '')
    },
    
    'telegram': {
        'enabled': bool(os.getenv('TELEGRAM_ENABLED', 'True')),
        'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
        'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
    },
    
    'email': {
        'enabled': bool(os.getenv('EMAIL_ENABLED', 'True')),
        'smtp_host': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('SMTP_PORT', '587')),
        'username': os.getenv('EMAIL_USERNAME', ''),
        'password': os.getenv('EMAIL_PASSWORD', ''),
        'recipients': os.getenv('EMAIL_RECIPIENTS', '').split(',')
    }
}
