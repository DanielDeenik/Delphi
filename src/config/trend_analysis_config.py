
import os
from typing import Dict

TREND_THRESHOLDS = {
    'VIRAL_THRESHOLD': float(os.getenv('VIRAL_THRESHOLD', '80')),
    'EARLY_THRESHOLD': float(os.getenv('EARLY_THRESHOLD', '70')),
    'FADING_THRESHOLD': float(os.getenv('FADING_THRESHOLD', '30'))
}

TREND_WEIGHTS = {
    'SOCIAL_WEIGHT': float(os.getenv('SOCIAL_WEIGHT', '0.6')),
    'DEMAND_WEIGHT': float(os.getenv('DEMAND_WEIGHT', '0.4'))
}

TREND_STAGES = {
    'EARLY': 'Early Discovery',
    'VIRAL': 'Viral Expansion',
    'FADING': 'Fading Trend',
    'UNDEFINED': 'Undefined'
}
