
import os
from typing import Dict

VALIDATOR_WEIGHTS = {
    'retail': float(os.getenv('VALIDATOR_RETAIL_WEIGHT', '0.4')),
    'traffic': float(os.getenv('VALIDATOR_TRAFFIC_WEIGHT', '0.35')), 
    'reviews': float(os.getenv('VALIDATOR_REVIEWS_WEIGHT', '0.25'))
}

VALIDATION_THRESHOLDS = {
    'strong': float(os.getenv('VALIDATION_STRONG_THRESHOLD', '0.7')),
    'moderate': float(os.getenv('VALIDATION_MODERATE_THRESHOLD', '0.5')),
    'early': float(os.getenv('VALIDATION_EARLY_THRESHOLD', '0.3'))
}
