
import os
from typing import Dict

EXECUTION_THRESHOLDS = {
    'min_signal_strength': float(os.getenv('MIN_SIGNAL_STRENGTH', '0.7')),
    'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '0.1')),
    'trend_reversal_threshold': float(os.getenv('TREND_REVERSAL_THRESHOLD', '0.8'))
}

SIGNAL_WEIGHTS = {
    'volume': float(os.getenv('VOLUME_WEIGHT', '0.3')),
    'options': float(os.getenv('OPTIONS_WEIGHT', '0.4')),
    'institutional': float(os.getenv('INSTITUTIONAL_WEIGHT', '0.3'))
}
