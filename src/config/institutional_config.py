
import os
from typing import Dict

FILING_THRESHOLDS = {
    'significant_holding': float(os.getenv('FILING_SIGNIFICANT_HOLDING', '0.05')),
    'major_change': float(os.getenv('FILING_MAJOR_CHANGE', '0.25')),
    'insider_significance': float(os.getenv('INSIDER_SIGNIFICANCE', '0.01'))
}

OPTIONS_THRESHOLDS = {
    'unusual_volume': float(os.getenv('OPTIONS_UNUSUAL_VOLUME', '2.0')),
    'significant_oi': float(os.getenv('OPTIONS_SIGNIFICANT_OI', '1000')),
    'put_call_ratio': float(os.getenv('OPTIONS_PUT_CALL_RATIO', '0.7'))
}

SHORT_THRESHOLDS = {
    'high_short_interest': float(os.getenv('HIGH_SHORT_INTEREST', '0.20')),
    'days_to_cover': float(os.getenv('DAYS_TO_COVER_THRESHOLD', '5'))
}
