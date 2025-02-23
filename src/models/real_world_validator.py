from src.config.validator_config import VALIDATOR_WEIGHTS, VALIDATION_THRESHOLDS

def calculate_adoption_score(retail_score, traffic_score, review_score):
        weights = VALIDATOR_WEIGHTS
        score = (retail_score * weights['retail']) + \
                (traffic_score * weights['traffic']) + \
                (review_score * weights['reviews'])
        
        if score > VALIDATION_THRESHOLDS['strong']:
            return 'STRONG_ADOPTION'
        elif score > VALIDATION_THRESHOLDS['moderate']:
            return 'MODERATE_ADOPTION'
        elif score > VALIDATION_THRESHOLDS['early']:
            return 'EARLY_ADOPTION'
        else:
            return 'NO_ADOPTION'