
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class OutlierDetectionBot:
    """Detects market outliers using SVD and clustering"""
    
    def __init__(self, n_components: int = 3):
        self.svd = TruncatedSVD(n_components=n_components)
        self.dbscan = DBSCAN(eps=0.3, min_samples=5)
        
    def detect_outliers(self, market_data: np.ndarray) -> Dict:
        """Detect outliers using SVD and DBSCAN"""
        try:
            # Apply SVD for dimensionality reduction
            reduced_data = self.svd.fit_transform(market_data)
            
            # Detect clusters and outliers
            clusters = self.dbscan.fit_predict(reduced_data)
            outliers = clusters == -1
            
            return {
                'outlier_indices': np.where(outliers)[0],
                'explained_variance': self.svd.explained_variance_ratio_,
                'n_outliers': sum(outliers),
                'confidence_score': self._calculate_confidence(reduced_data, outliers)
            }
            
        except Exception as e:
            logger.error(f"Error in outlier detection: {str(e)}")
            return {}
            
    def _calculate_confidence(self, data: np.ndarray, outliers: np.ndarray) -> float:
        """Calculate confidence score for outlier detection"""
        if len(outliers) == 0:
            return 0.0
        
        inlier_density = np.mean(self.dbscan.core_sample_indices_ is not None)
        outlier_distance = np.mean(data[outliers])
        return float(inlier_density * outlier_distance)
