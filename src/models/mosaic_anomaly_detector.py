
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import tensorflow as tf
from typing import Dict, List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MosaicAnomalyDetector:
    """Detects market anomalies using Mosaic Theory approach"""
    
    def __init__(self, n_components: int = 3):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.svd = TruncatedSVD(n_components=n_components)
        self.dbscan = DBSCAN(eps=0.3, min_samples=5)
        self.autoencoder = self._build_autoencoder()
        
    def _build_autoencoder(self) -> tf.keras.Model:
        """Build autoencoder for anomaly detection"""
        encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu')
        ])
        
        decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='sigmoid')
        ])
        
        return tf.keras.Sequential([encoder, decoder])
        
    def detect_anomalies(self, data: Dict) -> Dict:
        """Detect market anomalies using multiple techniques"""
        try:
            # Extract features
            features = self._extract_features(data)
            scaled_features = self.scaler.fit_transform(features)
            
            # PCA analysis
            pca_results = self._analyze_pca(scaled_features)
            
            # SVD analysis
            svd_results = self._analyze_svd(scaled_features)
            
            # Autoencoder reconstruction
            encoded_data = self._encode_data(scaled_features)
            
            # Cluster analysis
            clusters = self.dbscan.fit_predict(encoded_data)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'pca_signals': pca_results,
                'svd_signals': svd_results,
                'cluster_anomalies': np.where(clusters == -1)[0],
                'reconstruction_error': self._calculate_reconstruction_error(scaled_features),
                'confidence_score': self._calculate_confidence(clusters, encoded_data)
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return {}
            
    def _extract_features(self, data: Dict) -> np.ndarray:
        """Extract features from multiple data sources"""
        return np.column_stack([
            data.get('sentiment_scores', []),
            data.get('earnings_surprises', []),
            data.get('price_returns', []),
            data.get('volume_changes', []),
            data.get('options_flow', [])
        ])
        
    def _analyze_pca(self, features: np.ndarray) -> Dict:
        """Analyze principal components"""
        transformed = self.pca.fit_transform(features)
        return {
            'components': self.pca.components_,
            'explained_variance': self.pca.explained_variance_ratio_,
            'transformed_data': transformed
        }
        
    def _analyze_svd(self, features: np.ndarray) -> Dict:
        """Analyze singular value decomposition"""
        transformed = self.svd.fit_transform(features)
        return {
            'singular_values': self.svd.singular_values_,
            'components': self.svd.components_,
            'transformed_data': transformed
        }
        
    def _encode_data(self, features: np.ndarray) -> np.ndarray:
        """Encode data using autoencoder"""
        return self.autoencoder.predict(features)
        
    def _calculate_reconstruction_error(self, features: np.ndarray) -> np.ndarray:
        """Calculate reconstruction error for anomaly detection"""
        reconstructed = self.autoencoder.predict(features)
        return np.mean(np.power(features - reconstructed, 2), axis=1)
        
    def _calculate_confidence(self, clusters: np.ndarray, encoded_data: np.ndarray) -> float:
        """Calculate confidence score for anomaly detection"""
        if len(clusters) == 0:
            return 0.0
            
        outlier_ratio = np.sum(clusters == -1) / len(clusters)
        feature_importance = np.mean(np.abs(self.pca.components_))
        return float(1.0 - outlier_ratio) * feature_importance
