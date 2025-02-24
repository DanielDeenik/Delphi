
from typing import Dict

PINECONE_CONFIG = {
    'dimension': 512,
    'metric': 'cosine',
    'pod_type': 'p1'
}

MODEL_CONFIG = {
    'lstm': {
        'layers': [64, 32],
        'dropout': 0.2,
        'batch_size': 32,
        'epochs': 100
    },
    'hmm': {
        'n_components': 3,
        'n_iter': 100
    },
    'rag': {
        'top_k': 5,
        'similarity_threshold': 0.7
    }
}
