# Migration Plan: Databutton & Firebase Integration

## Phase 1: Initial Setup & Authentication (Week 1)

### 1.1 Environment Setup
```bash
# Required packages
pip install databutton firebase-admin pyrebase4 streamlit plotly pandas numpy tensorflow-cpu
```

### 1.2 Firebase Configuration
```python
# config.py
import os
from databutton import App
from firebase_admin import credentials, initialize_app

# Databutton configuration
app = App()

# Firebase configuration
FIREBASE_CONFIG = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL")
}

# Initialize Firebase Admin SDK
cred = credentials.Certificate("firebase-credentials.json")
firebase_app = initialize_app(cred)
```

### 1.3 Authentication Service
```python
# src/services/auth_service.py
import streamlit as st
import pyrebase
from typing import Optional, Dict
from config import FIREBASE_CONFIG

class FirebaseAuthService:
    def __init__(self):
        self.firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
        self.auth = self.firebase.auth()
        
    def sign_in(self, email: str, password: str) -> Optional[Dict]:
        try:
            user = self.auth.sign_in_with_email_and_password(email, password)
            return {
                'user_id': user['localId'],
                'email': user['email'],
                'token': user['idToken']
            }
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
            return None
            
    def sign_up(self, email: str, password: str) -> Optional[Dict]:
        try:
            user = self.auth.create_user_with_email_and_password(email, password)
            return {
                'user_id': user['localId'],
                'email': user['email'],
                'token': user['idToken']
            }
        except Exception as e:
            st.error(f"Registration failed: {str(e)}")
            return None
```

## Phase 2: Data Layer Migration (Week 2)

### 2.1 Firestore Database Schema
```python
# src/models/firestore_schema.py
from typing import TypedDict, List

class UserProfile(TypedDict):
    user_id: str
    email: str
    preferences: dict
    watchlist: List[str]
    alerts: List[dict]

class MarketData(TypedDict):
    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    indicators: dict
```

### 2.2 Data Service
```python
# src/services/data_service.py
from firebase_admin import firestore
from datetime import datetime
from typing import Dict, List

class FirestoreDataService:
    def __init__(self):
        self.db = firestore.client()
        
    def store_market_data(self, symbol: str, data: Dict):
        doc_ref = self.db.collection('market_data').document(symbol)
        doc_ref.set({
            'last_updated': datetime.now().isoformat(),
            'data': data
        })
        
    def get_market_data(self, symbol: str) -> Dict:
        doc_ref = self.db.collection('market_data').document(symbol)
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else None
```

## Phase 3: UI Migration (Week 3)

### 3.1 Databutton Pages Structure
```
pages/
├── __init__.py
├── dashboard.py
├── analysis.py
├── signals.py
└── settings.py
```

### 3.2 Main Dashboard
```python
# pages/dashboard.py
import databutton as db
import streamlit as st
from src.services.auth_service import require_auth
from src.services.trading_signal_service import TradingSignalService

@require_auth
def dashboard_page():
    st.title("Market Intelligence Dashboard")
    
    # User-specific data
    user = db.get_user()
    watchlist = db.storage.get(f"watchlist_{user.id}", default=[])
    
    # Display widgets
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Active Signals", len(watchlist))
    with col2:
        st.metric("Alert Count", len(alerts))
```

## Phase 4: Real-time Updates (Week 4)

### 4.1 Firebase Cloud Functions
```typescript
// functions/src/index.ts
import * as functions from 'firebase-functions';
import { calculateVolumeSignals } from './volume-analysis';

export const onMarketDataUpdate = functions.firestore
    .document('market_data/{symbol}')
    .onCreate(async (snap, context) => {
        const data = snap.data();
        const signals = await calculateVolumeSignals(data);
        
        // Store signals
        await admin.firestore()
            .collection('signals')
            .add({
                timestamp: admin.firestore.FieldValue.serverTimestamp(),
                symbol: context.params.symbol,
                signals: signals
            });
    });
```

### 4.2 Real-time Updates in UI
```python
# src/services/realtime_service.py
from firebase_admin import firestore
import streamlit as st
from typing import Callable

class RealtimeService:
    def __init__(self):
        self.db = firestore.client()
        
    def subscribe_to_signals(self, symbol: str, callback: Callable):
        def on_snapshot(doc_snapshot, changes, read_time):
            for doc in doc_snapshot:
                callback(doc.to_dict())
                
        self.db.collection('signals')\
            .where('symbol', '==', symbol)\
            .on_snapshot(on_snapshot)
```

## Phase 5: Testing & Optimization (Week 5)

### 5.1 Test Cases
```python
# tests/test_volume_analysis.py
import pytest
from src.services.volume_analysis_service import VolumeAnalysisService

def test_volume_spike_detection():
    service = VolumeAnalysisService()
    data = get_test_data()
    results = service.analyze_volume_patterns(data)
    assert len(results['predicted_spikes']) > 0
```

### 5.2 Performance Optimization
```python
# src/utils/caching.py
import databutton as db
from functools import lru_cache
from typing import Any

def cache_data(key: str, data: Any, ttl_seconds: int = 3600):
    db.storage.put(key, data, ttl=ttl_seconds)

@lru_cache(maxsize=100)
def get_cached_data(key: str) -> Any:
    return db.storage.get(key)
```

## Deployment Checklist

1. Environment Setup:
   - [ ] Install required packages
   - [ ] Configure Firebase project
   - [ ] Set up environment variables

2. Authentication:
   - [ ] Implement Firebase Auth
   - [ ] Create login/signup flows
   - [ ] Add session management

3. Data Migration:
   - [ ] Design Firestore schema
   - [ ] Migrate existing data
   - [ ] Set up backup procedures

4. UI Components:
   - [ ] Migrate Streamlit pages
   - [ ] Add Databutton components
   - [ ] Implement real-time updates

5. Testing:
   - [ ] Unit tests
   - [ ] Integration tests
   - [ ] Performance testing

## Security Considerations

1. Firebase Security Rules:
```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{userId} {
      allow read, write: if request.auth.uid == userId;
    }
    match /market_data/{symbol} {
      allow read: if request.auth != null;
      allow write: if false;  // Only backend can write
    }
  }
}
```

2. Environment Variables:
```bash
# Required environment variables
FIREBASE_API_KEY=
FIREBASE_AUTH_DOMAIN=
FIREBASE_PROJECT_ID=
FIREBASE_STORAGE_BUCKET=
FIREBASE_MESSAGING_SENDER_ID=
FIREBASE_APP_ID=
FIREBASE_MEASUREMENT_ID=
FIREBASE_DATABASE_URL=
ALPHA_VANTAGE_API_KEY=
```

## Rollback Plan

1. Data Backup:
```python
def backup_data():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    db.storage.export_collection(
        'market_data',
        f'backups/market_data_{timestamp}.json'
    )
```

2. Version Control:
- Maintain git tags for each deployment
- Document deployment steps and dependencies
- Keep configuration backups

## Monitoring & Maintenance

1. Health Checks:
```python
def health_check():
    checks = {
        'firebase': check_firebase_connection(),
        'alpha_vantage': check_alpha_vantage_api(),
        'ml_models': check_model_status()
    }
    return all(checks.values())
```

2. Error Handling:
```python
def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_error(e)
            notify_admin(e)
            return None
    return wrapper
```

## Performance Optimization Tips

1. Use Databutton's built-in caching:
```python
@db.cache(ttl_seconds=3600)
def fetch_market_data(symbol: str):
    return alpha_vantage.fetch_daily_adjusted(symbol)
```

2. Implement lazy loading for heavy computations:
```python
def lazy_load_ml_models():
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = load_ml_models()
    return st.session_state.ml_models
```

3. Optimize Firebase queries:
```python
# Add composite index for frequently used queries
# Add .limit() to pagination queries
# Use .select() to fetch only needed fields
```
