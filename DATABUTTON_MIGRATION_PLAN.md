# Databutton & Firebase Migration Plan

## Phase 1: Initial Setup & Configuration (Week 1)

### 1.1 Environment Setup
```bash
# Required packages
pip install databutton firebase-admin pyrebase4 streamlit plotly pandas numpy tensorflow-cpu
```

### 1.2 Firebase Project Setup
1. Create new Firebase project
2. Enable Authentication methods:
   - Email/Password
   - Google OAuth (optional)
3. Create Firestore database
4. Download service account credentials

### 1.3 Environment Variables
```plaintext
FIREBASE_API_KEY=your_api_key
FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_STORAGE_BUCKET=your_project.appspot.com
FIREBASE_MESSAGING_SENDER_ID=your_sender_id
FIREBASE_APP_ID=your_app_id
FIREBASE_MEASUREMENT_ID=your_measurement_id
FIREBASE_DATABASE_URL=your_database_url
ALPHA_VANTAGE_API_KEY=your_alphavantage_key
```

## Phase 2: Code Structure (Week 1-2)

### 2.1 Project Structure
```
/
├── pages/
│   ├── __init__.py
│   ├── dashboard.py
│   ├── analysis.py
│   ├── signals.py
│   └── settings.py
├── src/
│   ├── models/
│   ├── services/
│   └── utils/
├── config.py
└── main.py
```

### 2.2 Core Configuration (config.py)
```python
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

## Phase 3: Authentication Implementation (Week 2)

### 3.1 Auth Service (src/services/auth_service.py)
```python
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

## Phase 4: Data Layer Migration (Week 3)

### 4.1 Firestore Schema
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

### 4.2 Data Service
```python
# src/services/data_service.py
import databutton as db
from firebase_admin import firestore
from datetime import datetime
from typing import Dict, List

class DataService:
    def __init__(self):
        self.db = firestore.client()
        
    def store_market_data(self, symbol: str, data: Dict):
        # Cache in Databutton storage
        db.storage.put(f"market_data_{symbol}", data)
        
        # Store in Firestore
        doc_ref = self.db.collection('market_data').document(symbol)
        doc_ref.set({
            'last_updated': datetime.now().isoformat(),
            'data': data
        })
        
    def get_market_data(self, symbol: str) -> Dict:
        # Try Databutton cache first
        cached_data = db.storage.get(f"market_data_{symbol}")
        if cached_data is not None:
            return cached_data
            
        # Fall back to Firestore
        doc_ref = self.db.collection('market_data').document(symbol)
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else None
```

## Phase 5: UI Migration (Week 4)

### 5.1 Main Dashboard (pages/dashboard.py)
```python
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

### 5.2 Analysis Page (pages/analysis.py)
```python
import databutton as db
import streamlit as st
from src.services.volume_analysis_service import VolumeAnalysisService
from src.services.auth_service import require_auth

@require_auth
def analysis_page():
    st.title("Volume Analysis")
    
    # Get user preferences
    user = db.get_user()
    preferences = db.storage.get(f"preferences_{user.id}", default={})
    
    # Analysis components
    volume_service = VolumeAnalysisService()
    
    # Display analysis
    if symbol := st.selectbox("Select Symbol", preferences.get('watchlist', [])):
        data = volume_service.analyze_volume_patterns(symbol)
        display_volume_analysis(data)
```

## Phase 6: Real-time Updates (Week 5)

### 6.1 Firestore Triggers
```python
# functions/src/index.ts
import * as functions from 'firebase-functions';
import { calculateVolumeSignals } from './volume-analysis';

export const onMarketDataUpdate = functions.firestore
    .document('market_data/{symbol}')
    .onCreate(async (snap, context) => {
        const data = snap.data();
        const signals = await calculateVolumeSignals(data);
        
        await admin.firestore()
            .collection('signals')
            .add({
                timestamp: admin.firestore.FieldValue.serverTimestamp(),
                symbol: context.params.symbol,
                signals: signals
            });
    });
```

### 6.2 Real-time Updates in UI
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

## Phase 7: Performance Optimization (Week 6)

### 7.1 Caching Strategy
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

## Security Considerations

### Firebase Security Rules
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

## Deployment Checklist

1. Environment Setup:
   - [ ] Install required packages
   - [ ] Configure Firebase project
   - [ ] Set up environment variables
   - [ ] Initialize Databutton app

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
   - Document deployment steps
   - Keep configuration backups

3. Monitoring:
```python
def health_check():
    checks = {
        'firebase': check_firebase_connection(),
        'alpha_vantage': check_alpha_vantage_api(),
        'ml_models': check_model_status()
    }
    return all(checks.values())
```
