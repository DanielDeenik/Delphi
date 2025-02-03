import os
from typing import Dict

# Firebase configuration using provided web credentials
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyCQ6DR8OcVTcT1dje6Gtcuv6YgAlTlYcjE",
    "authDomain": "delphi-31d1a.firebaseapp.com",
    "projectId": "delphi-31d1a",
    "storageBucket": "delphi-31d1a.firebasestorage.app",
    "messagingSenderId": "158361021905",
    "appId": "1:158361021905:web:63832f8a2c83c736cd44f0",
    "measurementId": "G-EKQXY1M25M",
    "databaseURL": "https://delphi-31d1a-default-rtdb.firebaseio.com"  # Added database URL
}

# Initialize Firebase app without admin SDK
firebase_app = None