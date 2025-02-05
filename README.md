git clone <repository-url>
cd financial-intelligence-platform
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
ALPHA_VANTAGE_API_KEY=your_api_key
GOOGLE_CLOUD_PROJECT=your_project_id
```

3. Run the application:
```bash
streamlit run app.py
```

## Key Files and Structure

```
src/
├── models/
│   ├── rag_volume_analyzer.py     # Main volume analysis model
│   ├── hmm_regime_classifier.py   # Market regime classification
│   ├── lstm_price_predictor.py    # Price prediction model
│   └── ml_volume_analyzer.py      # ML-based volume analysis
├── services/
│   ├── timeseries_storage_service.py  # Data storage management
│   ├── volume_analysis_service.py     # Volume analysis service
│   └── sentiment_analysis_service.py  # Sentiment analysis
└── data/
    └── alpha_vantage_client.py    # Market data fetching