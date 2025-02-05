git clone <repository-url>
cd financial-intelligence-platform
```

2. Install required packages:
```bash
pip install streamlit plotly pandas numpy scikit-learn tensorflow-cpu google-cloud-storage google-cloud-bigquery pandas-gbq hmmlearn
```

3. Set up environment variables:
```bash
# Required for market data
export ALPHA_VANTAGE_API_KEY=your_api_key

# Required for cloud storage (when deploying to Google Cloud)
export GOOGLE_CLOUD_PROJECT=your_project_id
```

## Running Locally

1. Start the Streamlit application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:5000`

## Google Cloud Deployment

1. Install Google Cloud SDK:
- Follow instructions at https://cloud.google.com/sdk/docs/install

2. Initialize Google Cloud project:
```bash
gcloud init
gcloud config set project YOUR_PROJECT_ID
```

3. Enable required APIs:
```bash
gcloud services enable bigquery.googleapis.com
gcloud services enable storage.googleapis.com
```

4. Set up BigQuery dataset and Storage bucket:
```bash
# Create BigQuery dataset
bq mk --dataset market_data

# Create Cloud Storage bucket
gsutil mb gs://market_data_cold_storage
```

5. Deploy the application:
```bash
gcloud app deploy app.yaml
```

## Project Structure

```
src/
├── models/             # ML and AI Models
│   ├── rag_volume_analyzer.py     # Main volume analysis model
│   ├── hmm_regime_classifier.py   # Market regime classification
│   ├── lstm_price_predictor.py    # Price prediction model
│   └── ml_volume_analyzer.py      # ML-based volume analysis
├── services/           # Core Services
│   ├── timeseries_storage_service.py  # Data storage management
│   ├── volume_analysis_service.py     # Volume analysis service
│   └── sentiment_analysis_service.py  # Sentiment analysis
└── data/              # Data Management
    └── alpha_vantage_client.py    # Market data fetching