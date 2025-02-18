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
# Required for cloud storage
export GOOGLE_CLOUD_PROJECT=your_project_id
```

## Running Locally

1. Start the Streamlit application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:5000`

## Docker Development

1. Build the Docker image:
```bash
docker build -t rag-analysis:latest .
```

2. Run the container locally:
```bash
docker run -p 8080:8080 \
  -e ALPHA_VANTAGE_API_KEY=your_key \
  -e GOOGLE_CLOUD_PROJECT=your_project \
  rag-analysis:latest
```

## Google Cloud Deployment

1. Install Google Cloud SDK and kubectl:
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
# Install kubectl
gcloud components install kubectl
```

2. Configure Google Cloud:
```bash
gcloud init
gcloud config set project YOUR_PROJECT_ID
gcloud auth configure-docker
```

3. Enable required APIs:
```bash
gcloud services enable container.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable storage.googleapis.com
```

4. Create GKE cluster:
```bash
gcloud container clusters create rag-analysis-cluster \
  --num-nodes=3 \
  --machine-type=e2-standard-2 \
  --region=us-central1
```

5. Build and push Docker image:
```bash
docker build -t gcr.io/${GOOGLE_CLOUD_PROJECT}/rag-analysis:latest .
docker push gcr.io/${GOOGLE_CLOUD_PROJECT}/rag-analysis:latest
```

6. Create secrets:
```bash
kubectl create secret generic api-secrets \
  --from-literal=openai-api-key=$OPENAI_API_KEY \
  --from-literal=alpha-vantage-api-key=$ALPHA_VANTAGE_API_KEY \
  --from-literal=google-cloud-project=$GOOGLE_CLOUD_PROJECT
```

7. Deploy to Kubernetes:
```bash
kubectl apply -f k8s/rag-analysis-deployment.yaml
```

8. Monitor deployment:
```bash
kubectl get pods
kubectl get services
```

The LoadBalancer service will provide an external IP for accessing the application.

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