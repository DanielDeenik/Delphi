git clone <repository-url>
cd <repository-name>
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Variables**
Create a `.env` file with the following:
```
OPENAI_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
```

4. **Run the Application**
```bash
streamlit run app.py
```

## Google Colab Setup

1. Clone this notebook: [Colab Notebook Link]
2. Upload the following core files to your Colab session:
   - src/models/*
   - src/data/*
   - src/utils/*

3. Run the provided setup cell to install dependencies:
```python
!pip install -q streamlit sentence-transformers faiss-cpu tensorflow numpy pandas plotly
!pip install -q torch torchvision torchaudio
!pip install -q transformers scikit-learn xgboost
```

4. Execute the analysis notebooks in the `notebooks/` directory

## Cloud Deployment

### Kubernetes Deployment
1. Install kubectl and configure your cluster
2. Apply the Kubernetes manifests:
```bash
kubectl apply -f k8s/