# Financial Intelligence Platform

An advanced Retrieval-Augmented Generation (RAG) AI-powered financial intelligence platform that provides sophisticated, data-driven trading insights and adaptive market analysis.

## Features

- Python-based RAG AI system
- Streamlit web interface with dynamic loading
- FAISS vector search
- Sentence Transformer embeddings
- GPT-4 powered insight generation
- Multi-factor market sentiment analysis
- Alpha Vantage ticker integration

## Technology Stack

- Python 3.11
- Streamlit
- TensorFlow
- FAISS
- Sentence Transformers
- Pandas & NumPy
- Scikit-learn

## Cloud Deployment Setup

### 1. GitHub Integration
1. Create a new GitHub repository
2. Initialize git and push the code:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/financial-intelligence-platform.git
git push -u origin main
```

### 2. Cursor IDE Setup
1. Install Cursor IDE from https://cursor.sh
2. Open Cursor and select "Clone Repository"
3. Enter your GitHub repository URL
4. Configure Cursor's AI features for enhanced development

### 3. Google Cloud Setup (Cost-Optimized)
1. Install Google Cloud SDK
2. Initialize Google Cloud project:
```bash
gcloud init
gcloud app create
```
3. Set up authentication:
```bash
gcloud auth login
gcloud auth application-default login
```
4. Deploy the application:
```bash
./deploy.sh
```

Note: The app is configured for minimal cost using:
- F1 instance class (shared CPU)
- Conservative auto-scaling (1-2 instances)
- Efficient memory utilization
- Minimal concurrent requests

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-intelligence-platform.git
cd financial-intelligence-platform
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory and add your API keys:
```
OPENAI_API_KEY=your_openai_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

4. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
├── src/
│   ├── models/           # ML models and predictors
│   ├── services/         # Business logic and services
│   ├── utils/           # Utility functions
│   └── scripts/         # Automation scripts
├── assets/              # Static assets
├── tests/              # Test suite
└── app.py              # Main application entry
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.