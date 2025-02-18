#!/bin/bash

# Exit on any error
set -e

# Check required environment variables
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "Error: GOOGLE_CLOUD_PROJECT not set"
    exit 1
fi

if [ -z "$ALPHA_VANTAGE_API_KEY" ]; then
    echo "Error: ALPHA_VANTAGE_API_KEY not set"
    exit 1
fi

# Authenticate using service account or user credentials
if [ -n "$GOOGLE_CLOUD_CREDENTIALS" ]; then
    echo "Authenticating with service account..."
    echo "$GOOGLE_CLOUD_CREDENTIALS" > /tmp/gcloud-key.json
    gcloud auth activate-service-account --key-file=/tmp/gcloud-key.json
    rm /tmp/gcloud-key.json
fi

# Parse deployment type argument
DEPLOY_TYPE=${1:-"app-engine"}  # Default to app-engine if not specified

case $DEPLOY_TYPE in
    "app-engine")
        echo "Deploying to App Engine..."
        gcloud app deploy app.yaml --quiet
        ;;

    "kubernetes")
        echo "Deploying to Google Kubernetes Engine..."

        # Build and push Docker image
        echo "Building and pushing Docker image..."
        docker build -t gcr.io/${GOOGLE_CLOUD_PROJECT}/rag-analysis:latest .
        docker push gcr.io/${GOOGLE_CLOUD_PROJECT}/rag-analysis:latest

        # Create GKE cluster if it doesn't exist
        if ! gcloud container clusters describe rag-analysis-cluster --region=us-central1 > /dev/null 2>&1; then
            echo "Creating GKE cluster..."
            gcloud container clusters create rag-analysis-cluster \
                --num-nodes=3 \
                --machine-type=e2-standard-2 \
                --region=us-central1
        fi

        # Get cluster credentials
        gcloud container clusters get-credentials rag-analysis-cluster --region=us-central1

        # Create secrets if they don't exist
        if ! kubectl get secret api-secrets > /dev/null 2>&1; then
            echo "Creating Kubernetes secrets..."
            kubectl create secret generic api-secrets \
                --from-literal=openai-api-key="$OPENAI_API_KEY" \
                --from-literal=alpha-vantage-api-key="$ALPHA_VANTAGE_API_KEY" \
                --from-literal=google-cloud-project="$GOOGLE_CLOUD_PROJECT"
        fi

        # Apply Kubernetes configurations
        echo "Applying Kubernetes configurations..."
        kubectl apply -f k8s/rag-analysis-deployment.yaml

        # Wait for deployment to complete
        echo "Waiting for deployment to complete..."
        kubectl rollout status deployment/rag-analysis-service
        ;;

    *)
        echo "Invalid deployment type. Use 'app-engine' or 'kubernetes'"
        exit 1
        ;;
esac

echo "Deployment completed successfully!"

# Display access information
if [ "$DEPLOY_TYPE" = "kubernetes" ]; then
    echo "Getting service external IP..."
    kubectl get service rag-analysis-service
fi