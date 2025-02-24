#!/bin/bash

# Exit on any error
set -e

# Check required environment variables
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "Error: GOOGLE_CLOUD_PROJECT not set"
    exit 1
fi

if [ -z "$GOOGLE_CLOUD_CREDENTIALS" ]; then
    echo "Error: GOOGLE_CLOUD_CREDENTIALS not set"
    exit 1
fi

# Authenticate with Google Cloud
echo "$GOOGLE_CLOUD_CREDENTIALS" > /tmp/gcloud-key.json
gcloud auth activate-service-account --key-file=/tmp/gcloud-key.json
rm /tmp/gcloud-key.json

# Set project
gcloud config set project $GOOGLE_CLOUD_PROJECT

# Deploy to App Engine
echo "Deploying to App Engine..."
gcloud app deploy app.yaml --quiet

echo "Deployment completed successfully!"

# Display service URL
gcloud app browse