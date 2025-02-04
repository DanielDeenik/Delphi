#!/bin/bash

# Exit on any error
set -e

# Authenticate using service account
if [ -n "$GOOGLE_CLOUD_CREDENTIALS" ]; then
    echo "Authenticating with service account..."
    echo "$GOOGLE_CLOUD_CREDENTIALS" > /tmp/gcloud-key.json
    gcloud auth activate-service-account --key-file=/tmp/gcloud-key.json
    rm /tmp/gcloud-key.json
else
    echo "Error: GOOGLE_CLOUD_CREDENTIALS not set"
    exit 1
fi

# Deploy to Google Cloud
echo "Deploying to Google Cloud..."
gcloud app deploy app.yaml --quiet

echo "Deployment completed successfully!"