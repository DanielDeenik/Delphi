#!/bin/bash

# Exit on any error
set -e

# Deploy to Google Cloud
echo "Deploying to Google Cloud..."
gcloud app deploy app.yaml --quiet

echo "Deployment completed successfully!"
