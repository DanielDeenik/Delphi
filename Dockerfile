# Use Python 3.10 for better performance and features
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    FLASK_APP=/app/app.py \
    FLASK_ENV=production

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create directories for logs and status
RUN mkdir -p logs status data

# Expose port for Flask
EXPOSE 3000

# Create a script to choose between different entry points
RUN echo '#!/bin/bash\n\
if [ "$1" = "import" ]; then\n\
  shift\n\
  python run_time_series_import.py "$@"\n\
else\n\
  flask run --host=0.0.0.0 --port=3000\n\
fi' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (run the dashboard)
CMD ["dashboard"]
