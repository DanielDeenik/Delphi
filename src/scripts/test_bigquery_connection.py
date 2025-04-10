"""
Script to test BigQuery connectivity.
"""
import os
import logging
from google.cloud import bigquery
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_bigquery_connection():
    """Test BigQuery connectivity."""
    # Get project ID from environment variable
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
        return False
    
    try:
        # Initialize BigQuery client
        client = bigquery.Client(project=project_id)
        logger.info(f"Successfully initialized BigQuery client for project: {project_id}")
        
        # Run a simple query
        query = "SELECT 1 as test"
        query_job = client.query(query)
        results = query_job.result()
        
        # Check results
        for row in results:
            logger.info(f"Query result: {row.test}")
        
        logger.info("BigQuery connection test successful")
        return True
    
    except Exception as e:
        logger.error(f"Error testing BigQuery connection: {str(e)}")
        return False

def main():
    """Main function."""
    success = test_bigquery_connection()
    
    if success:
        logger.info("BigQuery connection test completed successfully")
    else:
        logger.error("BigQuery connection test failed")

if __name__ == "__main__":
    main()
