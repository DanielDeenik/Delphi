import logging
import sys
import os
import asyncio

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.services.scheduler_service import SchedulerService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('nightly_processing.log')
    ]
)

logger = logging.getLogger(__name__)

async def main():
    try:
        logger.info("Starting nightly processing routine...")

        # Initialize scheduler service
        scheduler = SchedulerService()

        # Run nightly processing
        results = await scheduler.run_nightly_processing()

        # Generate and log summary
        summary = scheduler.get_processing_summary(results)
        logger.info("\n" + summary)

        logger.info("Nightly processing completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error during nightly processing: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))