"""
Logging Utilities

This module provides utilities for logging.
"""

import os
import logging
import sys
from typing import Optional

def get_logger(name: str, level: int = None, log_file: str = None) -> logging.Logger:
    """
    Get a logger with the specified name and level.
    
    Args:
        name: Logger name
        level: Logging level (if None, will try to load from environment)
        log_file: Path to log file (if None, will try to load from environment)
        
    Returns:
        logging.Logger: Logger instance
    """
    # Get level from environment if not provided
    if level is None:
        level_name = os.environ.get("LOG_LEVEL", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)
    
    # Get log file from environment if not provided
    if log_file is None:
        log_file = os.environ.get("LOG_FILE")
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler if no handlers exist
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Create file handler if log file is provided
        if log_file:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Create file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
    
    return logger
