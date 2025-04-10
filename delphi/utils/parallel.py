"""
Parallel Processing Utilities

This module provides utilities for parallel processing.
"""

import logging
from typing import List, Callable, Any, TypeVar, Generic
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

def parallel_process(items: List[T], func: Callable[[T], R], max_workers: int = 4, use_processes: bool = False) -> List[R]:
    """
    Process items in parallel.
    
    Args:
        items: List of items to process
        func: Function to apply to each item
        max_workers: Maximum number of workers
        use_processes: Whether to use processes instead of threads
        
    Returns:
        List[R]: List of results
    """
    # Choose executor based on use_processes
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    # Process items in parallel
    results = []
    with executor_class(max_workers=max_workers) as executor:
        # Create a dictionary mapping futures to items
        future_to_item = {
            executor.submit(func, item): item
            for item in items
        }
        
        # Process completed futures
        for future in future_to_item:
            item = future_to_item[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item {item}: {str(e)}")
    
    return results
