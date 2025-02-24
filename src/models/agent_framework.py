
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.last_run = None
        
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent's primary function"""
        pass
        
    def log_execution(self, result: Dict[str, Any]):
        """Log execution results"""
        self.last_run = datetime.now()
        logger.info(f"Agent {self.name} executed with result: {result}")
