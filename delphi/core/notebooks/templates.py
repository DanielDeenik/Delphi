"""
Notebook templates module for Delphi.

This module provides a class for notebook templates.
"""
from typing import Dict, List, Optional, Any, Union
import json
import logging
import re

# Configure logger
logger = logging.getLogger(__name__)

class NotebookTemplate:
    """Class for notebook templates."""
    
    def __init__(self, name: str, content: Dict[str, Any]):
        """Initialize the notebook template.
        
        Args:
            name: Template name
            content: Template content
        """
        self.name = name
        self.content = content
        
        logger.debug(f"Initialized notebook template: {name}")
    
    def render(self, variables: Dict[str, str]) -> Dict[str, Any]:
        """Render the template with variables.
        
        Args:
            variables: Dictionary with variables to replace
            
        Returns:
            Rendered notebook content
        """
        try:
            # Convert template to string
            template_str = json.dumps(self.content)
            
            # Replace variables
            for key, value in variables.items():
                placeholder = f"{{{key}}}"
                template_str = template_str.replace(placeholder, value)
            
            # Convert back to dictionary
            rendered_content = json.loads(template_str)
            
            return rendered_content
            
        except Exception as e:
            logger.error(f"Error rendering template {self.name}: {str(e)}")
            return self.content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the template to a dictionary.
        
        Returns:
            Dictionary representation of the template
        """
        return {
            'name': self.name,
            'content': self.content
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NotebookTemplate':
        """Create a template from a dictionary.
        
        Args:
            data: Dictionary with template data
            
        Returns:
            NotebookTemplate instance
        """
        return cls(data['name'], data['content'])
    
    @classmethod
    def from_file(cls, path: str) -> 'NotebookTemplate':
        """Create a template from a file.
        
        Args:
            path: Path to the template file
            
        Returns:
            NotebookTemplate instance
        """
        try:
            # Load template
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # Extract name from path
            name = path.split('/')[-1].split('.')[0]
            
            return cls(name, content)
            
        except Exception as e:
            logger.error(f"Error loading template from {path}: {str(e)}")
            return None
