"""
Notebook launcher module for Delphi.

This module provides a class for launching Google Colab notebooks.
"""
from typing import Dict, List, Optional, Any, Union
import os
import json
import logging
import webbrowser
import time
from pathlib import Path

from delphi.core.base.service import Service

# Configure logger
logger = logging.getLogger(__name__)

class NotebookLauncher(Service):
    """Service for launching Google Colab notebooks."""
    
    def __init__(self, credentials_path: Optional[str] = None, **kwargs):
        """Initialize the notebook launcher.
        
        Args:
            credentials_path: Path to Google API credentials file
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        
        self.credentials_path = credentials_path
        
        logger.info("Initialized notebook launcher")
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the notebook launcher.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            True if initialization is successful, False otherwise
        """
        # Nothing to initialize
        return True
    
    def upload_to_drive(self, notebook_paths: List[Path]) -> Dict[str, str]:
        """Upload notebooks to Google Drive.
        
        Args:
            notebook_paths: List of paths to notebooks
            
        Returns:
            Dictionary mapping notebook names to URLs
        """
        try:
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            
            # Define scopes
            SCOPES = ['https://www.googleapis.com/auth/drive.file']
            
            # Authenticate
            creds = None
            token_path = Path('token.json')
            
            # Load credentials from token.json if it exists
            if token_path.exists():
                creds = Credentials.from_authorized_user_info(json.loads(token_path.read_text()), SCOPES)
            
            # If credentials don't exist or are invalid, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not self.credentials_path:
                        logger.error("Google API credentials path not provided")
                        return {}
                    
                    flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save credentials for next run
                token_path.write_text(creds.to_json())
            
            # Build Drive API client
            service = build('drive', 'v3', credentials=creds)
            
            # Create main folder
            folder_metadata = {
                'name': 'Delphi Stock Analysis',
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            folder = service.files().create(body=folder_metadata, fields='id').execute()
            main_folder_id = folder.get('id')
            
            # Create individual notebooks folder
            individual_folder_metadata = {
                'name': 'Individual Stock Notebooks',
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [main_folder_id]
            }
            
            individual_folder = service.files().create(body=individual_folder_metadata, fields='id').execute()
            individual_folder_id = individual_folder.get('id')
            
            # Upload notebooks
            notebook_urls = {}
            
            for notebook_path in notebook_paths:
                # Determine parent folder
                parent_folder_id = main_folder_id
                if 'individual' in str(notebook_path):
                    parent_folder_id = individual_folder_id
                
                # Upload file
                file_metadata = {
                    'name': notebook_path.name,
                    'parents': [parent_folder_id]
                }
                
                media = MediaFileUpload(str(notebook_path), mimetype='application/json')
                file = service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id,webViewLink'
                ).execute()
                
                # Store URL
                notebook_urls[notebook_path.stem] = file.get('webViewLink')
                
                logger.info(f"Uploaded {notebook_path.name} to Google Drive")
            
            return notebook_urls
            
        except Exception as e:
            logger.error(f"Error uploading notebooks to Google Drive: {str(e)}")
            return {}
    
    def launch_notebooks(self, notebook_urls: Dict[str, str], master_first: bool = True) -> bool:
        """Launch notebooks in browser.
        
        Args:
            notebook_urls: Dictionary mapping notebook names to URLs
            master_first: Whether to launch master notebook first
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Sort notebooks
            master_notebooks = {k: v for k, v in notebook_urls.items() if 'master' in k or 'performance' in k or 'model_training' in k}
            individual_notebooks = {k: v for k, v in notebook_urls.items() if 'master' not in k and 'performance' not in k and 'model_training' not in k}
            
            # Launch master notebooks first if requested
            if master_first:
                for name, url in master_notebooks.items():
                    logger.info(f"Launching master notebook: {name}")
                    webbrowser.open(url)
                    time.sleep(1)  # Wait a bit between launches
            
            # Launch individual notebooks
            for name, url in individual_notebooks.items():
                logger.info(f"Launching individual notebook: {name}")
                webbrowser.open(url)
                time.sleep(1)  # Wait a bit between launches
            
            # Launch master notebooks last if not launched first
            if not master_first:
                for name, url in master_notebooks.items():
                    logger.info(f"Launching master notebook: {name}")
                    webbrowser.open(url)
                    time.sleep(1)  # Wait a bit between launches
            
            return True
            
        except Exception as e:
            logger.error(f"Error launching notebooks: {str(e)}")
            return False
    
    def convert_to_colab_url(self, notebook_path: Path) -> str:
        """Convert a local notebook path to a Colab URL.
        
        Args:
            notebook_path: Path to the notebook
            
        Returns:
            Colab URL
        """
        try:
            # Get absolute path
            abs_path = notebook_path.resolve()
            
            # Convert to Colab URL
            colab_url = f"https://colab.research.google.com/github/file/{abs_path}"
            
            return colab_url
            
        except Exception as e:
            logger.error(f"Error converting to Colab URL: {str(e)}")
            return ""
