"""
Script to upload notebooks to Google Drive.

Note: This script is for reference only. It requires authentication with Google Drive,
which is not possible in this environment. You'll need to run this script manually
after setting up the necessary authentication.
"""
import os
import logging
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate_drive():
    """Authenticate with Google Drive API."""
    creds = None
    
    # Check if token.json exists
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_info(json.loads(open('token.json').read()))
    
    # If credentials don't exist or are invalid, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for future use
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return creds

def create_drive_folder(service, folder_name, parent_id=None):
    """Create a folder in Google Drive."""
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    
    if parent_id:
        file_metadata['parents'] = [parent_id]
    
    folder = service.files().create(body=file_metadata, fields='id').execute()
    logger.info(f"Created folder: {folder_name} (ID: {folder.get('id')})")
    
    return folder.get('id')

def upload_file_to_drive(service, file_path, folder_id=None):
    """Upload a file to Google Drive."""
    file_name = os.path.basename(file_path)
    
    file_metadata = {
        'name': file_name
    }
    
    if folder_id:
        file_metadata['parents'] = [folder_id]
    
    media = MediaFileUpload(file_path, mimetype='application/json')
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id,webViewLink'
    ).execute()
    
    logger.info(f"Uploaded file: {file_name} (ID: {file.get('id')})")
    logger.info(f"Web link: {file.get('webViewLink')}")
    
    return file.get('id'), file.get('webViewLink')

def upload_notebooks_to_drive():
    """Upload notebooks to Google Drive."""
    try:
        # Authenticate with Google Drive
        creds = authenticate_drive()
        service = build('drive', 'v3', credentials=creds)
        
        # Create main folder
        main_folder_id = create_drive_folder(service, 'Stock Volume Analysis')
        
        # Create individual notebooks folder
        individual_folder_id = create_drive_folder(service, 'Individual Stock Notebooks', main_folder_id)
        
        # Upload individual notebooks
        individual_notebooks_dir = Path('notebooks/individual')
        notebook_links = {}
        
        for notebook_path in individual_notebooks_dir.glob('*.ipynb'):
            file_id, web_link = upload_file_to_drive(service, str(notebook_path), individual_folder_id)
            ticker = notebook_path.stem.split('_')[0]
            notebook_links[ticker] = web_link
        
        # Upload master summary notebook
        master_path = Path('notebooks/master_summary.ipynb')
        if master_path.exists():
            file_id, web_link = upload_file_to_drive(service, str(master_path), main_folder_id)
            notebook_links['master'] = web_link
        
        # Save notebook links to a file
        with open('notebook_links.json', 'w') as f:
            json.dump(notebook_links, f, indent=2)
        
        logger.info(f"Uploaded {len(notebook_links) - 1} individual notebooks and 1 master notebook")
        logger.info(f"Notebook links saved to notebook_links.json")
        
        return True
    
    except Exception as e:
        logger.error(f"Error uploading notebooks to Google Drive: {str(e)}")
        return False

def main():
    """Main function."""
    logger.info("This script requires authentication with Google Drive.")
    logger.info("Please follow these steps:")
    logger.info("1. Go to the Google Cloud Console: https://console.cloud.google.com/")
    logger.info("2. Create a new project or select an existing project")
    logger.info("3. Enable the Google Drive API")
    logger.info("4. Create OAuth 2.0 credentials and download as credentials.json")
    logger.info("5. Place credentials.json in the same directory as this script")
    logger.info("6. Run this script again")
    
    # Check if credentials.json exists
    if not os.path.exists('credentials.json'):
        logger.error("credentials.json not found")
        return False
    
    # Upload notebooks to Google Drive
    success = upload_notebooks_to_drive()
    
    if success:
        logger.info("Notebooks uploaded to Google Drive successfully")
    else:
        logger.error("Failed to upload notebooks to Google Drive")

if __name__ == "__main__":
    main()
