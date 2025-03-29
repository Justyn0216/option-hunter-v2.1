"""
Google Drive Integration Module

This module handles integration with Google Drive for storing logs,
trade data, model files, and other persistent data.
"""

import os
import pickle
import logging
from datetime import datetime
import pandas as pd
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

class GoogleDriveManager:
    """
    Google Drive integration for data storage and retrieval.
    
    Handles:
    - Authentication with Google Drive
    - File uploads and downloads
    - Creating and managing folder structure
    - File search and management
    """
    
    # Define scopes for Google Drive API
    SCOPES = ['https://www.googleapis.com/auth/drive']
    
    def __init__(self, credentials_file, token_file, root_folder_id=None):
        """
        Initialize the Google Drive manager.
        
        Args:
            credentials_file (str): Path to the credentials.json file
            token_file (str): Path to save/load the token.pickle file
            root_folder_id (str, optional): ID of the root folder in Google Drive
        """
        self.logger = logging.getLogger(__name__)
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.root_folder_id = root_folder_id
        self.service = None
        self.folder_cache = {}  # Cache folder IDs to reduce API calls
        
        # Connect to Google Drive
        self._authenticate()
        
        # Create folder structure if root folder is provided
        if self.root_folder_id and self.service:
            self._initialize_folder_structure()
    
    def _authenticate(self):
        """Authenticate with Google Drive API."""
        try:
            creds = None
            
            # Check if token file exists
            if os.path.exists(self.token_file):
                with open(self.token_file, 'rb') as token:
                    creds = pickle.load(token)
            
            # If credentials don't exist or are invalid, refresh or create new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(self.credentials_file):
                        self.logger.error(f"Credentials file not found: {self.credentials_file}")
                        return
                        
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, self.SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(self.token_file, 'wb') as token:
                    pickle.dump(creds, token)
            
            # Build the Drive API service
            self.service = build('drive', 'v3', credentials=creds)
            self.logger.info("Successfully authenticated with Google Drive")
            
        except Exception as e:
            self.logger.error(f"Authentication with Google Drive failed: {str(e)}")
            self.service = None
    
    def _initialize_folder_structure(self):
        """Initialize the folder structure in Google Drive."""
        # Main folders
        folders = [
            'logs',
            'trades',
            'models',
            'data',
            'backtest_results',
            'performance'
        ]
        
        # Create main folders if they don't exist
        for folder_name in folders:
            folder_id = self.create_folder_if_not_exists(folder_name, self.root_folder_id)
            self.folder_cache[folder_name] = folder_id
            
        # Create subfolders
        subfolders = {
            'logs': ['system', 'trades', 'errors', 'performance'],
            'trades': ['completed', 'active'],
            'models': ['ml', 'rl', 'parameters'],
            'data': ['market_data', 'option_chains', 'sentiment'],
            'performance': ['daily', 'weekly', 'monthly']
        }
        
        for parent, subs in subfolders.items():
            parent_id = self.folder_cache.get(parent)
            if parent_id:
                for sub in subs:
                    subfolder_id = self.create_folder_if_not_exists(sub, parent_id)
                    self.folder_cache[f"{parent}/{sub}"] = subfolder_id
        
        self.logger.info("Google Drive folder structure initialized")
    
    def create_folder_if_not_exists(self, folder_name, parent_id=None):
        """
        Create a folder in Google Drive if it doesn't exist.
        
        Args:
            folder_name (str): Name of the folder to create
            parent_id (str, optional): ID of the parent folder
            
        Returns:
            str: ID of the created or existing folder
        """
        if not self.service:
            self.logger.error("Google Drive service not initialized")
            return None
            
        try:
            # Check if folder already exists
            query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder'"
            if parent_id:
                query += f" and '{parent_id}' in parents"
                
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            files = response.get('files', [])
            
            # If folder exists, return its ID
            if files:
                self.logger.debug(f"Folder already exists: {folder_name}")
                return files[0]['id']
            
            # If not, create the folder
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_id:
                folder_metadata['parents'] = [parent_id]
                
            folder = self.service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            
            self.logger.debug(f"Created folder: {folder_name}")
            return folder.get('id')
            
        except Exception as e:
            self.logger.error(f"Error creating folder {folder_name}: {str(e)}")
            return None
    
    def upload_file(self, filename, content, parent_folder=None, mime_type='text/plain'):
        """
        Upload a file to Google Drive.
        
        Args:
            filename (str): Name of the file
            content: File content (str, bytes, or file-like object)
            parent_folder (str, optional): Parent folder name or ID
            mime_type (str): MIME type of the file
            
        Returns:
            str: ID of the uploaded file, or None if upload failed
        """
        if not self.service:
            self.logger.error("Google Drive service not initialized")
            return None
            
        try:
            # Determine parent folder ID
            parent_id = self.root_folder_id
            
            if parent_folder:
                # Check if it's a folder name in our cache
                if parent_folder in self.folder_cache:
                    parent_id = self.folder_cache[parent_folder]
                # Check if it's a path (e.g., "logs/trades")
                elif '/' in parent_folder:
                    parent_id = self.folder_cache.get(parent_folder)
                # Otherwise assume it's a folder ID
                else:
                    parent_id = parent_folder
            
            # Create file metadata
            file_metadata = {
                'name': filename
            }
            
            if parent_id:
                file_metadata['parents'] = [parent_id]
            
            # Create media
            if isinstance(content, str):
                media = MediaIoBaseUpload(
                    io.BytesIO(content.encode('utf-8')),
                    mimetype=mime_type
                )
            elif isinstance(content, bytes):
                media = MediaIoBaseUpload(
                    io.BytesIO(content),
                    mimetype=mime_type
                )
            elif hasattr(content, 'read'):
                media = MediaIoBaseUpload(
                    content,
                    mimetype=mime_type
                )
            else:
                self.logger.error(f"Unsupported content type for file {filename}")
                return None
            
            # Check if file already exists and update it
            file_id = self.find_file(filename, parent_id)
            
            if file_id:
                # Update existing file
                file = self.service.files().update(
                    fileId=file_id,
                    media_body=media
                ).execute()
                
                self.logger.debug(f"Updated file: {filename}")
            else:
                # Create new file
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                
                self.logger.debug(f"Created file: {filename}")
            
            return file.get('id')
            
        except Exception as e:
            self.logger.error(f"Error uploading file {filename}: {str(e)}")
            return None
    
    def upload_file_from_path(self, file_path, parent_folder=None, rename=None):
        """
        Upload a file from a local path to Google Drive.
        
        Args:
            file_path (str): Path to the local file
            parent_folder (str, optional): Parent folder name or ID
            rename (str, optional): New name for the file in Google Drive
            
        Returns:
            str: ID of the uploaded file, or None if upload failed
        """
        if not self.service:
            self.logger.error("Google Drive service not initialized")
            return None
            
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return None
                
            # Determine filename
            filename = rename if rename else os.path.basename(file_path)
            
            # Determine parent folder ID (same logic as upload_file)
            parent_id = self.root_folder_id
            
            if parent_folder:
                if parent_folder in self.folder_cache:
                    parent_id = self.folder_cache[parent_folder]
                elif '/' in parent_folder:
                    parent_id = self.folder_cache.get(parent_folder)
                else:
                    parent_id = parent_folder
            
            # Create file metadata
            file_metadata = {
                'name': filename
            }
            
            if parent_id:
                file_metadata['parents'] = [parent_id]
            
            # Create media
            media = MediaFileUpload(file_path)
            
            # Check if file already exists and update it
            file_id = self.find_file(filename, parent_id)
            
            if file_id:
                # Update existing file
                file = self.service.files().update(
                    fileId=file_id,
                    media_body=media
                ).execute()
                
                self.logger.debug(f"Updated file: {filename}")
            else:
                # Create new file
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                
                self.logger.debug(f"Created file: {filename}")
            
            return file.get('id')
            
        except Exception as e:
            self.logger.error(f"Error uploading file {file_path}: {str(e)}")
            return None
    
    def download_file(self, file_id_or_name, parent_folder=None):
        """
        Download a file from Google Drive.
        
        Args:
            file_id_or_name (str): ID or name of the file
            parent_folder (str, optional): Parent folder name or ID, used if file_id_or_name is a name
            
        Returns:
            str: Content of the file, or None if download failed
        """
        if not self.service:
            self.logger.error("Google Drive service not initialized")
            return None
            
        try:
            # Determine file ID
            file_id = file_id_or_name
            
            # If it looks like a name, try to find the file ID
            if not file_id_or_name.startswith('0') and len(file_id_or_name) < 30:
                parent_id = None
                
                if parent_folder:
                    if parent_folder in self.folder_cache:
                        parent_id = self.folder_cache[parent_folder]
                    elif '/' in parent_folder:
                        parent_id = self.folder_cache.get(parent_folder)
                    else:
                        parent_id = parent_folder
                
                file_id = self.find_file(file_id_or_name, parent_id)
                
                if not file_id:
                    self.logger.error(f"File not found: {file_id_or_name}")
                    return None
            
            # Download the file content
            request = self.service.files().get_media(fileId=file_id)
            file_content = request.execute()
            
            # Decode if it's bytes
            if isinstance(file_content, bytes):
                try:
                    file_content = file_content.decode('utf-8')
                except UnicodeDecodeError:
                    # Not a text file, return bytes
                    pass
            
            return file_content
            
        except Exception as e:
            self.logger.error(f"Error downloading file {file_id_or_name}: {str(e)}")
            return None
    
    def download_file_binary(self, file_id_or_name, parent_folder=None):
        """
        Download a binary file from Google Drive.
        
        Args:
            file_id_or_name (str): ID or name of the file
            parent_folder (str, optional): Parent folder name or ID, used if file_id_or_name is a name
            
        Returns:
            bytes: Binary content of the file, or None if download failed
        """
        if not self.service:
            self.logger.error("Google Drive service not initialized")
            return None
            
        try:
            # Determine file ID (same logic as download_file)
            file_id = file_id_or_name
            
            if not file_id_or_name.startswith('0') and len(file_id_or_name) < 30:
                parent_id = None
                
                if parent_folder:
                    if parent_folder in self.folder_cache:
                        parent_id = self.folder_cache[parent_folder]
                    elif '/' in parent_folder:
                        parent_id = self.folder_cache.get(parent_folder)
                    else:
                        parent_id = parent_folder
                
                file_id = self.find_file(file_id_or_name, parent_id)
                
                if not file_id:
                    self.logger.error(f"File not found: {file_id_or_name}")
                    return None
            
            # Download the file content
            request = self.service.files().get_media(fileId=file_id)
            file_content = request.execute()
            
            return file_content
            
        except Exception as e:
            self.logger.error(f"Error downloading binary file {file_id_or_name}: {str(e)}")
            return None
    
    def download_file_to_path(self, file_id_or_name, local_path, parent_folder=None):
        """
        Download a file from Google Drive to a local path.
        
        Args:
            file_id_or_name (str): ID or name of the file
            local_path (str): Local path to save the file
            parent_folder (str, optional): Parent folder name or ID, used if file_id_or_name is a name
            
        Returns:
            bool: True if download succeeded, False otherwise
        """
        if not self.service:
            self.logger.error("Google Drive service not initialized")
            return False
            
        try:
            # Get file content
            content = self.download_file_binary(file_id_or_name, parent_folder)
            
            if content is None:
                return False
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
                
            # Write to local file
            with open(local_path, 'wb') as f:
                f.write(content)
                
            self.logger.debug(f"Downloaded file to {local_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading file to {local_path}: {str(e)}")
            return False
    
    def find_file(self, filename, parent_folder_id=None):
        """
        Find a file in Google Drive by name.
        
        Args:
            filename (str): Name of the file
            parent_folder_id (str, optional): ID of the parent folder
            
        Returns:
            str: ID of the file, or None if not found
        """
        if not self.service:
            self.logger.error("Google Drive service not initialized")
            return None
            
        try:
            # Build query
            query = f"name = '{filename}' and trashed = false"
            
            if parent_folder_id:
                query += f" and '{parent_folder_id}' in parents"
                
            # Search for the file
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            files = response.get('files', [])
            
            if not files:
                return None
                
            # Return the first match
            return files[0]['id']
            
        except Exception as e:
            self.logger.error(f"Error finding file {filename}: {str(e)}")
            return None
    
    def list_files(self, folder_id_or_name=None):
        """
        List files in a Google Drive folder.
        
        Args:
            folder_id_or_name (str, optional): ID or name of the folder
            
        Returns:
            list: List of files in the folder
        """
        if not self.service:
            self.logger.error("Google Drive service not initialized")
            return []
            
        try:
            # Determine folder ID
            folder_id = self.root_folder_id
            
            if folder_id_or_name:
                if folder_id_or_name in self.folder_cache:
                    folder_id = self.folder_cache[folder_id_or_name]
                elif '/' in folder_id_or_name:
                    folder_id = self.folder_cache.get(folder_id_or_name)
                else:
                    folder_id = folder_id_or_name
            
            if not folder_id:
                self.logger.error(f"Folder not found: {folder_id_or_name}")
                return []
                
            # List files in the folder
            query = f"'{folder_id}' in parents and trashed = false"
            
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, mimeType, createdTime, modifiedTime)'
            ).execute()
            
            return response.get('files', [])
            
        except Exception as e:
            self.logger.error(f"Error listing files in folder {folder_id_or_name}: {str(e)}")
            return []
    
    def delete_file(self, file_id_or_name, parent_folder=None):
        """
        Delete a file from Google Drive.
        
        Args:
            file_id_or_name (str): ID or name of the file
            parent_folder (str, optional): Parent folder name or ID, used if file_id_or_name is a name
            
        Returns:
            bool: True if deletion succeeded, False otherwise
        """
        if not self.service:
            self.logger.error("Google Drive service not initialized")
            return False
            
        try:
            # Determine file ID
            file_id = file_id_or_name
            
            if not file_id_or_name.startswith('0') and len(file_id_or_name) < 30:
                parent_id = None
                
                if parent_folder:
                    if parent_folder in self.folder_cache:
                        parent_id = self.folder_cache[parent_folder]
                    elif '/' in parent_folder:
                        parent_id = self.folder_cache.get(parent_folder)
                    else:
                        parent_id = parent_folder
                
                file_id = self.find_file(file_id_or_name, parent_id)
                
                if not file_id:
                    self.logger.error(f"File not found: {file_id_or_name}")
                    return False
            
            # Delete the file
            self.service.files().delete(fileId=file_id).execute()
            
            self.logger.debug(f"Deleted file: {file_id_or_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting file {file_id_or_name}: {str(e)}")
            return False
    
    def file_exists(self, filename, parent_folder=None):
        """
        Check if a file exists in Google Drive.
        
        Args:
            filename (str): Name of the file
            parent_folder (str, optional): Parent folder name or ID
            
        Returns:
            bool: True if the file exists, False otherwise
        """
        if not self.service:
            self.logger.error("Google Drive service not initialized")
            return False
            
        parent_id = self.root_folder_id
        
        if parent_folder:
            if parent_folder in self.folder_cache:
                parent_id = self.folder_cache[parent_folder]
            elif '/' in parent_folder:
                parent_id = self.folder_cache.get(parent_folder)
            else:
                parent_id = parent_folder
        
        file_id = self.find_file(filename, parent_id)
        return file_id is not None
    
    def upload_dataframe(self, df, filename, parent_folder=None, format='csv'):
        """
        Upload a pandas DataFrame to Google Drive.
        
        Args:
            df (pandas.DataFrame): DataFrame to upload
            filename (str): Name of the file
            parent_folder (str, optional): Parent folder name or ID
            format (str): File format ('csv', 'excel', or 'json')
            
        Returns:
            str: ID of the uploaded file, or None if upload failed
        """
        if not self.service:
            self.logger.error("Google Drive service not initialized")
            return None
            
        try:
            # Convert DataFrame to desired format
            buffer = io.BytesIO()
            
            if format.lower() == 'csv':
                df.to_csv(buffer, index=False)
                mime_type = 'text/csv'
                
                # Make sure filename ends with .csv
                if not filename.lower().endswith('.csv'):
                    filename += '.csv'
                    
            elif format.lower() == 'excel':
                df.to_excel(buffer, index=False)
                mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                
                # Make sure filename ends with .xlsx
                if not filename.lower().endswith('.xlsx'):
                    filename += '.xlsx'
                    
            elif format.lower() == 'json':
                df.to_json(buffer, orient='records')
                mime_type = 'application/json'
                
                # Make sure filename ends with .json
                if not filename.lower().endswith('.json'):
                    filename += '.json'
                    
            else:
                self.logger.error(f"Unsupported format: {format}")
                return None
            
            # Reset buffer position
            buffer.seek(0)
            
            # Upload the file
            return self.upload_file(filename, buffer, parent_folder, mime_type)
            
        except Exception as e:
            self.logger.error(f"Error uploading DataFrame to {filename}: {str(e)}")
            return None
    
    def log_trade(self, trade_data, log_file=None):
        """
        Log trade data to Google Drive.
        
        Args:
            trade_data (dict): Trade data to log
            log_file (str, optional): Name of the log file
            
        Returns:
            bool: True if logging succeeded, False otherwise
        """
        if not self.service:
            self.logger.error("Google Drive service not initialized")
            return False
            
        try:
            # Default log file name based on date
            if not log_file:
                date_str = datetime.now().strftime('%Y%m%d')
                log_file = f"trades_{date_str}.json"
            
            # Get existing log if it exists
            parent_folder = "trades"
            existing_data = []
            
            if self.file_exists(log_file, parent_folder):
                content = self.download_file(log_file, parent_folder)
                if content:
                    import json
                    existing_data = json.loads(content)
            
            # Add new trade data
            existing_data.append(trade_data)
            
            # Add timestamp if not present
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Upload updated log
            import json
            self.upload_file(
                log_file,
                json.dumps(existing_data, indent=2),
                parent_folder,
                'application/json'
            )
            
            self.logger.debug(f"Logged trade data to {log_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging trade data: {str(e)}")
            return False
