import os.path
import pickle
import sys
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload 


# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

def get_gdrive_service():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    # return Google Drive API service
    return build('drive', 'v3', credentials=creds)

def get_folder_id(service, folder_name):
    # Search for the folder by name
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if items:
        # Folder exists, return its ID
        return items[0]['id']
    else:
        # Folder does not exist, return None
        return None

def upload_to_gdrive(folder_name, file_path):
  # authenticate account
    service = get_gdrive_service()
    # folder details we want to make
    
     # Check if the folder exists
    folder_id = get_folder_id(service, folder_name)
    if folder_id:
        print(f"Folder '{folder_name}' already exists with ID: {folder_id}")
    else:
        # Folder does not exist, create it
        folder_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder"
        }
        file = service.files().create(body=folder_metadata, fields="id").execute()
        folder_id = file.get("id")
        print(f"Folder '{folder_name}' created with ID: {folder_id}")
    

    # upload a file text file
    # first, define file metadata, such as the name and the parent folder ID

     # Extract the filename from the full path
    file_name = os.path.basename(file_path)

    file_metadata = {
        "name": file_name,
        "parents": [folder_id]
    }
    # upload
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print("File created, id:", file.get("id"))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python gdrive_writer.py <folder_name> <file_path>")
        sys.exit(1)
    folder_name = sys.argv[1]
    file_path = sys.argv[2]
    upload_to_gdrive(folder_name, file_path)