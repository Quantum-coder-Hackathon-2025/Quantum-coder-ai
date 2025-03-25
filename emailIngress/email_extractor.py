from __future__ import print_function
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64
import email
from email import policy
from email.parser import BytesParser
from gdrive_writer import upload_to_gdrive, get_gdrive_service
import sys


# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

   

def clean(text):
    return "".join(c if c.isalnum() else "_" for c in text)

def get_email_body_and_attachments(msg):
    body = None
    attachments = []
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if "attachment" in content_disposition:
                filename = part.get_filename()
                if filename:
                    attachments.append((filename, part.get_payload(decode=True)))
            elif content_type == "text/plain" and body is None:
                body = part.get_payload(decode=True).decode()
    else:
        body = msg.get_payload(decode=True).decode()

    return body, attachments

def save_email_as_eml(service, message_id, eml_folder):
    msg = service.users().messages().get(userId='me', id=message_id, format='raw').execute()
    raw_message = msg['raw']
    subject = msg['snippet'][:50]  # Use snippet as a placeholder for subject
    eml_file_path = f"{clean(subject)}.eml"
    eml_file_path = os.path.join(eml_folder, eml_file_path)
    with open(eml_file_path, 'wb') as eml_file:
        eml_file.write(base64.urlsafe_b64decode(raw_message))
    print(f"Email saved as {eml_file_path}")
    return eml_file_path, raw_message


def extract_email(folder_name):
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail messages.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    # Connect to the Gmail API
    service = build('gmail', 'v1', credentials=creds)

    # Create a folder to save the attachments
    # attachments_folder = folder_name+"\\attachments"
    # os.makedirs(attachments_folder, exist_ok=True)
    eml_folder = folder_name
    os.makedirs(eml_folder, exist_ok=True)
    # Fetch the list of messages
    results = service.users().messages().list(userId='me', q='is:unread').execute()
    messages = results.get('messages', [])

    if not messages:
        print('No messages found.')
    else:
        print('Messages:')
        for message in messages[:10]:  # Fetch the first 10 messages
            msg_id = message['id']  # Get the message ID
            print(f"Message ID: {msg_id}")
            eml_file_path, raw_message = save_email_as_eml(service, msg_id, eml_folder)
            print(f"Email saved as {eml_file_path}")


            #upload email file to google drive
            # upload_to_gdrive(msg_id, eml_file_path)
             # Parse the email content
            msg_bytes = base64.urlsafe_b64decode(raw_message)
            msg = BytesParser(policy=policy.default).parsebytes(msg_bytes)
            body, attachments = get_email_body_and_attachments(msg)

            # Print the email body
            print(f"Email body: {body}")

             # Save the email body to a text file
            # body_file_path = os.path.join(attachments_folder, f"{msg_id}.txt")
            # with open(body_file_path, 'w', encoding='utf-8') as body_file:
            #     body_file.write(body)
            # print(f"Email body saved as {body_file_path}")
            # upload_to_gdrive(msg_id, body_file_path)

            # Save attachments
            # for filename, content in attachments:
            #     attachment_path = os.path.join(attachments_folder, filename)
            #     with open(attachment_path, 'wb') as f:
            #         f.write(content)
            #     print(f"Attachment saved as {attachment_path}")
            #     upload_to_gdrive(msg_id, attachment_path)

             # Mark the email as read
            service.users().messages().modify(userId='me', id=message['id'], body={'removeLabelIds': ['UNREAD']}).execute()



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python email_extractor.py <folder_name>")
        sys.exit(1)
    folder_name = sys.argv[1]
    extract_email(folder_name)