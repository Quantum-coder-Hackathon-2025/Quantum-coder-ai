import schedule
import time
import os
import logging
from email_extractor import extract_email
from dotenv import load_dotenv
import json
import requests
import base64
from email.message import EmailMessage

# Load .env file
load_dotenv()

# Email details
sender_email = "your_email@example.com"
receiver_email = "recipient@example.com"
subject = "Facility Lender Share Adjustment Notification"
body = """Dear Recipient,

Please find attached the latest facility lender share adjustment details.

Best regards,  
ROBERTS SCOTT  
XYZ Bank N.A.  
"""


api_URL = os.getenv("API_END_POINT","http://127.0.0.1")  # Replace with actual API endpoint
headers = {"Content-Type": "application/json"}
INPUT_EML = os.getenv("EMAIL_INPUT_FOLDER")
INPUT_JSON = os.getenv("INPUT_JSON", "input_json")
OUTPUT = os.getenv("OUTPUT", "output_json")
DUPLICATE= os.getenv("DUPLICATE", "duplicate_json");

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def getPayload(eml_path):
    with open(eml_path, "r") as file:
        email_data = json.load(file)
        return email_data
    

def email_generation_frominput(eml_path):
    # Create email message
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(body)

    # Attach the file 1
    file_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/new-version/Quantum-coder-ai/process_folder/input/sample1.txt"
    with open(file_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(file_path)

    msg.add_attachment(file_data, maintype="text", subtype="plain", filename=file_name)
    eml_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/new-version/Quantum-coder-ai/process_folder/input_eml//sample1.eml"
        # Save as .eml file
    with open(eml_path, "wb") as eml_file:
        eml_file.write(msg.as_bytes())

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(body)

    # Attach the file 2
    file_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/new-version/Quantum-coder-ai/process_folder/input/sample2.txt"
    with open(file_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(file_path)

    msg.add_attachment(file_data, maintype="text", subtype="plain", filename=file_name)
    eml_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/new-version/Quantum-coder-ai/process_folder/input_eml//sample2.eml"
        # Save as .eml file
    with open(eml_path, "wb") as eml_file:
        eml_file.write(msg.as_bytes())

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(body)

    # Attach the file 3
    file_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/new-version/Quantum-coder-ai/process_folder/input/sample3.txt"
    with open(file_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(file_path)

    msg.add_attachment(file_data, maintype="text", subtype="plain", filename=file_name)
    eml_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/new-version/Quantum-coder-ai/process_folder/input_eml//sample3.eml"
        # Save as .eml file
    with open(eml_path, "wb") as eml_file:
        eml_file.write(msg.as_bytes())

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(body)

    # Attach the file 4
    file_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/new-version/Quantum-coder-ai/process_folder/input/sample4.txt"
    with open(file_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(file_path)
    msg.add_attachment(file_data, maintype="text", subtype="plain", filename=file_name)

    file_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/new-version/Quantum-coder-ai/process_folder/input/sample5.txt"
    with open(file_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(file_path)
    msg.add_attachment(file_data, maintype="text", subtype="plain", filename=file_name)

    file_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/new-version/Quantum-coder-ai/process_folder/input/sample6.txt"
    with open(file_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(file_path)
    msg.add_attachment(file_data, maintype="text", subtype="plain", filename=file_name)


    msg.add_attachment(file_data, maintype="text", subtype="plain", filename=file_name)
    eml_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/new-version/Quantum-coder-ai/process_folder/input_eml//multiple.eml"
        # Save as .eml file
    with open(eml_path, "wb") as eml_file:
        eml_file.write(msg.as_bytes())




def job():
    # setp1 extract the email from inbox
    if not INPUT_EML:
        logging.error("Error: EMAIL_INPUT_FOLDER environment variable is not set.")
        return
    logging.info(f"Starting email extraction for folder: {INPUT_EML}")
    email_generation_frominput(INPUT_EML)
    # extract_email(INPUT_EML)
    logging.info("Email extraction completed.")

    for filename in os.listdir(INPUT_EML):
        if filename.endswith(".eml"):
            #setp2 emial extraction(convert eml to json format) API Call
            eml_path = os.path.join(INPUT_EML, filename)
            data = {}
            files = {"file": (filename, open(eml_path, "rb"), "message/rfc822")}
            requests.post(api_URL+":"+os.getenv("EXTRACTION_API_PORT")+"/emailToJson", files=files,)

            #step3 call classifier API Call
            json_filename = filename.replace(".eml", ".json")
            json_input_path = os.path.join(INPUT_JSON, json_filename)
            data = getPayload(json_input_path)
            classiferResponse = requests.post(api_URL+":"+os.getenv("CLASSIFIER_PORT")+"/process_email/generatedInput", json=data, headers=headers)


            #step4 call duplicate check API Call
            duplicateCheckResponse = requests.post(api_URL+":"+os.getenv("DUPLICATE_CHECK_PORT")+"/duplicateCheck", json=data, headers=headers)
    
            #step5 generate output in output folder API Call
            output_json= classiferResponse.json()
            logging.info(f"duplicateIndicator:{duplicateCheckResponse.json().get("duplicateIndicator")}")
            output_json["duplicateCheck"] = duplicateCheckResponse.json().get("duplicateIndicator")
            logging.info(f"duplicateCheck:{output_json["duplicateCheck"]}")

            print(output_json["duplicateCheck"])
            json_output_filename = filename.replace(".eml", "_output.json")
            json_output_path = os.path.join(OUTPUT, json_output_filename)
            print(json_output_path)
            with open(json_output_path, "w", encoding="utf-8") as json_file:
                json.dump(output_json, json_file, indent=4)


            #step6 get vertex AI ouput API Call and stored in output
            #waiting for vivek

            os.remove(eml_path)
            os.remove(json_input_path)
            logging.info(f"File removed:{filename}")
            logging.info(f"File removed:{json_input_path}")
# Schedule the job every 1 minute
schedule.every(1).minutes.do(job)
logging.info("Scheduler started. Running job every 1 minute.")

while True:
    schedule.run_pending()
    logging.info("Waiting for the next scheduled job...")
    time.sleep(1)