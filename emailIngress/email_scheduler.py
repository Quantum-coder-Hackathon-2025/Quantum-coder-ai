import schedule
import time
import os
import logging
from email_extractor import extract_email
from dotenv import load_dotenv
import json
import requests
import base64

# Load .env file
load_dotenv()

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
    

def job():
    # setp1 extract the email from inbox
    if not INPUT_EML:
        logging.error("Error: EMAIL_INPUT_FOLDER environment variable is not set.")
        return
    logging.info(f"Starting email extraction for folder: {INPUT_EML}")
    extract_email(INPUT_EML)
    logging.info("Email extraction completed.")

    for filename in os.listdir(INPUT_EML):
        if filename.endswith(".eml"):
            #setp2 emial extraction(convert eml to json format) API Call
            eml_path = os.path.join(INPUT_EML, filename)
            data = {}
            files = {"file": (filename, open(eml_path, "rb"), "message/rfc822")}
            response = requests.post(api_URL+":"+os.getenv("EXTRACTION_API_PORT")+"/emailToJson", files=files,)
            # logging.info(f"File translated to eml to json:{response.json()}")

            #step3 call classifier API Call
            json_filename = filename.replace(".eml", ".json")
            json_input_path = os.path.join(INPUT_JSON, json_filename)
            data = getPayload(json_input_path)
            response = requests.post(api_URL+":"+os.getenv("CLASSIFIER_PORT")+"/process_email/generatedInput", json=data, headers=headers)
            logging.info(f"File classified:{response.json()}")

            json_output_filename = filename.replace(".eml", "_output.json")
            json_output_path = os.path.join(OUTPUT, json_output_filename)
            print(json_output_path)
            with open(json_output_path, "w", encoding="utf-8") as json_file:
                json.dump(response.json(), json_file, indent=4)
            #step4 call duplicate check API Call
            response = requests.post(api_URL+":"+os.getenv("DUPLICATE_CHECK_PORT")+"/duplicateCheck", json=data, headers=headers)
            logging.info(f"File classified:{response.json()}")

            json_duplicate_filename = filename.replace(".eml", "_dupllicate_check.json")
            json_duplicate_path = os.path.join(DUPLICATE, json_duplicate_filename)
            print(json_duplicate_path)
            with open(json_duplicate_path, "w", encoding="utf-8") as json_file:
                json.dump(response.json(), json_file, indent=4)

            #step5 generate output in output folder API Call
            #output has generated above steps so no need to call API

            #step6 get vertex AI ouput API Call
            
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