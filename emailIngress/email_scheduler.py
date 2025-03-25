import schedule
import time
import os
import logging
from email_extractor import extract_email
from dotenv import load_dotenv
import requests

# Load .env file
load_dotenv()

api_URL = os.getenv("API_END_POINT","http://127.0.0.1")  # Replace with actual API endpoint
headers = {"Content-Type": "application/json"}
INPUT_EML = os.getenv("EMAIL_INPUT_FOLDER")
INPUT_JSON = os.getenv("INPUT_JSON", "output_json")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def job():
    #setp1 extract the email from inbox
    # if not INPUT_EML:
    #     logging.error("Error: EMAIL_INPUT_FOLDER environment variable is not set.")
    #     return
    # logging.info(f"Starting email extraction for folder: {INPUT_EML}")
    # extract_email(INPUT_EML)
    # logging.info("Email extraction completed.")

    for filename in os.listdir(INPUT_EML):
        if filename.endswith(".eml"):
            #setp2 emial extraction(convert eml to json format) API Call
            eml_path = os.path.join(INPUT_EML, filename)
            data = {}
            files = {"file": (filename, open(eml_path, "rb"), "message/rfc822")}
            response = requests.post(api_URL+":"+os.getenv("EXTRACTION_API_PORT")+"/", files=files,)
            logging.info(f"File translated to eml to json:{response}")

            #step3 call classifier API Call


            #step4 call duplicate check API Call
            #step5 generate output in output folder API Call
    

# Schedule the job every 1 minute
schedule.every(1).minute.do(job)
logging.info("Scheduler started. Running job every 1 minute.")

while True:
    schedule.run_pending()
    logging.info("Waiting for the next scheduled job...")
    time.sleep(1)