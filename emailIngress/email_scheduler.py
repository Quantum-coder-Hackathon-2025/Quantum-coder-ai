import schedule
import time
import os
import logging
from email_extractor import extract_email
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def job():
    folder_name = os.getenv("EMAIL_OUTPUT_FOLDER")
    if not folder_name:
        logging.error("Error: EMAIL_OUTPUT_FOLDER environment variable is not set.")
        return
    logging.info(f"Starting email extraction for folder: {folder_name}")
    extract_email(folder_name)
    logging.info("Email extraction completed.")

    #setp2 emial extraction API Call

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