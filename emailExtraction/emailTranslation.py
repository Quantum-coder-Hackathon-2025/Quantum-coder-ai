import json
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from email import policy
from email.parser import BytesParser
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import uvicorn



# Load environment variables
load_dotenv()

# Get paths from .env file
INPUT_EML = os.getenv("EML_INPUT", "input_emails")
INPUT_JSON = os.getenv("INPUT_JSON", "output_json")

# Ensure folders exist
os.makedirs(INPUT_EML, exist_ok=True)
os.makedirs(INPUT_JSON, exist_ok=True)

# Initialize FastAPI
app = FastAPI()

# Helper functions for text extraction
def extract_text_from_pdf(pdf_bytes):
    """Extract text from a PDF file."""
    text = ""
    pdf_reader = PdfReader(BytesIO(pdf_bytes))
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(docx_bytes):
    """Extract text from a DOCX file."""
    doc = Document(BytesIO(docx_bytes))
    return "\n".join([para.text for para in doc.paragraphs]).strip()

def extract_text_from_image(image_bytes):
    """Extract text from an image using OCR."""
    image = Image.open(BytesIO(image_bytes))
    return pytesseract.image_to_string(image).strip()

def extract_text_from_txt(txt_bytes):
    """Extract text from a TXT file."""
    return txt_bytes.decode("utf-8").strip()

def read_eml_file(eml_path):
    """Read an EML file and extract content + attachments."""
    with open(eml_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    email_data = {
        "from_email": msg["From"],
        "to": msg["To"],
        "subject": msg["Subject"],
        "body": "",
        "attachments": []
    }

    # Extract email body
    if msg.is_multipart():
        for part in msg.iter_parts():
            if part.get_content_type() == "text/plain" and not part.get_filename():
                email_data["body"] = part.get_content().strip()
    else:
        email_data["body"] = msg.get_content().strip()

    # Extract all attachments
    for part in msg.iter_attachments():
        attachment_filename = part.get_filename()
        file_bytes = part.get_payload(decode=True)

        extracted_text = None
        if attachment_filename.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file_bytes)
        elif attachment_filename.endswith(".docx"):
            extracted_text = extract_text_from_docx(file_bytes)
        elif attachment_filename.endswith(".txt"):
            extracted_text = extract_text_from_txt(file_bytes)
        elif attachment_filename.endswith((".jpg", ".jpeg", ".png")):
            extracted_text = extract_text_from_image(file_bytes)

        email_data["attachments"].append({
            "filename": attachment_filename,
            "content": extracted_text if extracted_text else "Binary file (not extracted)"
        })

    return email_data

def process_eml_file(eml_bytes):
    """Read an EML file and extract content + attachments."""
    msg = BytesParser(policy=policy.default).parse(BytesIO(eml_bytes))

    email_data = {
        "from_email": msg["From"],
        "to": msg["To"],
        "subject": msg["Subject"],
        "body": "",
        "attachments": []
    }

    # Extract email body
    if msg.is_multipart():
        for part in msg.iter_parts():
            if part.get_content_type() == "text/plain" and not part.get_filename():
                email_data["body"] = part.get_content().strip()
    else:
        email_data["body"] = msg.get_content().strip()

    # Extract all attachments
    for part in msg.iter_attachments():
        attachment_filename = part.get_filename()
        file_bytes = part.get_payload(decode=True)

        extracted_text = None
        if attachment_filename.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file_bytes)
        elif attachment_filename.endswith(".docx"):
            extracted_text = extract_text_from_docx(file_bytes)
        elif attachment_filename.endswith(".txt"):
            extracted_text = extract_text_from_txt(file_bytes)
        elif attachment_filename.endswith((".jpg", ".jpeg", ".png")):
            extracted_text = extract_text_from_image(file_bytes)

        email_data["attachments"].append({
            "filename": attachment_filename,
            "content": extracted_text if extracted_text else "Binary file (not extracted)"
        })

    return email_data

# API Endpoint to reset system and upload EML file
@app.post("/emailToJson")
async def reset_system(file: UploadFile = File(...)):
    """
    Upload an .eml file, extract its contents, and generate output JSON.
    """
    try:
        # Read the uploaded .eml file
        eml_bytes = await file.read()
        email_json = process_eml_file(eml_bytes)

        # Generate output JSON filename
        base_filename = os.path.splitext(file.filename)[0]  # Extracts filename without extension
        print(base_filename);
        json_output_path = os.path.join(INPUT_JSON, f"{base_filename}.json")
        print(json_output_path);

        # Save JSON output
        with open(json_output_path, "w", encoding="utf-8") as json_file:
            json.dump(email_json, json_file, indent=4)

        return JSONResponse(content={"message": "File processed successfully", "output_file": json_output_path, "data": email_json})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API Endpoint to reset system and upload EML file
@app.post("/bulk/emailToJson")
async def reset_system():
    """Process all EML files in the input folder and convert to JSON."""
    processed_files = []
    for filename in os.listdir(INPUT_EML):
        if filename.endswith(".eml"):
            eml_path = os.path.join(INPUT_EML, filename)
            json_filename = filename.replace(".eml", ".json")
            json_output_path = os.path.join(INPUT_JSON, json_filename)
            print(json_output_path)
            try:
                email_json = read_eml_file(eml_path)
                # Save JSON output
                with open(json_output_path, "w", encoding="utf-8") as json_file:
                    json.dump(email_json, json_file, indent=4)

                # Delete the processed file
                os.remove(eml_path)
                processed_files.append(json_filename)
            except Exception as e:
                return {"error": f"Failed to process {filename}: {str(e)}"}

    return {"message": "Processing complete", "processed_files": processed_files}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# Run the FastAPI app with:
# uvicorn filename:app --host 0.0.0.0 --port 8000 --reload
