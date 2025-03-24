import json
import email
import pytesseract
from PIL import Image
from email import policy
from email.parser import BytesParser
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document

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
        "from": msg["From"],
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

# Example Usage
eml_file_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/Quantum-coder-ai/python-AI/output/sample3.eml"  # Update with the correct file path
email_json = read_eml_file(eml_file_path)

# Save JSON output
json_output_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/Quantum-coder-ai/python-AI/output/email_output.json"
with open(json_output_path, "w", encoding="utf-8") as json_file:
    json.dump(email_json, json_file, indent=4)

print(f"Extracted email data saved to {json_output_path}")
