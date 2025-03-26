import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.params import Body
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from PIL import Image
import fitz  # PyMuPDF for PDF handling
import docx  # for Word documents
from typing import List, Dict, Optional

# Initialize FastAPI app
app = FastAPI()

class Attachment(BaseModel):
    filename: str
    content: str

class EmailRequest(BaseModel):
    from_email: str
    to: str
    subject: str
    body: str
    attachments: list[Attachment]

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# In-memory storage for embeddings
embedding_storage: Dict[str, torch.Tensor] = {}

# Function to extract embeddings using BERT
def get_bert_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
        embedding = output.last_hidden_state.mean(dim=1).squeeze()
    return embedding

# Function to extract text from images using OCR
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract text from Word documents
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = " ".join([paragraph.text for paragraph in doc.paragraphs])
    return text

# Function to extract text from TXT files
def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


@app.get("/clearDict")
async def clearnDictioanry():
    embedding_storage.clear()
    return {"message": "Dictionary cleared"}


@app.post("/duplicateCheck")
async def process_email(email_request: EmailRequest, threshold: float = 0.9):
    email_text = email_request.body

    # Get embedding for email text
    email_embedding = get_bert_embedding(email_text)

    # Check for duplicates in in-memory storage
    duplicates_email = []
    for stored_key, stored_embedding in list(embedding_storage.items()):  # Create a copy of the items
        similarity = cosine_similarity(email_embedding.unsqueeze(0), stored_embedding.unsqueeze(0))[0][0]
        similarity = float(similarity)
        if similarity > threshold:
            duplicates_email.append({"duplicate_with": stored_key, "similarity": similarity})

    # Process attachments if provided
    attachment_results = []
    duplicates_attachments = []
    if email_request.attachments:
        for attachment in email_request.attachments:
            # Get embedding for extracted text
            attachment_embedding = get_bert_embedding(attachment.content)
           
            # Check for duplicates in in-memory storage for attachments
            for stored_key, stored_embedding in list(embedding_storage.items()):  # Create a copy of the items
                similarity = cosine_similarity(attachment_embedding.unsqueeze(0), stored_embedding.unsqueeze(0))[0][0]
                similarity = float(similarity)
                if similarity > threshold:
                    duplicates_attachments.append({"duplicate_with": stored_key, "similarity": similarity})              
                    # Append results for attachments
                attachment_results.append({"filename": attachment.filename, "Attachment Similarity": similarity})
            # Add attachment embedding to storage
            embedding_storage[attachment.filename] = attachment_embedding        
    # Add email text embedding to storage
    embedding_storage[email_text] = email_embedding    
    # Check if duplicates_email has similarity score 1 then return result as duplicateIndicator as True
    duplicateIndicator = False
    #if attachments are not given then check for duplicates_email result only
    if not email_request.attachments:   
        for duplicate in duplicates_email:
            duplicateIndicator = True
            break
    #if attachments are provided and content of the email itself is not a duplicate then mark duplicateIndicator as false
    if email_request.attachments and not duplicates_email:
        for duplicate in duplicates_attachments:
            if duplicate["similarity"] > 0.95:
                # loop through duplicate_email and check if any of the duplicates_email has similarity score 1 then mark duplicateIndicator as True
                for duplicate in duplicates_email:
                    if duplicate["similarity"] >0.80:
                        duplicateIndicator = True
                        break
                
    #if attachments are provided and content of the email is a duplicate then loop through duplicate_attachments 
    if email_request.attachments and duplicates_email:
        for duplicate in duplicates_attachments:
            if duplicate["similarity"] > threshold:
                duplicateIndicator = True
                break
    
    return {"duplicateIndicator": duplicateIndicator, "email_text": email_text, "duplicates_email": duplicates_email, "duplicates_attachments": duplicates_attachments, "attachments_results": attachment_results, "duplicateIndicator": duplicateIndicator} 
   