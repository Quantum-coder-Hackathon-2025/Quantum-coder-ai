import os
from fastapi import FastAPI, File, UploadFile, Form
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

# API route to process an email and its attachments
@app.post("/process_email/")
async def process_email(email_text: str = Form(...), files: Optional[List[UploadFile]] = File(None), threshold: float = 0.9):
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
    if files and isinstance(files, list):   # Check if files are provided
        for file in files:
            # Save the uploaded file locally
            file_path = f"./{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            
            # Determine file type and extract text
            if file.filename.endswith(".png") or file.filename.endswith(".jpg"):  # Image
                text = extract_text_from_image(file_path)
            elif file.filename.endswith(".pdf"):  # PDF
                text = extract_text_from_pdf(file_path)
            elif file.filename.endswith(".docx"):  # Word document
                text = extract_text_from_docx(file_path)
            elif file.filename.endswith(".txt"):  # Plain text file
                text = extract_text_from_txt(file_path)
            else:
                attachment_results.append({"filename": file.filename, "error": "Unsupported file type"})
                continue
            
            # Get embedding for extracted text
            attachment_embedding = get_bert_embedding(text)
           
            # Remove the file after processing
            os.remove(file_path)  

            # Check for duplicates in in-memory storage for attachments
            for stored_key, stored_embedding in list(embedding_storage.items()):  # Create a copy of the items
                similarity = cosine_similarity(attachment_embedding.unsqueeze(0), stored_embedding.unsqueeze(0))[0][0]
                similarity = float(similarity)
                if similarity > threshold:
                    duplicates_attachments.append({"duplicate_with": stored_key, "similarity": similarity})              
                    # Append results for attachments
                attachment_results.append({"filename": file.filename, "Attachment Similarity": similarity})
            # Add attachment embedding to storage
            embedding_storage[file.filename] = attachment_embedding        
    # Add email text embedding to storage
    embedding_storage[email_text] = email_embedding   
    # Check if duplicates_email has similarity score 1 then return result as duplicateIndicator as True
    duplicateIndicator = False
    #if attachments are not given then check for duplicates_email result only
    if not files:   
        for duplicate in duplicates_email:
            duplicateIndicator = True
            break
    #if attachments are provided and content of the email itself is not a duplicate then mark duplicateIndicator as false
    if files and not duplicates_email:
        for duplicate in duplicates_attachments:
            if duplicate["similarity"] > 0.95:
                # loop through duplicate_email and check if any of the duplicates_email has similarity score 1 then mark duplicateIndicator as True
                for duplicate in duplicates_email:
                    if duplicate["similarity"] >0.80:
                        duplicateIndicator = True
                        break
                
    #if attachments are provided and content of the email is a duplicate then loop through duplicate_attachments 
    if files and duplicates_email:
        for duplicate in duplicates_attachments:
            if duplicate["similarity"] > threshold:
                duplicateIndicator = True
                break
    
    return {"duplicateIndicator": duplicateIndicator, "email_text": email_text, "duplicates_email": duplicates_email, "duplicates_attachments": duplicates_attachments, "attachments_results": attachment_results, "duplicateIndicator": duplicateIndicator} 
   