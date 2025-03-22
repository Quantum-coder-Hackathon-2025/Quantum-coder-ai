import os
import pdfplumber
import docx
import json
import spacy
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Define categories
categories = {
    "Account Maintenance": ["Update Account Information", "Change Payment Schedule", "Interest Rate Adjustment", "Loan Restructuring"],
    "Payment Processing": ["Scheduled Payment", "Prepayment", "Partial Payment", "Late Payment", "Payment Reversal"],
    "Fee Management": ["Fee Payment", "Ongoing Fee", "Amendment Fee", "Reallocation Fee", "Letter of Credit Fee"],
}

# Dummy training data (use a real dataset for better accuracy)
X_train = ["change my payment date", "interest rate adjustment request", "request to restructure loan"]
y_train = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# Train Classifier
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
classifier.fit(X_train_vec, y_train)

# Initialize FastAPI App
app = FastAPI()

# Function to extract text from attachments
def extract_text_from_attachment(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to extract relevant details using NLP
def extract_details(text):
    doc = nlp(text)
    extracted_info = {}

    for ent in doc.ents:
        if ent.label_ == "DATE":
            extracted_info["Date"] = ent.text
        if ent.label_ == "MONEY":
            extracted_info["Amount"] = ent.text

    return extracted_info

# API Endpoint to Process Emails
@app.post("/process_email/")
async def process_email(email_text: str = Form(...), attachment: UploadFile = File(None)):
    attachment_text = ""

    # Save and extract text from the attachment
    if attachment:
        file_path = f"temp_{attachment.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await attachment.read())

        attachment_text = extract_text_from_attachment(file_path)
        os.remove(file_path)  # Delete after processing

    # Combine email and attachment text
    combined_text = email_text + "\n" + attachment_text

    # Transform input text for classification
    input_vec = vectorizer.transform([combined_text])
    y_pred = classifier.predict_proba(input_vec)

    results = {}
    for i, prob_array in enumerate(y_pred):
        confidence = float(prob_array[0][1]) if len(prob_array[0]) > 1 else float(prob_array[0][0])
        if confidence > 0.05:  # Adjust threshold if needed
            results[list(categories.keys())[i]] = round(confidence * 100, 2)

    # Extract additional details
    extracted_details = extract_details(combined_text)

    # Final Output
    response = {
        "Predicted_Categories": results,
        "Extracted_Details": extracted_details
    }

    return response
