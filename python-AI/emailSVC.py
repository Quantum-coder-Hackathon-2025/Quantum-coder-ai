import os
import pdfplumber
import docx
import spacy
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import joblib
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
# from transformers import pipeline


#Load summarization model
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Define Loan Categories & Subcategories
import pandas as pd

# Load the Excel sheet
df = pd.read_excel("loan_categories.xlsx")

# Display first few rows
# print(df.head())
category_dict = df["Category"].unique()
# print(categories)
categories= df.groupby("Category")["Subcategory"].apply(list).to_dict()

# def generate_summary(text):
#     if len(text.split()) >50:
#         summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
#         return summary[0]['summary_text']
#     return text

# Load Training Data
def load_Training():
    # Train Model
    # Define training data
    # Load the Excel file
    df = pd.read_excel("TrainingSet.xlsx")
    X_train = df["Training Request"].tolist()
    y_train = list(zip(df["Request Type"], df["Sub-Request Type"]))
    vectorizer = TfidfVectorizer()
    classifier = SVC(kernel="linear", probability=True)
    pipeline = Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])

    X_train_vec = vectorizer.fit_transform(X_train)
    y_train_labels = np.array([list(categories.keys()).index(cat) for cat, sub in y_train])

    classifier.fit(X_train_vec, y_train_labels)
    # Save trained model and vectorizer
    joblib.dump(classifier, "loan_request_svc_model.pkl")
    joblib.dump(vectorizer, "vectorizer_svc.pkl")
    print("Model and vectorizer saved successfully!")

# Initialize FastAPI App
app = FastAPI()

# Extract text from attachments
def extract_text_from_attachment(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        # Perform OCR on image files
        image = Image.open(file_path)
        # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        pytesseract.pytesseract.tesseract_cmd = r"/usr/local/Cellar/tesseract/5.5.0_1/bin/tesseract"
        text = pytesseract.image_to_string(image)
    return text

@app.get("/loadTrainData")
def load_training_api():
    load_Training()
    return {"message loaded training data"}

# Extract details using NLP
def extract_details(text):
    doc = nlp(text)
    extracted_info = {"Dates": [], "Amounts": []}

    for ent in doc.ents:
        if ent.label_ == "DATE":
            extracted_info["Dates"].append(ent.text)
        if ent.label_ == "MONEY":
            extracted_info["Amounts"].append(ent.text)

    return extracted_info

# API Endpoint to Process Emails
@app.post("/classify/")
async def classify_request(email_text: str = Form(...), attachment: UploadFile = File(None)):
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

    # Transform text for classification
    # Load saved classifier and vectorizer
    classifier = joblib.load("loan_request_svc_model.pkl")
    vectorizer = joblib.load("vectorizer_svc.pkl")

    input_vec = vectorizer.transform([combined_text])
    category_index = classifier.predict(input_vec)[0]
    category = list(categories.keys())[category_index]

    # Subcategory Prediction (Basic Match)
    subcategories = categories[category]
    subcategory = next((sub for sub in subcategories if sub.lower() in combined_text.lower()), "Unknown")

    # Extract additional details
    extracted_details = extract_details(combined_text)

    # Confidence Score (Placeholder for real probability scores)
    confidence_score = round(np.random.uniform(60, 95), 2)

    # Final Output
    response = {
        "Category": category,
        "Subcategory": subcategory,
        "Confidence_Score": confidence_score,
        "Extracted_Details": extracted_details,
        "Summary": combined_text[:500]  # First 500 chars of the text
    }

    return response
