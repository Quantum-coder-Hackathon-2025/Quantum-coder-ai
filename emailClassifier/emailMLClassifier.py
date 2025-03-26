import os
import pandas as pd
import pdfplumber
import docx
import spacy
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from scipy.special import softmax  # Apply softmax to normalize probabilities
import joblib
from PIL import Image
import pytesseract
from sklearn.metrics.pairwise import cosine_similarity
import re
from dateutil import parser
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# from transformers import pipeline


#Load summarization model
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Define Loan Categories & Subcategories

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
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    pipeline = Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])

    X_train_vec = vectorizer.fit_transform(X_train)
    y_train_labels = np.array([list(categories.keys()).index(cat) for cat, sub in y_train])

    classifier.fit(X_train_vec, y_train_labels)
    # Save trained model and vectorizer
    joblib.dump(classifier, "loan_request_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Model and vectorizer saved successfully!")

# Initialize FastAPI App
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
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSER_ACT")
        text = pytesseract.image_to_string(image)
    return text

@app.get("/loadTrainingData")
def load_training_api():
    load_Training()
    return {"message loaded training data"}

# Extract details using NLP
def extract_details(text):
    doc = nlp(text)
    extracted_info = {
        "Dates": [],
        "Amounts": [],
        "Names": [],
        "Organizations": [],
        "Emails": []
    }

    # Regex patterns for dates & amounts
    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # 25/03/2025 or 03-25-2025
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s?\d{1,2},?\s?\d{4}\b"  # March 25, 2025
    ]
    amount_patterns = [
        r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?",  # $1,200.50
        r"\b\d{1,3}(?:,\d{3})*\s?(?:USD|dollars)\b",  # 1,200 USD or 1200 dollars
        r"\b\d+\.\d{2}\b"  # 1200.50
    ]
    
    # Extract Dates using regex
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                parsed_date = parser.parse(match, fuzzy=True).date()  # Convert to YYYY-MM-DD
                extracted_info["Dates"].append(str(parsed_date))
            except:
                continue  # Ignore invalid dates

    # Extract Amounts using regex
    for pattern in amount_patterns:
        matches = re.findall(pattern, text)
        extracted_info["Amounts"].extend(matches)
    
    # Extract details using Spacy
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            extracted_info["Names"].append(ent.text)
        elif ent.label_ == "ORG":
            extracted_info["Organizations"].append(ent.text)
        elif ent.label_ == "EMAIL":
            extracted_info["Emails"].append(ent.text)

    # Remove duplicates & sort
    for key in extracted_info:
        extracted_info[key] = list(set(extracted_info[key]))

    return extracted_info

def generate_summary(text,num_sentences=2):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    # Compute TF-IDF scores
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    
    # Rank sentences based on sum of TF-IDF scores
    sentence_scores = X.sum(axis=1)
    ranked_sentences = [sent for _, sent in sorted(zip(sentence_scores, sentences), reverse=True)]
    
    return " ".join(ranked_sentences[:num_sentences])

def clean_text(text):
    # Remove new lines, tabs, and the word "COMMENT" (case insensitive)
    text = re.sub(r"\n|\t", " ", text)  # Replace \n and \t with space
    text = re.sub(r"\bCOMMENT\b", "", text, flags=re.IGNORECASE)  # Remove "COMMENT"
    text = re.sub(r"[^a-zA-Z0-9.,'\"?!\s]", "", text)  # Remove special characters (except punctuation)
    
    # Remove extra spaces
    return re.sub(r"\s+", " ", text).strip()

def clean_json(data):
    if isinstance(data, dict):
        return {k: clean_json(v) for k, v in data.items() if v not in (None, "", [], {})}
    elif isinstance(data, list):
        return [clean_json(item) for item in data if item not in (None, "", [], {})]
    return data


# Classify text using trained model
def classify_text(text, alpha=0.6, beta=0.4):
    """Classify loan requests & improve confidence score calculations."""
    
    # Load Model & Vectorizer
    classifier = joblib.load("loan_request_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    try:
        # Transform Input Text
        input_vec = vectorizer.transform([text])
        if input_vec.shape[1] == 0:  # No valid tokens
            return [{"Category": "Unknown", "Subcategory": "Insufficient Data", "confidence_score": 0}]

        raw_probabilities = classifier.predict_proba(input_vec)[0]
        
        # Apply Softmax to Normalize Probabilities
        probabilities = softmax(raw_probabilities)  

        unique_results = {}
        
        for category, prob in zip(categories.keys(), probabilities):
            if prob <= 0:  # Ignore zero-probability categories
                continue

            subcategories = categories[category]
            sub_vecs = vectorizer.transform(subcategories)
            text_similarities = cosine_similarity(input_vec, sub_vecs)[0]  # Get cosine similarities
            
            best_subcategory, best_sub_confidence = None, 0
            for sub, sim in zip(subcategories, text_similarities):
                sub_confidence = alpha * prob + beta * sim  # Weighted sum of prob & similarity
                sub_confidence = round(sub_confidence, 2)  # Round for better readability

                if sub_confidence > best_sub_confidence:
                    best_subcategory, best_sub_confidence = sub, sub_confidence

            if best_subcategory and best_sub_confidence > 0:
                key = (category, best_subcategory)
                if key not in unique_results or unique_results[key]["confidence_score"] < best_sub_confidence:
                    unique_results[key] = {
                        "Category": category,
                        "Subcategory": best_subcategory,
                        "confidence_score": best_sub_confidence
                    }

        # Normalize Confidence Scores (Optional)
        max_conf = max([x["confidence_score"] for x in unique_results.values()], default=1)
        for item in unique_results.values():
            item["confidence_score"] = round(item["confidence_score"] / max_conf, 2)*100

        # Return Top 2 Results with Best Scores
        sorted_results = sorted(unique_results.values(), key=lambda x: x["confidence_score"], reverse=True)
        return sorted_results[:2] if sorted_results else []
    except ValueError:
        return [{"Category": "Error", "Subcategory": "Empty Input", "confidence_score": 0}]


# FastAPI endpoint to process emails and multiple attachments
@app.post("/process_email/attachemnts")
async def process_email(email_text: str = Form(...), attachments: list[UploadFile] = File(None)):
    results = []    
    #Process email body
    email_classification = classify_text(email_text)
    if email_classification:  # Only add if classification exists
        results.append({
            "Summary": clean_text(generate_summary(email_text)),
            "Classification": email_classification,
            "Extracted_Details": extract_details(email_text)
        })
    else:
        email_text = ""

    # Process Attachments
    if attachments:
        results = []
        for attachment in attachments:
            file_path = f"temp_{attachment.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await attachment.read())

            text_content = email_text + "\n" + extract_text_from_attachment(file_path)
            os.remove(file_path)  # Delete after processing
            
            if not text_content.strip():
                results.append({
                    "Source": "Not available",
                    "Message": "Not valid content"
                })
                continue
            
            attachment_classification = classify_text(text_content)
            if attachment_classification:  # Only add if classification exists
                results.append({
                    "Summary": clean_text(generate_summary(text_content)),
                    "Classification": attachment_classification,
                    "Extracted_Details": extract_details(text_content)
                })

    results = clean_json(results)
    # If no classifications exist, return an empty response
    return {"duplicateCheck":"Yet to Update","Results": results} if results else {"duplicateCheck":"","Results": []}

@app.post("/process_email/generatedInput")
async def process_email(email_request: EmailRequest):
    results = []    
    email_text = email_request.body
        #Process email body
    email_classification = classify_text(email_text)
    if email_classification:  # Only add if classification exists
        results.append({
            "Summary": clean_text(generate_summary(email_text)),
            "Classification": email_classification,
            "Extracted_Details": extract_details(email_text)
        })
    else:
        email_text = ""

    # Process Attachments
    if email_request.attachments:
        results = []
        for attachment in email_request.attachments:
            # file_path = f"temp_{attachment.filename}"
            # with open(file_path, "wb") as buffer:
            #     buffer.write(await attachment.read())

            text_content = email_text + "\n" + attachment.content
            # os.remove(file_path)  # Delete after processing
            
            if not text_content.strip():
                results.append({
                    "Source": "Not available",
                    "Message": "Not valid content"
                })
                continue
            
            attachment_classification = classify_text(text_content)
            if attachment_classification:  # Only add if classification exists
                results.append({
                    "Summary": clean_text(generate_summary(text_content)),
                    "Classification": attachment_classification,
                    "Extracted_Details": extract_details(text_content)
                })
    results = clean_json(results)
    # If no classifications exist, return an empty response
    return {"duplicateCheck":"Yet to Update","Results": results} if results else {"duplicateCheck":"","Results": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)