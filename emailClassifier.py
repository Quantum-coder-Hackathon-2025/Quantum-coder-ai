import os
import pdfplumber
import docx
import spacy
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib


# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Define Loan Categories & Subcategories
categories = {"Account Maintenance": ["Update Account Information","Change Payment Schedule","Interest Rate Adjustment","Loan Restructuring"],
    "Payment Processing": ["Scheduled Payment","Prepayment","Partial Payment","Late Payment","Payment Reversal"],
    "Fee Management": ["Fee Payment","Ongoing Fee","Amendment Fee","Reallocation Fee","Letter of Credit Fee"],
    "Money Movement": ["Principal","Interest","Principal + Interest","Principal + Interest + Fee""Timebound","Foreign Currency"],
    "Loan Adjustments": ["Adjustment","AU Transfer"],
    "Commitment Changes": ["Cashless Roll","Increase Commitment","Decrease Commitment"],
    "Closing Requests": ["Closing Notice","Reallocation Principal"],
    "Collateral Management": ["Collateral Valuation","Collateral Substitution","Collateral Release","Collateral Reassessment"],
    "Customer Service Requests": ["Inquiry Handling","Dispute Resolution","Statement Generation","Account Information Requests"],
    "Compliance and Reporting": ["Regulatory Compliance","Internal Audits","External Audits","Compliance Reporting"],
    "Risk Management": ["Credit Risk Assessment","Market Risk Assessment", "Operational Risk Assessment","Risk Mitigation Strategies"],
    "Loan Monitoring": ["Delinquency Monitoring","Covenant Monitoring","Financial Performance Monitoring","Early Warning Systems"],
    "Collections and Recovery": ["Delinquency Notices","Collections Calls","Legal Actions","Recovery Strategies"],
    "Loan Modifications": ["Forbearance Agreements","Loan Extensions","Interest Rate Modifications","Principal Reductions"],
    "Escrow Management": ["Tax Payments","Insurance Payments","Escrow Analysis","Escrow Disbursements"],
    "Documentation Management": ["Document Storage","Document Retrieval","Document Updates","Document Compliance"],
    "Customer Communication": ["Notification Management","Email Communication","SMS Alerts","Customer Portals"],
    "Financial Reporting": ["Loan Performance Reports","Financial Statements","Budgeting and Forecasting","Profit and Loss Analysis"],
    "Technology and Automation": ["Loan Servicing Software","Automated Payment Systems","Digital Communication Channels","Data Analytics and Reporting Tools"],
    "Training and Development": ["Staff Training Programs","Compliance Training","Customer Service Training","Technology Training"]
  }
# Dummy Training Data (Replace with real data)

# Define training data
X_train = [
    "I need to update my contact details on my loan account.",
    "Can you please change my loan repayment schedule to the 15th of each month?",
    "I would like to request a reduction in my loan interest rate.",
    "I am struggling with payments. Is loan restructuring possible?",
    "Can you confirm if my payment for this month has been processed?",
    "I want to make an early payment on my loan. How can I proceed?",
    "I would like to pay only a part of my monthly installment this time.",
    "I missed my loan payment due date. What are the penalties?",
    "Can you reverse the payment I made by mistake?",
    "I need clarification on a fee charged to my loan account.",
    "I see a recurring fee on my statement. Can you explain it?",
    "Can you tell me more about the amendment fee charged?",
    "How do I reallocate a fee to another part of my loan?",
    "I need details on the letter of credit fee in my statement.",
    "I want to transfer money to my loan account to pay off the principal.",
    "How do I send money for both principal and interest together?",
    "Can I make a timebound payment transfer for my loan?",
    "Do you accept foreign currency for my loan payments?",
    "I want to adjust my loan balance due to an overpayment.",
    "Can you transfer my AU to another entity?",
    "I want to increase my loan commitment. What are the steps?",
    "How do I close my loan account and get a final statement?",
    "I need a reallocation of my loan principal amount.",
    "Can you provide me with a copy of my loan agreement?",
    "I am receiving too many notifications. Can you adjust my preferences?",
    "I would like to receive SMS alerts for my loan payments."
]

y_train = [
    ("Account Maintenance", "Update Account Information"),
    ("Account Maintenance", "Change Payment Schedule"),
    ("Account Maintenance", "Interest Rate Adjustment"),
    ("Account Maintenance", "Loan Restructuring"),
    ("Payment Processing", "Scheduled Payment"),
    ("Payment Processing", "Prepayment"),
    ("Payment Processing", "Partial Payment"),
    ("Payment Processing", "Late Payment"),
    ("Payment Processing", "Payment Reversal"),
    ("Fee Management", "Fee Payment"),
    ("Fee Management", "Ongoing Fee"),
    ("Fee Management", "Amendment Fee"),
    ("Fee Management", "Reallocation Fee"),
    ("Fee Management", "Letter of Credit Fee"),
    ("Money Movement", "Principal"),
    ("Money Movement", "Principal + Interest"),
    ("Money Movement", "Timebound"),
    ("Money Movement", "Foreign Currency"),
    ("Loan Adjustments", "Adjustment"),
    ("Loan Adjustments", "AU Transfer"),
    ("Commitment Changes", "Increase Commitment"),
    ("Closing Requests", "Closing Notice"),
    ("Closing Requests", "Reallocation Principal"),
    ("Documentation Management", "Document Retrieval"),
    ("Customer Communication", "Notification Management"),
    ("Customer Communication", "SMS Alerts")
]
def load_Training():
    # Train Model
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
    classifier = joblib.load("loan_request_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

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
