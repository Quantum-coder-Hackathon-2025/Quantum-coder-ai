from fastapi import FastAPI, File, UploadFile
import pdfplumber
import docx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# Loan categories
categories = {
    "Account Maintenance": "update account information, change payment schedule, interest rate adjustment, loan restructuring",
    "Payment Processing": "scheduled payment, prepayment, partial payment, late payment, payment reversal",
    "Fee Management": "fee payment, ongoing fee, amendment fee, reallocation fee, letter of credit fee",
    "Money Movement": "inbound money movement, outbound money movement, principal, interest, foreign currency, timebound",
    "Loan Adjustments": "adjustment, AU transfer",
    "Commitment Changes": "cashless roll, increase commitment, decrease commitment",
    "Closing Requests": "closing notice, reallocation principal",
    "Collateral Management": "collateral valuation, collateral substitution, collateral release, collateral reassessment",
    "Customer Service Requests": "inquiry handling, dispute resolution, statement generation, account information requests",
    "Compliance and Reporting": "regulatory compliance, internal audits, external audits, compliance reporting",
    "Risk Management": "credit risk assessment, market risk assessment, operational risk assessment, risk mitigation strategies",
    "Loan Monitoring": "delinquency monitoring, covenant monitoring, financial performance monitoring, early warning systems",
    "Collections and Recovery": "delinquency notices, collections calls, legal actions, recovery strategies",
    "Loan Modifications": "forbearance agreements, loan extensions, interest rate modifications, principal reductions",
    "Escrow Management": "tax payments, insurance payments, escrow analysis, escrow disbursements",
    "Documentation Management": "document storage, document retrieval, document updates, document compliance",
    "Customer Communication": "notification management, email communication, sms alerts, customer portals",
    "Financial Reporting": "loan performance reports, financial statements, budgeting and forecasting, profit and loss analysis",
    "Technology and Automation": "loan servicing software, automated payment systems, digital communication channels, data analytics",
    "Training and Development": "staff training, compliance training, customer service training, technology training"
}

# Train the Model
# vectorizer = TfidfVectorizer()
vectorizer = TfidfVectorizer(ngram_range=(1,2)) 
X_train = vectorizer.fit_transform(list(categories.values()))
y_train = np.eye(len(categories))

# classifier = MultiOutputClassifier(LogisticRegression())
classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))
classifier.fit(X_train, y_train)

# Function to extract text from files
def extract_text(file):
    text = ""
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(file.file) as pdf:
            text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file.file)
        text = " ".join([para.text for para in doc.paragraphs])
    elif file.filename.endswith(".txt"):
        text = file.file.read().decode("utf-8")
    return text.strip()

# API Endpoint to classify emails with attachments
@app.post("/classify-email")
async def classify_email(email_body: str, file: UploadFile = None):
    print(email_body)
    text_data = email_body
    if file:
        text_data += " " + extract_text(file)

    print(text_data)
    X_test = vectorizer.transform([text_data])
    y_pred = np.array(classifier.predict_proba(X_test))

    print(X_test)

    # Ensure y_pred is a list of arrays
    y_pred = classifier.predict_proba(X_test)

    print("y_pred:", y_pred)
    for i, prob_array in enumerate(y_pred):
        print(f"Category: {list(categories.keys())[i]}, Probabilities: {prob_array}")


    # Extract probabilities correctly
    results = {}

    for i, prob_array in enumerate(y_pred):
        prob_array = np.array(prob_array).flatten()  # Ensure it's a 1D array

        # Extracting the probability of belonging to the category
        if prob_array.shape[0] > 1:  # If two values exist
            confidence = prob_array[1]  # Take the positive class probability
        else:
            confidence = prob_array[0]  # If only one value exists, take it

        # confidence = float(confidence)  # Convert to Python float for safety

        results[list(categories.keys())[i]] = round(confidence * 100, 2)

    print("Raw confidence scores:")
    for category, confidence in results.items():
        print(f"{category}: {confidence}%")
        
    return {"predicted_categories": results}
