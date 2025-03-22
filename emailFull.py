import os
import email
import imaplib
import numpy as np
from email import policy
from email.parser import BytesParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import pdfplumber
import docx

# Email server details (Gmail example)
EMAIL_HOST = "imap.gmail.com"
EMAIL_USER = "your_email@gmail.com"
EMAIL_PASS = "your_password"

# Loan service categories
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

# Prepare training data
texts = list(categories.values())  # Training text (category descriptions)
labels = list(categories.keys())   # Labels (categories)

# Create TF-IDF vectorizer and multi-label classifier
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(texts)
y_train = np.eye(len(labels))  # One-hot encoding for multi-label classification

# Train model using Logistic Regression with MultiOutputClassifier
classifier = MultiOutputClassifier(LogisticRegression())
classifier.fit(X_train, y_train)

# Function to extract email content
def extract_email_content(msg):
    text = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain" or content_type == "text/html":
                text += part.get_payload(decode=True).decode(errors="ignore") + " "
    else:
        text = msg.get_payload(decode=True).decode(errors="ignore")
    return text.strip()

# Function to extract text from attachments
def extract_text_from_attachment(part):
    content_type = part.get_content_type()
    filename = part.get_filename()
    text = ""

    if filename:
        filepath = f"/tmp/{filename}"
        with open(filepath, "wb") as f:
            f.write(part.get_payload(decode=True))

        # Extract text from PDF
        if filename.endswith(".pdf"):
            with pdfplumber.open(filepath) as pdf:
                text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
        
        # Extract text from DOCX
        elif filename.endswith(".docx"):
            doc = docx.Document(filepath)
            text = " ".join(para.text for para in doc.paragraphs)
        
        # Extract text from TXT
        elif filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

        os.remove(filepath)  # Clean up temporary file

    return text.strip()

# Function to read and classify emails
def classify_email():
    mail = imaplib.IMAP4_SSL(EMAIL_HOST)
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select("inbox")

    _, messages = mail.search(None, "UNSEEN")  # Fetch unread emails
    message_ids = messages[0].split()

    for msg_id in message_ids:
        _, msg_data = mail.fetch(msg_id, "(RFC822)")
        raw_email = msg_data[0][1]

        # Parse email
        msg = BytesParser(policy=policy.default).parsebytes(raw_email)
        email_content = extract_email_content(msg)

        # Extract text from attachments
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                email_content += " " + extract_text_from_attachment(part)

        # Predict categories
        X_test = vectorizer.transform([email_content])
        y_pred = classifier.predict_proba(X_test)

        # Assign categories with confidence scores
        results = {
            labels[i]: round(y_pred[0][i] * 100, 2) for i in range(len(labels)) if y_pred[0][i] > 0.2
        }

        print(f"\n📩 Email Subject: {msg['subject']}")
        print(f"Predicted Categories: {results}")

# Run the email classifier
if __name__ == "__main__":
    classify_email()
