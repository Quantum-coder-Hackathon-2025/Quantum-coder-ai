# 🔹 Approach
# Train a Text Classification Model
# Use predefined keywords and descriptions as training data.
# Convert the text to numerical format using TF-IDF Vectorization.
# Train a Naïve Bayes Classifier or use cosine similarity for confidence scoring.
# Predict Category & Confidence Score
# Given a request, the model predicts a category with a probability (confidence score).

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Define loan service categories with descriptions
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

# Create a TF-IDF vectorizer and Naïve Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(texts, labels)

# Function to classify a loan service request
def classify_request(request):
    pred_prob = model.predict_proba([request])  # Get category probabilities
    top_idx = np.argmax(pred_prob)  # Index of highest confidence category
    predicted_category = labels[top_idx]
    confidence_score = pred_prob[0][top_idx]  # Confidence score
    
    return predicted_category, round(confidence_score * 100, 2)  # Convert to percentage

# Example Usage
if __name__ == "__main__":
    request = input("Enter a loan service request: ")
    category, confidence = classify_request(request)
    print(f"Predicted Category: {category} (Confidence: {confidence}%)")
