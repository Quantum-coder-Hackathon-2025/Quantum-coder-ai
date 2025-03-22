# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample Data (Loan Service Requests with their respective Categories)
data = {
    'Request_Description': [
        'Change payment schedule for loan',
        'Request for loan restructuring',
        'Inquiry about loan interest rates',
        'Make partial payment for loan',
        'Request for collateral release'
    ],
    'Category': ['Account Maintenance', 'Loan Modifications', 'Customer Service Requests', 'Payment Processing', 'Collateral Management']
}
df = pd.DataFrame(data)
df = pd.DataFrame(data)
print(df)
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Request_Description'], df['Category'], test_size=0.2, random_state=42)

# Vectorizing the text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Training a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Making predictions
y_pred = model.predict(X_test_tfidf)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))