import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

def spacy_summarize(text, num_sentences=2):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    # Compute TF-IDF scores
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    
    # Rank sentences based on sum of TF-IDF scores
    sentence_scores = X.sum(axis=1)
    ranked_sentences = [sent for _, sent in sorted(zip(sentence_scores, sentences), reverse=True)]
    
    return " ".join(ranked_sentences[:num_sentences])

text = """Google Cloud has announced a new AI model for text summarization. 
The model is capable of summarizing long documents into concise summaries.
It is based on a transformer architecture and optimized for speed and accuracy.
Companies are excited about integrating this AI model into their workflow."""

summary = spacy_summarize(text, 2)
print("Summary:", summary)