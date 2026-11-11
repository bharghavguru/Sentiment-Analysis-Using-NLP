
import re
import joblib

# Load the trained model and TF-IDF vectorizer
model = joblib.load("models/best_sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# Clean the input review
def clean_review(text):
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Predict sentiment
def predict_sentiment(review):
    cleaned_review = clean_review(review)

    review_tfidf = vectorizer.transform([cleaned_review])

    prediction = model.predict(review_tfidf)[0]

    probability = model.predict_proba(review_tfidf)[0]

    confidence = max(probability) * 100

    return prediction, confidence


# Main program
print("===================================")
print("     MOVIE REVIEW SENTIMENT ANALYSIS")
print("===================================")

review = input("\nEnter your movie review: ")

sentiment, confidence = predict_sentiment(review)

print("\nPrediction Result")
print("------------------------------")
print("Sentiment:", sentiment.upper())
print(f"Model confidence: {confidence:.2f}%")
print("------------------------------")
