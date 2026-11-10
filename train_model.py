import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading advanced dataset...")

data = pd.read_csv("reviews_advanced.csv")
data = data.dropna(subset=["review", "sentiment"])

X = data["review"].astype(str)
y = data["sentiment"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("-------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Save evaluation metrics
with open("evaluation_metrics.json", "w") as file:
    json.dump({
        "accuracy": accuracy,
        "report": report
    }, file, indent=4)

# Save confusion matrix
labels = ["Negative", "Neutral", "Positive"]

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_df.to_csv("confusion_matrix.csv")

print("\nModel, vectorizer and evaluation files saved!")