# sentiment_analyzer.py

import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset


# ---------- 1. Load IMDb dataset ----------
print("Loading dataset...")
dataset = load_dataset("imdb")

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")


# ---------- 2. Preprocess text ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


train_df["text"] = train_df["text"].apply(clean_text)
test_df["text"] = test_df["text"].apply(clean_text)


# ---------- 3. Train/Validation Split ----------
X_train, X_val, y_train, y_val = train_test_split(
    train_df["text"], train_df["label"],
    test_size=0.2, random_state=42, stratify=train_df["label"]
)


# ---------- 4. TF-IDF + Logistic Regression ----------
print("Training model...")
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(test_df["text"])

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)


# ---------- 5. Evaluate on Validation ----------
val_preds = model.predict(X_val_tfidf)
print("\nValidation Report:\n")
print(classification_report(y_val, val_preds))

cm = confusion_matrix(y_val, val_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix (Validation)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# ---------- 6. Evaluate on Test Data ----------
test_preds = model.predict(X_test_tfidf)
print("\nTest Report:\n")
print(classification_report(test_df["label"], test_preds))


# ---------- 7. Predict Custom Sentences ----------
def predict_sentiment(text):
    text_clean = clean_text(text)
    tfidf = vectorizer.transform([text_clean])
    pred = model.predict(tfidf)[0]
    return "Positive 😊" if pred == 1 else "Negative 😡"


# ---------- 8. Example Predictions ----------
print("\nCustom Predictions:")
print("Example 1:", predict_sentiment("The movie was absolutely fantastic!"))
print("Example 2:", predict_sentiment("I hated the film, it was boring and long."))
print("Example 2:", predict_sentiment("I love to play Cricket."))
