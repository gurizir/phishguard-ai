import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD & PREPARE DATA
# =========================

df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Add Indian phishing examples
indian_phishing = [
    ("spam", "Your SBI account will be blocked today. Verify immediately."),
    ("spam", "Income Tax Department: Refund pending. Submit PAN details now."),
    ("spam", "UPI Alert: Rs.5000 debited. If not you, click here to reverse."),
    ("spam", "Aadhaar suspended due to verification failure. Update KYC immediately."),
    ("spam", "Congratulations! You have won Rs.10,00,000 in KBC lottery."),
    ("spam", "IRCTC account locked. Confirm details to restore access."),
    ("spam", "Electricity bill overdue. Pay now to avoid disconnection."),
    ("spam", "Dear customer, your SIM will be deactivated today. Verify instantly.")
]

indian_df = pd.DataFrame(indian_phishing, columns=["label", "text"])
df = pd.concat([df, indian_df], ignore_index=True)

# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

#print(df['label'].value_counts())

# =========================
# TRAIN MODEL
# =========================

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

#print("\nAccuracy:", accuracy_score(y_test, y_pred))
#print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# EXPLAINABILITY ENGINE
# =========================

def explain_message(message, model, vectorizer, top_n=5):
    vec = vectorizer.transform([message])
    prob = model.predict_proba(vec)[0][1]
    risk_score = int(prob * 100)

    feature_names = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = vec.toarray()[0]
    top_indices = tfidf_scores.argsort()[-top_n:][::-1]
    risky_words = feature_names[top_indices]

    high_risk_keywords = [
        "blocked", "verify", "urgent", "account",
        "upi", "kyc", "refund", "deactivated", "suspended"
    ]

    message_lower = message.lower()
    keyword_hits = [w for w in high_risk_keywords if w in message_lower]

    # Hybrid decision
    if risk_score >= 30 or len(keyword_hits) >= 2:
        pred = 1
    else:
        pred = 0

    explanation = (
        "This message shows phishing characteristics such as urgency and "
        "sensitive account-related terms: "
        + ", ".join(set(list(risky_words) + keyword_hits))
        + "."
    )

    return pred, risk_score, risky_words, explanation


###########################################################################################

joblib.dump(model, "model/phishguard_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model and vectorizer saved.")