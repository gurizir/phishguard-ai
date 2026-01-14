from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from spellchecker import SpellChecker
spell = SpellChecker()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = joblib.load("model/phishguard_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# =========================
# EXPLAIN FUNCTION
# =========================

def analyze_message(message):
    vec = vectorizer.transform([message])

    # ML probability
    prob = model.predict_proba(vec)[0][1]

    # TF-IDF explanation
    feature_names = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = vec.toarray()[0]
    message_words=set(message.lower().split())
    candidate_words=feature_names[tfidf_scores.argsort()[-15:][::-1]]
    top_words=[
        w for w in candidate_words
        if w in message_words and len(w)>2
    ][:5]

    # Rule-based keywords
    high_risk_keywords = [
        "blocked", "verify", "urgent", "account",
        "upi", "kyc", "refund", "deactivated",
        "credit", "card"
    ]

    message_lower = message.lower()
    # Spelling anomaly detection
    words = [w for w in message_lower.split() if w.isalpha()]
    misspelled = spell.unknown(words)

    spelling_error_rate = len(misspelled) / max(len(words), 1)

    linguistic_risk=spelling_error_rate>0.2
    # Grammar anomaly detection
    grammar_flags = 0

    grammar_phrases = [
        "has been block",
        "will be blocked today",
        "verify immediately",
        "avoid block",
       "account will be block",
       "do the needful"
    ]

    for phrase in grammar_phrases:
       if phrase in message_lower:
           grammar_flags += 1

    if message.isupper():
       grammar_flags += 1

    if message.count("!") >= 2:
       grammar_flags += 1

    if "kindly" in message_lower and "verify" in message_lower:
       grammar_flags += 1

    if "please" in message_lower and ("blocked" in message_lower or "suspended" in message_lower):
      grammar_flags += 1

    if len(message_lower.split()) < 6 and ("blocked" in message_lower or "verify" in message_lower):
     grammar_flags += 1

    grammar_risk = grammar_flags >= 1

    hits = [w for w in high_risk_keywords if w in message_lower]


    # Attack pattern classification
    attack_patterns = []

    if any(word in message_lower for word in ["card", "upi", "refund", "debit", "credit", "payment"]):
     attack_patterns.append("Financial Threat")

    if any(word in message_lower for word in ["verify", "login", "password", "otp", "update", "kyc"]):
     attack_patterns.append("Credential Harvesting")

    if any(word in message_lower for word in ["blocked", "suspended", "deactivated"]):
     attack_patterns.append("Account Impersonation")

    if any(word in message_lower for word in ["urgent", "immediately", "today", "now"]):
     attack_patterns.append("Urgency / Fear-based Scam")

    if not attack_patterns:
     attack_patterns.append("General Message")


    # Threat level (presentation)
    if prob >= 0.6:
        threat = "High"
    elif prob >= 0.3 or len(hits) >= 1 or linguistic_risk or grammar_risk:
        threat = "Medium"
    else:
        threat = "Low"


    # Final phishing decision (hybrid)
    phishing = prob >= 0.3 or len(hits) >= 1 or linguistic_risk or grammar_risk

    explanation_parts=[]
    if hits:
        explanation_parts.append("use of sensitive financial terms")
    if misspelled:
        explanation_parts.append("spelling mistakes such as "+", ".join(misspelled))
    if grammar_risk:
        explanation_parts.append("unnatural grammar patterns common in scam messages")

    if explanation_parts:
       explanation = "This message is flagged due to " + ", ".join(explanation_parts) + "."
    else:
       explanation = "No significant phishing indicators were detected in this message."

    return phishing, threat, top_words, explanation, attack_patterns


# =========================
# FLASK APP
# =========================

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        msg = request.form["message"]

        phishing, threat, words, explanation, patterns = analyze_message(msg)

        # Highlight suspicious words inside the message
        highlighted_msg = msg
        for w in words:
            highlighted_msg = highlighted_msg.replace(
                w,
                f"<mark>{w}</mark>"
            )

        result = {
            "prediction": "Phishing" if phishing else "Safe",
            "threat": threat,
            "words": ", ".join(words),
            "patterns": ", ".join(patterns),
            "highlighted": highlighted_msg,
            "explanation": explanation
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
