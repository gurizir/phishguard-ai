indian_phishing = [
    ("spam", "Your SBI account will be blocked today. Verify immediately at http://sbi-secure-login.in"),
    ("spam", "Income Tax Department: Refund pending. Submit PAN details now."),
    ("spam", "UPI Alert: Rs.5000 debited. If not you, click here to reverse transaction."),
    ("spam", "Aadhaar suspended due to verification failure. Update KYC immediately."),
    ("spam", "Congratulations! You have won Rs.10,00,000 in KBC lottery. Claim now."),
    ("spam", "IRCTC account locked. Confirm details to restore access."),
    ("spam", "Electricity bill overdue. Pay now to avoid disconnection."),
    ("spam", "Dear customer, your SIM will be deactivated today. Verify instantly.")
]

indian_df = pd.DataFrame(indian_phishing, columns=["label", "text"])

df = pd.concat([df, indian_df], ignore_index=True)

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
