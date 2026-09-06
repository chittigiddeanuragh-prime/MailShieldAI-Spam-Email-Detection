import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from src.utils.text_cleaner import preprocess_email_text

# Set stdout UTF-8 encoding for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SAMPLE_DATASET = [
    # SPAM & PHISHING EXAMPLES (Label 1)
    ("URGENT: Your bank account has been suspended! Verify your details immediately at http://bank-secure-update.com/login", 1),
    ("Congratulations! You have won $1,000,000 in the International Email Lottery! Click here to claim your cash prize now!", 1),
    ("Dear friend, I need your assistance to transfer $15.5 Million out of Nigeria. You will receive 30% commission.", 1),
    ("FINAL WARNING: Your Netflix subscription will be canceled today. Update your credit card details now: http://netflix-billing-fix.com", 1),
    ("Get cheap pharmacy drugs online! No prescription required! Buy Viagra, Cialis, Xanax with 80% discount!", 1),
    ("Act fast! Limited time offer! Guaranteed fast cash loan up to $50,000 with 0% interest! Apply online today!", 1),
    ("SECURITY ALERT: Unusual login activity detected on your PayPal account. Click link to secure account: http://paypal-security-check.net", 1),
    ("Earn $5000 per week working from home! No experience required! Sign up for fast work online today!", 1),
    ("Your Apple ID has been locked for security reasons. Please click here http://apple-id-verify.org to restore access.", 1),
    ("Exclusive Crypto Opportunity: Double your Bitcoin in 24 hours! Guaranteed returns on crypto investment platform!", 1),
    ("Unclaimed funds waiting for you! Social Security rebate payment of $3,200 pending. Fill form now to get paid.", 1),
    ("Hot deals on designer watches! Rolex 90% off! Limited stock available, buy now before sale ends!", 1),
    ("Notice of Court Summons: You are required to appear in court. Download attached case file document immediately.", 1),
    ("Your package delivery failed. Pay $2.99 re-delivery fee at http://postal-package-redelivery.org", 1),
    ("Urgent Password Reset Required for your Corporate Network Account. Click here to maintain access.", 1),

    # HAM / LEGITIMATE EXAMPLES (Label 0)
    ("Hi Team, Attached is the quarterly project status report for your review. Let's discuss during tomorrow's sync.", 0),
    ("Dear Anuragh, Thanks for scheduling the meeting. I have updated the slide deck and added the requested metrics.", 0),
    ("Your order #98234 has been shipped! Track your package delivery status with FedEx tracking code 123987456.", 0),
    ("Hi John, Here are the notes from our team standup today. Please review the pending task action items.", 0),
    ("Monthly Account Statement for August 2026 is now available in your online banking portal.", 0),
    ("Hello, Please find attached the invoice for consulting services rendered last month. Thank you for your business.", 0),
    ("Reminder: Team lunch tomorrow at 12:30 PM at the main cafeteria. Hope to see everyone there!", 0),
    ("Git Commit Notification: Pull request #45 merged into main branch successfully by devops pipeline.", 0),
    ("Flight Confirmation: Your flight to San Francisco (SFO) is confirmed for September 15. Seat 12A.", 0),
    ("Hi Sarah, Could you please review the latest code changes in repository when you get a chance? Thanks!", 0),
    ("University Seminar Invitation: Keynote lecture on Artificial Intelligence and Security scheduled for Thursday.", 0),
    ("Your subscription renewal receipt from GitHub Enterprise Services for $15.00 USD.", 0),
    ("Meeting Notes: Sprint planning discussion for upcoming product release version 2.4.", 0),
    ("Dear Customer, Your utility bill payment was received successfully. Thank you for using auto-pay.", 0),
    ("Hi Team, Please complete the annual compliance survey by end of day Friday. Link is in the intranet portal.", 0)
]

class TrainPipeline:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.vectorizer_path = os.path.join(self.model_dir, "vectorizer.pkl")
        self.model_path = os.path.join(self.model_dir, "model.pkl")

    def run_training(self) -> dict:
        """
        Trains and evaluates Logistic Regression, SVM, Random Forest, and Complement Naive Bayes models.
        Selects the best performing model based on F1-score and saves vectorizer + model artifacts.
        """
        df = pd.DataFrame(SAMPLE_DATASET, columns=["text", "label"])
        df["cleaned_text"] = df["text"].apply(preprocess_email_text)
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
        X = vectorizer.fit_transform(df["cleaned_text"])
        y = df["label"]
        
        models = {
            "Logistic Regression": CalibratedClassifierCV(LogisticRegression(C=1.0), cv=3),
            "SVM (Linear)": CalibratedClassifierCV(SVC(kernel='linear', probability=True), cv=3),
            "Random Forest": CalibratedClassifierCV(RandomForestClassifier(n_estimators=50, random_state=42), cv=3),
            "Complement Naive Bayes": CalibratedClassifierCV(ComplementNB(), cv=3)
        }
        
        evaluation_results = {}
        best_f1 = -1.0
        best_model = None
        best_name = ""
        
        print("\n--- Model Evaluation & Comparison ---")
        for name, clf in models.items():
            clf.fit(X, y)
            preds = clf.predict(X)
            
            p = precision_score(y, preds, zero_division=0)
            r = recall_score(y, preds, zero_division=0)
            f1 = f1_score(y, preds, zero_division=0)
            cm = confusion_matrix(y, preds)
            
            evaluation_results[name] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1_score": round(f1, 4),
                "confusion_matrix": cm.tolist()
            }
            
            print(f"[{name}] Precision: {p:.4f} | Recall: {r:.4f} | F1-Score: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = clf
                best_name = name
                
        print(f"\n[Selected Best Model]: {best_name} (F1-Score: {best_f1:.4f})")
        
        # Save best model artifacts
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
            
        with open(self.model_path, "wb") as f:
            pickle.dump(best_model, f)
            
        print(f"[Success] Saved model artifacts to '{self.model_dir}'\n")
        return evaluation_results
