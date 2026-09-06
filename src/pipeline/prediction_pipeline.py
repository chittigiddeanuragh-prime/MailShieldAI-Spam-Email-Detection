import os
import pickle
import pandas as pd
from typing import Dict, Any
from src.utils.text_cleaner import preprocess_email_text, extract_threat_signals
from src.utils.mbox_parser import parse_mbox_file
from src.pipeline.train_pipeline import TrainPipeline

class PredictionPipeline:
    def __init__(self, model_dir: str = "models", load_models: bool = True):
        self.model_dir = model_dir
        self.vectorizer_path = os.path.join(self.model_dir, "vectorizer.pkl")
        self.model_path = os.path.join(self.model_dir, "model.pkl")
        self.vectorizer = None
        self.model = None
        
        if load_models:
            self._load_or_train_models()

    def _load_or_train_models(self):
        """Loads vectorizer and model. If missing, runs auto-training."""
        if not os.path.exists(self.vectorizer_path) or not os.path.exists(self.model_path):
            print("Model files not found. Auto-triggering initial training pipeline...")
            trainer = TrainPipeline(model_dir=self.model_dir)
            self.model, self.vectorizer = trainer.run_training()
        else:
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

    def calculate_threat_score(self, spam_prob: float, signals: dict) -> int:
        """
        Calculates an integrated 0-100 threat score combining ML spam probability 
        and heuristic threat signals (URL density, suspicious keywords, formatting).
        """
        base_score = spam_prob * 70.0  # ML model contributes up to 70 points
        
        # Heuristic additions
        keyword_score = min(signals["suspicious_keyword_count"] * 8, 20)
        url_score = min(signals["url_count"] * 5, 10)
        
        total_score = int(round(base_score + keyword_score + url_score))
        return min(max(total_score, 0), 100)

    def get_risk_level(self, threat_score: int) -> str:
        if threat_score < 25:
            return "Low (Clean)"
        elif threat_score < 50:
            return "Medium (Suspicious)"
        elif threat_score < 75:
            return "High (Threat)"
        else:
            return "Critical (Malicious)"

    def predict_single_email(self, email_text: str) -> Dict[str, Any]:
        """
        Classifies a single email text input and extracts security signals.
        """
        if not email_text or not email_text.strip():
            return {
                "prediction": "Ham",
                "confidence": 0.0,
                "threat_score": 0,
                "risk_level": "Low (Clean)",
                "signals": extract_threat_signals("")
            }

        # Preprocess text and extract threat signals
        cleaned_text = preprocess_email_text(email_text)
        signals = extract_threat_signals(email_text)
        
        # Vectorize and predict
        features = self.vectorizer.transform([cleaned_text])
        probabilities = self.model.predict_proba(features)[0]
        
        # Class 1 = Spam, Class 0 = Ham
        spam_prob = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        prediction_label = "Spam" if spam_prob >= 0.50 else "Ham"
        
        confidence = spam_prob * 100.0 if prediction_label == "Spam" else (1.0 - spam_prob) * 100.0
        threat_score = self.calculate_threat_score(spam_prob, signals)
        risk_level = self.get_risk_level(threat_score)
        
        return {
            "prediction": prediction_label,
            "confidence": round(confidence, 1),
            "threat_score": threat_score,
            "risk_level": risk_level,
            "spam_probability": round(spam_prob, 3),
            "signals": signals
        }

    def predict_mbox_file(self, file_path: str) -> pd.DataFrame:
        """
        Parses an MBOX file and batch-classifies all emails within it.
        """
        emails = parse_mbox_file(file_path)
        if not emails:
            return pd.DataFrame(columns=["Time", "Subject", "Sender", "Message-ID", "Prediction", "Confidence", "Threat Score", "Risk Level"])
            
        results = []
        for item in emails:
            combined_text = f"{item['Subject']} {item['Body']}"
            res = self.predict_single_email(combined_text)
            
            results.append({
                "Time": item["Time"],
                "Subject": item["Subject"],
                "Sender": item["Sender"],
                "Message-ID": item["Message-ID"],
                "Prediction": res["prediction"],
                "Confidence": f"{res['confidence']:.1f}%",
                "Threat Score": res["threat_score"],
                "Risk Level": res["risk_level"]
            })
            
        return pd.DataFrame(results)
