import sys
import os

# Ensure root path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.prediction_pipeline import PredictionPipeline
from fastapi.testclient import TestClient
from api import app

def test_pipeline_single_email():
    print("[Testing] Single Email Prediction...")
    pipeline = PredictionPipeline(load_models=True)
    
    # Test Spam Input
    spam_sample = "URGENT: Your bank account has been suspended! Click http://secure-update-bank.com/login"
    res_spam = pipeline.predict_single_email(spam_sample)
    assert res_spam['prediction'] == "Spam", f"Expected Spam, got {res_spam['prediction']}"
    assert res_spam['threat_score'] > 50, f"Expected high threat score, got {res_spam['threat_score']}"
    print(f"  ✓ Spam sample classified correctly: {res_spam['prediction']} (Threat Score: {res_spam['threat_score']})")
    
    # Test Ham Input
    ham_sample = "Hi Team, Attached is the weekly project report. Let's sync tomorrow."
    res_ham = pipeline.predict_single_email(ham_sample)
    assert res_ham['prediction'] == "Ham", f"Expected Ham, got {res_ham['prediction']}"
    assert res_ham['threat_score'] < 30, f"Expected low threat score, got {res_ham['threat_score']}"
    print(f"  ✓ Ham sample classified correctly: {res_ham['prediction']} (Threat Score: {res_ham['threat_score']})")

def test_pipeline_mbox_batch():
    print("[Testing] Batch MBOX Processing...")
    pipeline = PredictionPipeline(load_models=True)
    mbox_path = os.path.join("sample_data", "sample_emails.mbox")
    
    df = pipeline.predict_mbox_file(mbox_path)
    assert len(df) == 4, f"Expected 4 emails, parsed {len(df)}"
    print(f"  ✓ Successfully parsed and predicted {len(df)} emails from sample MBOX file.")

def test_fastapi_endpoints():
    print("[Testing] FastAPI Endpoints...")
    client = TestClient(app)
    
    # Health Endpoint
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"
    print("  ✓ /health endpoint responded with 200 OK")
    
    # Predict Endpoint
    resp_pred = client.post("/api/v1/predict", json={
        "email_text": "URGENT: Verify your account now at http://phishing-site.com"
    })
    assert resp_pred.status_code == 200
    json_data = resp_pred.json()
    assert json_data["prediction"] == "Spam"
    print("  ✓ /api/v1/predict endpoint returned correct prediction response")

if __name__ == "__main__":
    print("\n--- Running MailShield AI Automated Suite ---")
    test_pipeline_single_email()
    test_pipeline_mbox_batch()
    test_fastapi_endpoints()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!\n")
