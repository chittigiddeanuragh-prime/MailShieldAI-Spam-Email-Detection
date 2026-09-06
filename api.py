import os
import sys
import tempfile
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Ensure root path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="🛡️ MailShield AI REST API",
    description="Intelligent Email Threat & Spam Detection API Service",
    version="1.0.0"
)

# Initialize Prediction Pipeline
pipeline = PredictionPipeline(load_models=True)

class SingleEmailRequest(BaseModel):
    email_text: str = Field(..., example="URGENT: Your bank account has been suspended! Verify at http://bank-secure-update.com")

class ThreatSignalDetail(BaseModel):
    url_count: int
    uppercase_ratio: float
    suspicious_keyword_count: int
    detected_keywords: List[str]
    has_html: bool

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    threat_score: int
    risk_level: str
    spam_probability: float
    signals: ThreatSignalDetail

@app.get("/health")
def health_check():
    """Returns API service health status."""
    return {"status": "healthy", "service": "MailShield AI", "version": "1.0.0"}

@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_email(request: SingleEmailRequest):
    """
    Classifies a single email text input and provides threat scoring + security signals.
    """
    if not request.email_text or not request.email_text.strip():
        raise HTTPException(status_code=400, detail="email_text field cannot be empty.")
    
    try:
        result = pipeline.predict_single_email(request.email_text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/v1/predict-mbox")
def predict_mbox(file: UploadFile = File(...)):
    """
    Upload an MBOX file to batch predict all emails contained in the file.
    """
    if not file.filename.endswith(('.mbox', '.txt')):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .mbox or .txt file.")
        
    try:
        content = file.file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mbox') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
            
        try:
            df = pipeline.predict_mbox_file(tmp_path)
            return {
                "filename": file.filename,
                "total_emails": len(df),
                "spam_count": int(len(df[df['Prediction'] == 'Spam'])),
                "results": df.to_dict(orient="records")
            }
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
