import re
from bs4 import BeautifulSoup

# Suspicious keywords indicative of spam, phishing, or email threats
SUSPICIOUS_KEYWORDS = [
    "urgent", "winner", "congratulations", "lottery", "claim", "verify account",
    "password reset", "bank account", "wire transfer", "credit card", "social security",
    "100% free", "risk free", "limited time", "act now", "click here", "unclaimed money",
    "inheritance", "bitcoin", "crypto", "nigerian prince", "suspended", "security breach",
    "unusual activity", "billing problem", "confidential", "fast cash", "guaranteed"
]

def clean_html(text: str) -> str:
    """Strip HTML markup from string using BeautifulSoup."""
    if not text:
        return ""
    try:
        soup = BeautifulSoup(text, "html.parser")
        cleaned = soup.get_text(separator=" ")
        return cleaned
    except Exception:
        # Fallback regex HTML tag removal
        return re.sub(r'<[^>]+>', ' ', text)

def preprocess_email_text(raw_text: str) -> str:
    """
    Cleans raw email content for ML processing:
    1. Strip HTML tags
    2. Lowercase text
    3. Normalize whitespace and clean special formatting
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""
        
    text = clean_html(raw_text)
    # Remove URLs for text normalization (or normalize them)
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' url_link ', text)
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email_address ', text)
    # Normalize extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def extract_threat_signals(raw_text: str) -> dict:
    """
    Extracts security and threat signals from raw email text for risk scoring.
    """
    if not raw_text:
        return {
            "url_count": 0,
            "uppercase_ratio": 0.0,
            "suspicious_keyword_count": 0,
            "detected_keywords": [],
            "has_html": False
        }
        
    # Check for HTML
    has_html = bool(re.search(r'<[a-z][\s\S]*>', raw_text, re.IGNORECASE))
    
    # URL extraction count
    urls = re.findall(r'http[s]?://\S+|www\.\S+', raw_text)
    url_count = len(urls)
    
    # Uppercase character ratio
    letters = [ch for ch in raw_text if ch.isalpha()]
    uppercase_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters) if letters else 0.0
    
    # Suspicious keywords matching
    lowered = raw_text.lower()
    detected_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in lowered]
    
    return {
        "url_count": url_count,
        "uppercase_ratio": round(uppercase_ratio, 3),
        "suspicious_keyword_count": len(detected_keywords),
        "detected_keywords": detected_keywords,
        "has_html": has_html
    }
