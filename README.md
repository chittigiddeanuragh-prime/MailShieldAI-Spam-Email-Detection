# 🛡️ MailShield AI — Intelligent Email Threat & Spam Detection System

An end-to-end Machine Learning & Natural Language Processing system for detecting email threats, phishing attempts, and spam messages. Featuring single-email threat scoring, batch `.mbox` file scanning, a Streamlit web application, and a high-performance REST API powered by FastAPI.

---

## 🛠️ Tech Stack & Technologies

| Category | Technology / Library |
| :--- | :--- |
| **Programming Language** | 🐍 Python 3.10+ |
| **Machine Learning** | Scikit-learn (Logistic Regression, SVM, Random Forest, Naive Bayes) |
| **NLP & Vectorization** | TF-IDF Vectorizer, N-Gram Analysis, Text Preprocessing |
| **Data Processing** | Pandas, NumPy |
| **HTML Parsing** | BeautifulSoup4 |
| **Web UI** | Streamlit |
| **REST API** | FastAPI + Uvicorn + Pydantic |
| **Package Management** | `uv` / `pyproject.toml` & `requirements.txt` |
| **Model Evaluation** | Precision, Recall, F1-Score, Confusion Matrix |
| **Experimentation** | Jupyter Notebook (`notebook.ipynb`) |
| **Version Control** | Git + GitHub |

---

## 📁 Project Architecture

```
mailshield-ai/
├── app.py                      # Streamlit Web Dashboard UI
├── api.py                      # FastAPI REST service endpoints
├── train.py                    # Standalone model training execution script
├── notebook.ipynb              # Jupyter Notebook for EDA & model experimentation
├── requirements.txt            # Dependency configuration file
├── pyproject.toml              # uv / modern build setup
├── README.md                   # Documentation
├── .gitignore                  # Git ignore rules
├── sample_data/
│   └── sample_emails.mbox      # Sample MBOX archive for testing
├── models/
│   ├── vectorizer.pkl          # Serialized TF-IDF Vectorizer
│   └── model.pkl               # Serialized ML Classifier
└── src/
    ├── __init__.py
    ├── utils/
    │   ├── text_cleaner.py     # HTML cleaning, regex extraction & threat signals
    │   └── mbox_parser.py     # Email header & body parser for MBOX archives
    └── pipeline/
        ├── train_pipeline.py  # Model trainer & benchmark evaluator
        └── prediction_pipeline.py # Core prediction engine & threat scoring
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Clone or enter your project directory and set up a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(Alternatively, if using `uv`: `uv sync`)*

---

### 2. Model Training & Evaluation

Train the Machine Learning classifiers and save binary model artifacts:

```bash
python train.py
```

This evaluates Logistic Regression, SVM, Random Forest, and Naive Bayes models, prints **Precision, Recall, F1-Score**, and saves the optimal model artifacts to `models/`.

---

### 3. Launch Streamlit Web Dashboard

Start the interactive Web Interface:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to test:
- **Single Email Threat Scanner**: Paste email content to view Verdict (Spam/Ham), Confidence Score, 0-100 Threat Score, and Security Signals.
- **Batch MBOX Processing**: Upload `.mbox` or `.txt` files to process bulk email dumps and download tabular CSV results.
- **Model Analytics**: View tech specs and benchmark precision/recall metrics.

---

### 4. Launch FastAPI REST API Service

Run the REST API server:

```bash
python api.py
```

Or using Uvicorn:

```bash
uvicorn api:app --reload --port 8000
```

Interactive API documentation will be available at `http://localhost:8000/docs`.

#### API Endpoints:
- `GET /health` : API Service Health status.
- `POST /api/v1/predict` : Predict threat score and spam label for single text input.
- `POST /api/v1/predict-mbox` : Upload `.mbox` file for bulk processing.

---

## 📊 Notebook & Experimentation

Launch `notebook.ipynb` in Jupyter Notebook or VS Code to run interactive exploratory data analysis, evaluate confusion matrices, and benchmark new feature engineering ideas.
