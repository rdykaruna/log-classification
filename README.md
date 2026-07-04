Hybrid Log Classification System

This repository provides a FastAPI-based web service that automatically classifies system log messages using a hybrid approach combining Regex rules, Machine Learning (BERT Embeddings + Scikit-Learn Classifier), and a Large Language Model (DeepSeek via Groq).

Features

Hybrid Routing Pipeline:

Routes LegacyCRM source logs directly to DeepSeek-R1 (via Groq API) for high-reasoning classification.

Routes other sources through fast Regex pattern matching.

Falls back to a local BERT-based Scikit-Learn classifier if regex matching fails.

FastAPI Endpoint: Accepts a .csv log file, processes it asynchronously, and returns the classified .csv file directly.

Robust Model Preloading: Embeddings and classifiers are cached globally at startup to ensure low-latency inference.

Setup & Installation

1. Clone the Repository
Bash
git clone https://github.com/rdykaruna/log-classification.git
cd log-classification
2. Create a Virtual Environment & Install Dependencies
Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install fastapi uvicorn pandas groq sentence-transformers scikit-learn joblib python-dotenv
3. Environment Configuration
Create a .env file in the root directory and add your Groq API key:

Code snippet

GROQ_API_KEY=your_actual_groq_api_key_here
4. Model Artifact Placement
Ensure you place your pre-trained traditional ML model artifact in a directory named models:

log-classification/
├── models/
│   └── log_classifier.joblib
├── classify.py
├── server.py
└── .env
Running the Application
Start the FastAPI local development server using uvicorn:

Bash
uvicorn server:app --reload
The service will be up and running at http://127.0.0.1:8000. You can access the interactive API documentation at http://127.0.0.1:8000/docs.

API Endpoint Usage
POST /classify/
Upload a CSV file containing the logs you want to categorize.

Input Requirements

The uploaded CSV must contain the following headers:

source: The origin name of the log (e.g., LegacyCRM, AuthServer).

log_message: The actual text payload of the log entry.
