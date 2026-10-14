# End-to-End NLP Sentiment Analysis System (50,000 IMDb Reviews)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An industrial-grade, end-to-end Natural Language Processing (NLP) binary classification pipeline built on the benchmark IMDb 50,000 movie reviews dataset. The project features exploratory data analysis (EDA), specialized text cleaning, TF-IDF n-gram feature engineering, multi-model benchmarking, explainability analysis, a FastAPI REST backend, and an interactive Streamlit web user interface.

---

## Project Overview

- **Domain and Task:** Natural Language Processing (NLP) - Binary Sentiment Classification
- **Dataset Size:** 50,000 rows (25,000 Positive, 25,000 Negative)
- **Data Quality:** Removed 418 duplicates, resulting in 49,582 unique reviews
- **Test Accuracy:** 90.92%
- **ROC-AUC Score:** 0.968
- **Model Pipeline:** Sublinear TF-IDF (1, 2 n-grams, 25,000 features) + Logistic Regression (C=2.0)
- **Backend API:** FastAPI with automated Swagger documentation
- **Frontend Web UI:** Streamlit with real-time inference, sample presets, and CSV batch processing
- **Containerization:** Docker-ready configuration

---

## Model Comparison and Benchmarks

All models were evaluated on an independent 20% stratified hold-out test set (9,917 reviews) using identical TF-IDF feature matrices:

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Champion)** | **90.92%** | **90.01%** | **92.14%** | **91.06%** | **0.49 s** |
| **Linear Support Vector Classifier (LinearSVC)** | 90.39% | 90.12% | 90.88% | 90.50% | 1.27 s |
| **Multinomial Naive Bayes (Baseline)** | 87.99% | 87.45% | 88.78% | 88.11% | 0.03 s |

---

## Technical Highlights

1. **Negation Preservation:** Standard stopword filters remove words like "not", "never", and "no", inverting the meaning of phrases like "not good". This pipeline expands contractions (e.g., "don't" to "do not") and preserves emotional negation modifiers.
2. **Unigram and Bigram Context (1, 2):** Captures multi-word sentiment signals such as "must watch", "waste of time", and "masterfully acted".
3. **Sublinear Term Frequency:** Applies 1 + log(TF) scaling to dampen the dominance of unusually long reviews.
4. **Calibrated Confidence:** Logistic regression outputs well-calibrated class probabilities, allowing both the REST API and Streamlit UI to display exact prediction confidence percentages.

---

## Project Structure

```text
Sentiment_Analysis_Proj/
├── app/
│   ├── __init__.py
│   ├── api.py                     # FastAPI application with REST endpoints
│   └── ui.py                      # Streamlit web interface
├── dataset/
│   └── README.md                  # Instructions for downloading the dataset
├── models/
│   └── sentiment_pipeline.joblib  # Serialized TF-IDF + Logistic Regression pipeline
├── sentiment_analysis.ipynb       # Full notebook: EDA, Preprocessing, Benchmarking, Explainability
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Containerization instructions
├── PROJECT_DOCUMENTATION.md       # Comprehensive project report
└── README.md                      # Project documentation
```

---

## Dataset Acquisition

The dataset is excluded from git tracking to maintain a lightweight repository. To set up the dataset locally:

1. Download the **IMDb Dataset of 50K Movie Reviews** from Kaggle:  
   [https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Extract the archive and place `IMDB Dataset.csv` inside the `dataset/` directory:
   ```text
   dataset/IMDB Dataset.csv
   ```

---

## Quickstart Guide

### 1. Clone the Repository and Install Dependencies
```bash
git clone https://github.com/conbott/Sentiment-Analysis.git
cd Sentiment-Analysis
pip install -r requirements.txt
```

### 2. Run the FastAPI Backend
```bash
python app/api.py
```
* **API Root:** http://127.0.0.1:8000/
* **Interactive Swagger Documentation:** http://127.0.0.1:8000/docs

#### Sample API Request (curl):
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"review": "This movie was an absolute masterpiece with breathtaking visuals!"}'
```

#### Sample JSON Response:
```json
{
  "original_review": "This movie was an absolute masterpiece with breathtaking visuals!",
  "cleaned_review": "this movie was an absolute masterpiece with breathtaking visuals",
  "sentiment": "Positive",
  "label": 1,
  "confidence_percentage": 96.04,
  "probabilities": {
    "positive": 0.9604,
    "negative": 0.0396
  }
}
```

### 3. Run the Streamlit Web Application
```bash
python -m streamlit run app/ui.py
```
Open your browser at `http://localhost:8501` to test the interactive user interface.

---

## Docker Deployment (Optional)

Build and run the containerized application:
```bash
docker build -t sentiment-analysis-app .
docker run -p 8000:8000 -p 8501:8501 sentiment-analysis-app
```

---

## Author Information

- **Project:** NLP Sentiment Analysis on 50,000 Movie Reviews
- **Submission Date:** October 15, 2026
- **Presentation Date:** October 16, 2026
