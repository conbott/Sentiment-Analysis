"""
FastAPI Backend for IMDb Sentiment Analysis
Provides REST API endpoints for real-time and batch sentiment predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
from contextlib import asynccontextmanager
import joblib
import os
import re

# Resolve Model Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_pipeline.joblib")
pipeline = None


def clean_text(text: str) -> str:
    """Preprocesses input text to match training pipeline format."""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    contractions = {
        r"won\'t": "will not", r"can\'t": "cannot", r"n\'t": " not",
        r"\'re": " are", r"\'s": " is", r"\'d": " would",
        r"\'ll": " will", r"\'t": " not", r"\'ve": " have", r"\'m": " am"
    }
    for contraction, replacement in contractions.items():
        text = re.sub(contraction, replacement, text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model pipeline on startup and cleans up on shutdown."""
    global pipeline
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Train and save the model first.")
    pipeline = joblib.load(MODEL_PATH)
    print(f"[OK] Sentiment pipeline successfully loaded from: {MODEL_PATH}")
    yield


# Initialize FastAPI App with modern Lifespan handler
app = FastAPI(
    title="Movie Review Sentiment Analysis API",
    description="Production-ready NLP Sentiment Analysis REST API built with Scikit-Learn TF-IDF, Logistic Regression, and FastAPI.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Data Schemas ---
class ReviewRequest(BaseModel):
    review: str = Field(
        ...,
        min_length=1,
        example="The cinematography was breathtaking and the performances were deeply moving!"
    )

class BatchReviewRequest(BaseModel):
    reviews: List[str] = Field(
        ...,
        min_items=1,
        example=[
            "An absolute masterpiece of modern cinema!",
            "Horrible pacing, terrible acting, and a complete waste of time."
        ]
    )

class SentimentResponse(BaseModel):
    original_review: str
    cleaned_review: str
    sentiment: str
    label: int
    confidence_percentage: float
    probabilities: Dict[str, float]

class BatchSentimentResponse(BaseModel):
    total_reviews: int
    positive_count: int
    negative_count: int
    predictions: List[SentimentResponse]


# --- API Endpoints ---
@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to the IMDb Sentiment Analysis API!",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "healthy" if pipeline is not None else "unhealthy",
        "model_loaded": pipeline is not None,
        "model_file": os.path.basename(MODEL_PATH)
    }


@app.post("/predict", response_model=SentimentResponse, tags=["Inference"])
def predict_sentiment(request: ReviewRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    raw_text = request.review.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    
    cleaned = clean_text(raw_text)
    pred_label = int(pipeline.predict([cleaned])[0])
    probs = pipeline.predict_proba([cleaned])[0]
    
    prob_neg = float(probs[0])
    prob_pos = float(probs[1])
    
    sentiment = "Positive" if pred_label == 1 else "Negative"
    confidence = (prob_pos if pred_label == 1 else prob_neg) * 100
    
    return SentimentResponse(
        original_review=raw_text,
        cleaned_review=cleaned,
        sentiment=sentiment,
        label=pred_label,
        confidence_percentage=round(confidence, 2),
        probabilities={
            "positive": round(prob_pos, 4),
            "negative": round(prob_neg, 4)
        }
    )


@app.post("/predict-batch", response_model=BatchSentimentResponse, tags=["Inference"])
def predict_batch_sentiment(request: BatchReviewRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    predictions = []
    pos_count = 0
    neg_count = 0
    
    for raw_text in request.reviews:
        cleaned = clean_text(raw_text)
        pred_label = int(pipeline.predict([cleaned])[0])
        probs = pipeline.predict_proba([cleaned])[0]
        prob_neg = float(probs[0])
        prob_pos = float(probs[1])
        sentiment = "Positive" if pred_label == 1 else "Negative"
        confidence = (prob_pos if pred_label == 1 else prob_neg) * 100
        
        if pred_label == 1:
            pos_count += 1
        else:
            neg_count += 1
            
        predictions.append(SentimentResponse(
            original_review=raw_text,
            cleaned_review=cleaned,
            sentiment=sentiment,
            label=pred_label,
            confidence_percentage=round(confidence, 2),
            probabilities={
                "positive": round(prob_pos, 4),
                "negative": round(prob_neg, 4)
            }
        ))
        
    return BatchSentimentResponse(
        total_reviews=len(predictions),
        positive_count=pos_count,
        negative_count=neg_count,
        predictions=predictions
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
