"""
Utility functions for IMDb Sentiment Analysis:
- clean_text: regex-based normalization matching the training pipeline
- calculate_stars: probabilistic mapping from positive sentiment to 1-5 star ratings
"""

import re
from typing import Dict, Any


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


def calculate_stars(prob_pos: float) -> Dict[str, Any]:
    """
    Maps continuous positive sentiment probability [0.0, 1.0] to a 1-5 star rating.
    
    Formula:
        score = 1.0 + 4.0 * prob_pos (ranges 1.0 to 5.0)
        
    Thresholds:
        - Score >= 4.5 (prob >= 87.5%): 5 Stars (Excellent / Overwhelmingly Positive)
        - Score >= 3.5 (prob >= 62.5%): 4 Stars (Good / Strongly Positive)
        - Score >= 2.5 (prob >= 37.5%): 3 Stars (Average / Mixed / Neutral)
        - Score >= 1.5 (prob >= 12.5%): 2 Stars (Poor / Moderately Negative)
        - Score < 1.5  (prob < 12.5%):  1 Star  (Very Poor / Overwhelmingly Negative)
    """
    prob_pos = max(0.0, min(1.0, float(prob_pos)))
    score = round(1.0 + 4.0 * prob_pos, 1)

    if score >= 4.5:
        stars = 5
        label = "Excellent"
    elif score >= 3.5:
        stars = 4
        label = "Good"
    elif score >= 2.5:
        stars = 3
        label = "Average / Mixed"
    elif score >= 1.5:
        stars = 2
        label = "Poor"
    else:
        stars = 1
        label = "Very Poor"

    return {
        "stars": stars,
        "score": score,
        "label": label,
        "star_icons": "★" * stars + "☆" * (5 - stars),
    }
