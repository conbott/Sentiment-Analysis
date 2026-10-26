"""
Utility functions for IMDb Sentiment Analysis:
- clean_text: regex-based normalization matching training format with slang/rating mapping
- is_valid_review: checks if text contains evaluatable words
- calculate_stars: probabilistic mapping from positive sentiment to 1-5 star ratings
"""

import re
from typing import Dict, Any


# Rating score shorthands
RATING_SHORTHANDS = [
    (r'\b10\s*/\s*10\b', 'masterpiece perfect'),
    (r'\b9\s*/\s*10\b', 'excellent wonderful'),
    (r'\b8\s*/\s*10\b', 'very good enjoyable'),
    (r'\b7\s*/\s*10\b', 'good decent'),
    (r'\b5\s*/\s*5\b', 'masterpiece excellent'),
    (r'\b4\s*/\s*5\b', 'great enjoyable'),
    (r'\b3\s*/\s*5\b', 'average okay'),
    (r'\b2\s*/\s*5\b', 'poor bad'),
    (r'\b1\s*/\s*5\b', 'terrible awful'),
    (r'\b0\s*/\s*5\b', 'horrible disaster'),
    (r'\b0\s*/\s*10\b', 'horrible disaster'),
    (r'\b1\s*/\s*10\b', 'worst terrible'),
    (r'\b2\s*/\s*10\b', 'awful terrible'),
    (r'\b3\s*/\s*10\b', 'bad boring'),
]

# Slang and vulgarity mapping to in-vocabulary semantic synonyms
SLANG_MAP = {
    r'\bdogshit\b': 'terrible crap',
    r'\bbullshit\b': 'garbage crap',
    r'\bhorseshit\b': 'terrible crap',
    r'\bbatshit\b': 'crazy terrible',
    r'\bdipshit\b': 'idiot',
    r'\bpiece of shit\b': 'terrible garbage',
    r'\bshitty\b': 'terrible',
    r'\bshittiest\b': 'worst',
    r'\bshit\b': 'crap',
    r'\bfuck\b': 'terrible',
    r'\bfucking\b': 'extremely',
    r'\bfucked\b': 'ruined terrible',
    r'\bsux\b': 'sucks',
    r'\bsuck\b': 'sucks',
    r'\bcrappy\b': 'crap',
    r'\btrashy\b': 'trash',
    r'\bdumpster fire\b': 'disaster disaster',
    r'\bpoop\b': 'crap',
}

CONTRACTIONS = {
    r"won\'t": "will not", r"can\'t": "cannot", r"n\'t": " not",
    r"\'re": " are", r"\'s": " is", r"\'d": " would",
    r"\'ll": " will", r"\'t": " not", r"\'ve": " have", r"\'m": " am"
}


def clean_text(text: str) -> str:
    """
    Preprocesses input text for sentiment analysis:
    1. Strips HTML tags and URLs
    2. Maps numeric ratings (e.g. '10/10', '1/10') to semantic sentiment tokens
    3. Maps vulgarity / internet slang to in-vocabulary equivalents (e.g. 'dogshit' -> 'terrible crap')
    4. Expands negation contractions (e.g. "don't" -> "do not")
    5. Normalizes to lowercase letters and single spaces
    """
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 1. Rating shorthands (preserve rating meaning before numbers are stripped)
    for pattern, replacement in RATING_SHORTHANDS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 2. Slang & vulgarity normalization
    for pattern, replacement in SLANG_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 3. Contraction expansion (preserves negations like "not")
    for contraction, replacement in CONTRACTIONS.items():
        text = re.sub(contraction, replacement, text, flags=re.IGNORECASE)

    # 4. Lowercase and keep letters only
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def is_valid_review(text: str) -> bool:
    """Returns True if the text contains meaningful words after cleaning."""
    cleaned = clean_text(text)
    return bool(cleaned and len(cleaned.split()) > 0)


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
