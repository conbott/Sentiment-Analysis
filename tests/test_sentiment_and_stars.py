"""
Unit and integration tests for IMDb Sentiment Analysis and Star Rating Engine.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from app.utils import clean_text, calculate_stars
from fastapi.testclient import TestClient
from app.api import app, lifespan


def test_clean_text_basic():
    raw = "<b>Great movie!</b> Check http://example.com for more."
    cleaned = clean_text(raw)
    assert "<" not in cleaned
    assert "http" not in cleaned
    assert "great movie" in cleaned


def test_clean_text_negations_preserved():
    raw = "The film didn't deliver, wasn't good, and won't be remembered."
    cleaned = clean_text(raw)
    # Negations should be expanded, preserving 'not'
    assert "not" in cleaned
    assert "will not" in cleaned


def test_calculate_stars_edge_cases():
    # 0.0 prob
    res0 = calculate_stars(0.0)
    assert res0["stars"] == 1
    assert res0["score"] == 1.0
    assert res0["label"] == "Very Poor"
    assert res0["star_icons"] == "★☆☆☆☆"

    # 1.0 prob
    res1 = calculate_stars(1.0)
    assert res1["stars"] == 5
    assert res1["score"] == 5.0
    assert res1["label"] == "Excellent"
    assert res1["star_icons"] == "★★★★★"

    # 0.5 neutral / mixed prob
    res_half = calculate_stars(0.5)
    assert res_half["stars"] == 3
    assert res_half["score"] == 3.0
    assert res_half["label"] == "Average / Mixed"
    assert res_half["star_icons"] == "★★★☆☆"

    # Clamping tests for values out of [0, 1]
    res_neg = calculate_stars(-0.5)
    assert res_neg["stars"] == 1
    assert res_neg["score"] == 1.0

    res_overflow = calculate_stars(1.5)
    assert res_overflow["stars"] == 5
    assert res_overflow["score"] == 5.0


def test_calculate_stars_monotonicity():
    probs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.95, 1.0]
    scores = [calculate_stars(p)["score"] for p in probs]
    stars = [calculate_stars(p)["stars"] for p in probs]

    # Scores must be non-decreasing
    assert scores == sorted(scores)
    # Stars must be non-decreasing
    assert stars == sorted(stars)


def test_fastapi_endpoints():
    with TestClient(app) as client:
        # Test health
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

        # Test single predict
        resp = client.post("/predict", json={
            "review": "An extraordinary, brilliant, and breathtaking film!"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["sentiment"] == "Positive"
        assert data["label"] == 1
        assert "stars" in data
        assert "score_out_of_5" in data
        assert "star_rating" in data
        assert "rating_label" in data
        assert data["stars"] in [4, 5]
        assert data["score_out_of_5"] >= 3.5

        # Test negative review
        resp_neg = client.post("/predict", json={
            "review": "A total waste of time, terrible acting, horrible script."
        })
        assert resp_neg.status_code == 200
        data_neg = resp_neg.json()
        assert data_neg["sentiment"] == "Negative"
        assert data_neg["label"] == 0
        assert data_neg["stars"] in [1, 2]
        assert data_neg["score_out_of_5"] <= 2.5

        # Test batch predict
        resp_batch = client.post("/predict-batch", json={
            "reviews": [
                "Loved every single second of this masterpiece!",
                "Unwatchable and boring garbage."
            ]
        })
        assert resp_batch.status_code == 200
        data_batch = resp_batch.json()
        assert data_batch["total_reviews"] == 2
        assert data_batch["positive_count"] == 1
        assert data_batch["negative_count"] == 1
        assert "average_stars" in data_batch
        assert 1.0 <= data_batch["average_stars"] <= 5.0
        assert len(data_batch["predictions"]) == 2
