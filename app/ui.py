"""
Streamlit Web UI for IMDb Sentiment Analysis
Provides an interactive, modern web application for real-time and batch sentiment predictions,
featuring an automated 1-5 Star Rating conversion system.
"""

import sys
import os
import re
import pandas as pd
import streamlit as st
import joblib

# Resolve Model and Project Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from app.utils import clean_text, calculate_stars, is_valid_review
except ModuleNotFoundError:
    from utils import clean_text, calculate_stars, is_valid_review

MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_pipeline.joblib")

# Page Configuration
st.set_page_config(
    page_title="Movie Sentiment & Star Rating AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .positive-card {
        background-color: #ECFDF5;
        border-radius: 10px;
        padding: 1.2rem;
        border: 2px solid #10B981;
        color: #065F46;
    }
    .negative-card {
        background-color: #FEF2F2;
        border-radius: 10px;
        padding: 1.2rem;
        border: 2px solid #EF4444;
        color: #991B1B;
    }
    .rating-card {
        background-color: #FFFBEB;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 2px solid #FCD34D;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Caches and loads the serialized pipeline."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please train and save the model first.")
        return None
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()


# --- Sidebar ---
with st.sidebar:
    st.header("Project Overview")
    st.markdown("""
    **NLP Sentiment Analysis & Star Rating AI**
    - **Dataset:** IMDb 50,000 Movie Reviews
    - **Model:** TF-IDF (1, 2) + Logistic Regression
    - **Accuracy:** **90.92%**
    - **ROC-AUC:** **0.968**
    - **Star Rating:** Probabilistic 1–5 Star Rating Engine
    - **Backend:** FastAPI REST API
    - **Frontend:** Streamlit
    """)
    st.divider()
    st.subheader("Interactive Options")
    min_confidence = st.slider("Confidence Warning Threshold (%)", 50, 95, 60, step=5)
    st.info("Reviews with confidence below this threshold will display an uncertainty note.")


# --- Main Header ---
st.markdown("<div class='main-title'>Movie Review Sentiment & Star Rating Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>An End-to-End NLP Deep Dive trained on 50,000 IMDb movie reviews with probabilistic 1–5 star rating estimation.</div>", unsafe_allow_html=True)


# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Single Review Prediction", "Batch Analysis (CSV)", "Model Architecture & Insights"])


# --- TAB 1: Single Prediction ---
with tab1:
    col1, col2 = st.columns([1.6, 1])
    
    with col1:
        st.subheader("Enter a Movie Review")
        
        # Quick Presets
        preset_cols = st.columns(3)
        sample_positive = "An absolute cinematic masterpiece! The directing was flawless and the acting was deeply poignant."
        sample_negative = "A complete disaster of a film. The plot was full of holes, the pacing was agonizingly slow, and I walked out."
        sample_tricky = "The movie was not good at all, despite having a stellar cast and incredible budget."
        
        if preset_cols[0].button("Sample Positive"):
            st.session_state["review_input"] = sample_positive
        if preset_cols[1].button("Sample Negative"):
            st.session_state["review_input"] = sample_negative
        if preset_cols[2].button("Sample Tricky (Negation)"):
            st.session_state["review_input"] = sample_tricky

        review_input = st.text_area(
            "Write or paste a review below:",
            value=st.session_state.get("review_input", sample_positive),
            height=160
        )
        
        analyze_clicked = st.button("Analyze Sentiment", type="primary", use_container_width=True)

    with col2:
        st.subheader("Prediction Result")
        if analyze_clicked and pipeline is not None:
            raw_text = review_input.strip()
            if not raw_text:
                st.warning("Please type or select a review to analyze.")
            else:
                cleaned = clean_text(raw_text)
                if not cleaned or len(cleaned.split()) == 0:
                    st.warning("⚠️ **No valid words or sentiment detected.** Numbers, symbols, and punctuation alone do not convey sentiment. Please write a review containing actual words.")
                else:
                    pred_label = int(pipeline.predict([cleaned])[0])
                    probs = pipeline.predict_proba([cleaned])[0]
                    prob_neg = float(probs[0])
                    prob_pos = float(probs[1])
                    
                    is_pos = (pred_label == 1)
                    confidence = (prob_pos if is_pos else prob_neg) * 100
                    star_info = calculate_stars(prob_pos)
                    
                    # 1. Sentiment Banner Card
                    if is_pos:
                        st.markdown(f"""
                        <div class='positive-card'>
                            <h2 style='margin:0; color:#065F46;'>POSITIVE SENTIMENT</h2>
                            <p style='font-size:1.15rem; margin-top:0.4rem;'>Confidence: <b>{confidence:.2f}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='negative-card'>
                            <h2 style='margin:0; color:#991B1B;'>NEGATIVE SENTIMENT</h2>
                            <p style='font-size:1.15rem; margin-top:0.4rem;'>Confidence: <b>{confidence:.2f}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 2. Star Rating Card
                    filled_stars = "<span style='color: #F59E0B;'>★</span>" * star_info['stars']
                    empty_stars = "<span style='color: #D1D5DB;'>☆</span>" * (5 - star_info['stars'])
                    stars_html = filled_stars + empty_stars
                    
                    st.markdown(f"""
                    <div class='rating-card'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <div>
                                <div style='font-size:0.75rem; font-weight:700; color:#B45309; text-transform:uppercase; letter-spacing:0.05em;'>ESTIMATED STAR RATING</div>
                                <div style='margin-top:2px;'>
                                    <span style='font-size:1.8rem; letter-spacing:2px;'>{stars_html}</span>
                                    <span style='font-size:1.2rem; font-weight:700; color:#1F2937; margin-left:8px;'>{star_info['stars']} / 5 Stars</span>
                                </div>
                            </div>
                            <div style='text-align:right;'>
                                <div style='font-size:1.4rem; font-weight:800; color:#D97706;'>{star_info['score']:.1f}<span style='font-size:0.85rem; color:#6B7280; font-weight:500;'> / 5.0</span></div>
                                <div style='font-size:0.85rem; font-weight:600; color:#4B5563;'>{star_info['label']}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                        
                    st.write("**Probability Breakdown:**")
                    st.progress(prob_pos, text=f"Positive Probability: {prob_pos*100:.1f}%")
                    st.progress(prob_neg, text=f"Negative Probability: {prob_neg*100:.1f}%")
                    
                    if confidence < min_confidence:
                        st.warning(f"Confidence is below {min_confidence}%. This review has mixed or subtle emotional cues.")
                        
                    with st.expander("View Preprocessed NLP Tokens"):
                        st.code(cleaned)


# --- TAB 2: Batch Analysis ---
with tab2:
    st.subheader("Batch Review Analysis")
    st.write("Upload a CSV file containing movie reviews or enter multiple reviews to predict sentiment and star ratings all at once.")
    
    uploaded_file = st.file_uploader("Upload CSV (must contain a column named 'review')", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            if 'review' not in batch_df.columns:
                st.error("Uploaded CSV must have a 'review' column!")
            else:
                st.write(f"Loaded {len(batch_df)} rows. Processing predictions and star ratings...")
                
                cleaned_batch = batch_df['review'].astype(str).apply(clean_text)
                
                batch_preds = []
                batch_probs = []
                star_results = []
                
                for c in cleaned_batch:
                    if c and len(c.split()) > 0:
                        p_label = int(pipeline.predict([c])[0])
                        p_probs = pipeline.predict_proba([c])[0]
                        p_neg = float(p_probs[0])
                        p_pos = float(p_probs[1])
                        batch_preds.append("Positive" if p_label == 1 else "Negative")
                        batch_probs.append((p_neg, p_pos))
                        star_results.append(calculate_stars(p_pos))
                    else:
                        batch_preds.append("Unclear / No Words")
                        batch_probs.append((0.5, 0.5))
                        star_results.append({
                            "stars": 3,
                            "score": 3.0,
                            "label": "Neutral / Unclear",
                            "star_icons": "★★★☆☆"
                        })
                
                batch_df['Predicted_Sentiment'] = batch_preds
                batch_df['Positive_Prob'] = [round(pr[1], 4) for pr in batch_probs]
                batch_df['Negative_Prob'] = [round(pr[0], 4) for pr in batch_probs]
                
                batch_df['Estimated_Stars'] = [r['stars'] for r in star_results]
                batch_df['Star_Rating'] = [f"{r['star_icons']} ({r['stars']}/5)" for r in star_results]
                batch_df['Rating_Score'] = [r['score'] for r in star_results]
                batch_df['Rating_Label'] = [r['label'] for r in star_results]
                
                # Metrics Summary
                pos_total = (batch_df['Predicted_Sentiment'] == 'Positive').sum()
                neg_total = (batch_df['Predicted_Sentiment'] == 'Negative').sum()
                avg_stars = float(batch_df['Rating_Score'].mean())
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Reviews", len(batch_df))
                m2.metric("Positive Reviews", pos_total, f"{pos_total/len(batch_df)*100:.1f}%")
                m3.metric("Negative Reviews", neg_total, f"{neg_total/len(batch_df)*100:.1f}%")
                m4.metric("Average Rating", f"{avg_stars:.2f} / 5.0", f"{'★' * int(round(avg_stars))}")
                
                st.write("**Processed Results Preview:**")
                st.dataframe(batch_df[['review', 'Predicted_Sentiment', 'Star_Rating', 'Rating_Score', 'Positive_Prob', 'Negative_Prob']].head(10))
                
                csv_data = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Predictions as CSV",
                    csv_data,
                    "sentiment_predictions_with_stars.csv",
                    "text/csv",
                    key='download-csv'
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")


# --- TAB 3: Model Architecture & Insights ---
with tab3:
    st.subheader("Pipeline Architecture & Key Insights")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Dataset Size", "50,000 Rows")
    c2.metric("Unique Clean Reviews", "49,582")
    c3.metric("Test Accuracy", "90.92%")
    c4.metric("ROC-AUC Score", "0.968")
    
    st.divider()
    st.markdown("""
    ### Why this NLP Architecture Works:
    1. **Contraction and Negation Preservation:** Phrases like `"not good"` and `"didn't like"` retain their emotional polarity instead of being stripped out by generic stopword filters.
    2. **Unigrams + Bigrams (1, 2):** Captures multi-word sentiments like `"must watch"`, `"waste of time"`, `"well directed"`.
    3. **Sublinear Term Frequency:** Prevents overly verbose reviews from artificially skewing term importance.
    4. **Logistic Regression with C=2.0:** Delivers optimal generalization without overfitting, yielding rapid sub-second inference speeds ideal for production deployments.

    ---

    ### Sentiment-to-Star Conversion Engine:
    While traditional binary classifiers output binary decisions (`Positive` vs. `Negative`), consumer review platforms (IMDb, Google Reviews, Amazon) communicate satisfaction via 1 to 5 star ratings.
    
    - **Continuous Rating Formulation:**
      $$\\text{Rating Score} = 1.0 + 4.0 \\times P(\\text{Positive})$$
      This bounds the continuous score precisely between **1.0 / 5.0** ($P=0$) and **5.0 / 5.0** ($P=1$), with **3.0 / 5.0** representing neutral balance ($P=0.5$).
    
    - **Probabilistic Calibration Thresholds:**
      - **5 Stars (Score $\\ge 4.5$, $P \\ge 87.5\\%$):** **Excellent / Masterpiece** &mdash; Overwhelmingly positive sentiment.
      - **4 Stars ($3.5 \\le \\text{Score} < 4.5$, $62.5\\% \\le P < 87.5\\%$):** **Good / Enjoyable** &mdash; Strongly favorable review.
      - **3 Stars ($2.5 \\le \\text{Score} < 3.5$, $37.5\\% \\le P < 62.5\\%$):** **Average / Mixed** &mdash; Balanced polarity with mixed or subtle cues.
      - **2 Stars ($1.5 \\le \\text{Score} < 2.5$, $12.5\\% \\le P < 37.5\\%$):** **Poor / Disappointing** &mdash; Noticeably negative reaction.
      - **1 Star (Score $< 1.5$, $P < 12.5\\%$):** **Very Poor / Terrible** &mdash; Severe negative criticism.
    """)
