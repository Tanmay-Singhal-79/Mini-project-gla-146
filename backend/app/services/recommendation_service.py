"""
recommendation_service.py
Loads the ML model once at module import time and provides
recommend_topics() and generate_learning_path() functions.
"""

import os
import pickle
import logging
from typing import List, Dict, Any

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ─── Path Configuration ──────────────────────────────────────────────────────
# Ensure we find the ML directory regardless of where the app is started from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # app/
ML_DIR = os.path.join(BASE_DIR, "ml")
VECTORIZER_PATH = os.path.join(ML_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(ML_DIR, "model.pkl")

# ─── Lazy Loading Mechanism ─────────────────────────────────────────────────
_vectorizer = None
_model_data: Dict[str, Any] = {}
_is_loading = False

def _get_model():
    global _vectorizer, _model_data, _is_loading
    
    if _vectorizer is not None:
        return _vectorizer, _model_data
    
    if _is_loading:
        # Simple spin lock or wait logic could go here, 
        # but for now we'll just let the first caller load it
        pass

    try:
        _is_loading = True
        logger.info("[*] Initializing ML Model (Lazy Load)...")
        
        # Check if model files exist
        if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
            logger.info("[*] Model artifacts missing. Triggering automatic training...")
            try:
                from app.ml.train_model import train
                train()
            except Exception as train_error:
                logger.error(f"[✗] Automatic training failed: {train_error}")
                return None, {}

        with open(VECTORIZER_PATH, "rb") as f:
            _vectorizer = pickle.load(f)
        with open(MODEL_PATH, "rb") as f:
            _model_data = pickle.load(f)
        
        logger.info(f"[✓] ML model ready.")
        return _vectorizer, _model_data
    except Exception as e:
        logger.error(f"[✗] Failed to load ML model: {e}")
        return None, {}
    finally:
        _is_loading = False


# ─── Service Functions ────────────────────────────────────────────────────────

def _is_model_ready() -> bool:
    vec, data = _get_model()
    return vec is not None and "df" in data


def recommend_topics(interest: str, top_n: int = 5) -> List[Dict]:
    """
    Given a user interest string, find the top_n most similar topics
    from the dataset using TF-IDF cosine similarity.
    """
    vec, data = _get_model()
    if vec is None:
        return []

    df: pd.DataFrame = data["df"]
    unique_interests: List[str] = data["unique_interests"]
    tfidf_matrix = data["tfidf_matrix"]

    # Vectorize user query
    query_vec = vec.transform([interest.lower()])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get top matching interest labels
    top_indices = similarities.argsort()[::-1][:top_n]
    matched_interests = [unique_interests[i] for i in top_indices if similarities[i] > 0]

    if not matched_interests:
        # Fallback: just return first domain topics
        matched_interests = [unique_interests[0]]

    # Gather unique topics from matched interests
    matched_df = df[df["interest"].isin(matched_interests)]
    unique_topics = (
        matched_df[["topic", "difficulty"]]
        .drop_duplicates(subset=["topic"])
        .head(top_n)
    )

    results = []
    for i, row in enumerate(unique_topics.itertuples(), start=1):
        results.append({
            "id": str(i),
            "title": row.topic,
            "difficulty": row.difficulty,
        })

    return results


def generate_learning_path(interest: str = "Web Development") -> List[Dict]:
    """
    Generate a sequential learning roadmap derived from the ML model.
    Uses the best-matching domain and returns its ordered steps.
    """
    vec, data = _get_model()
    if vec is None:
        return []

    df: pd.DataFrame = data["df"]
    unique_interests: List[str] = data["unique_interests"]
    tfidf_matrix = data["tfidf_matrix"]

    # Vectorize query
    query_vec = vec.transform([interest.lower()])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = int(similarities.argmax())
    best_interest = unique_interests[best_idx]

    # Get all steps for the matching domain
    matches = df[df["interest"] == best_interest]
    if matches.empty:
        # Fallback to the first domain if somehow no match is found
        matched_domain_id = df["domain_id"].iloc[0]
    else:
        matched_domain_id = matches["domain_id"].iloc[0]
        
    path_df = (
        df[df["domain_id"] == matched_domain_id]
        .drop_duplicates(subset=["learning_step_order"])
        .sort_values("learning_step_order")
    )

    steps = []
    for row in path_df.itertuples():
        steps.append({
            "id": str(row.learning_step_order),
            "title": row.learning_step,
            "description": str(row.description) if pd.notna(getattr(row, "description", None)) else None,
            "status": "pending",
        })

    return steps
