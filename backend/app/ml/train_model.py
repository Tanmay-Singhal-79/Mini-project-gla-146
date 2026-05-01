"""
train_model.py
Trains a TF-IDF + Cosine Similarity recommendation model on learning_data.csv.
Saves:
  - vectorizer.pkl  (fitted TfidfVectorizer)
  - model.pkl       (compressed TF-IDF matrix + metadata DataFrame)
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ML_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(ML_DIR, "learning_data.csv")
VECTORIZER_PATH = os.path.join(ML_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(ML_DIR, "model.pkl")


def train():
    print("[*] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"[OK] Loaded {len(df)} records.")

    # --- Build corpus: unique interests for vectorization ---
    unique_interests = df["interest"].unique().tolist()
    print(f"[*] Unique interests: {len(unique_interests)}")

    # --- Fit TF-IDF Vectorizer on interest phrases ---
    print("[*] Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer="word",
        min_df=1,
        stop_words=None,
    )
    tfidf_matrix = vectorizer.fit_transform(unique_interests)
    print(f"[OK] TF-IDF matrix shape: {tfidf_matrix.shape}")

    # --- Package model artifacts ---
    model_data = {
        "unique_interests": unique_interests,
        "tfidf_matrix": tfidf_matrix,
        "df": df,
    }

    # --- Persist ---
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[OK] Vectorizer saved -> {VECTORIZER_PATH}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"[OK] Model data saved -> {MODEL_PATH}")

    print("\n[OK] Training complete. Model is ready to serve recommendations.")


if __name__ == "__main__":
    train()
