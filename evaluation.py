import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def train_test_split_interactions(interactions_df, test_size=0.2):
    """
    Simulates an 80/20 Train/Test split for the Recommender Evaluation.
    Ensures that active users preserve at least some history in the training matrix.
    """
    train_list = []
    test_list = []
    np.random.seed(42)  # Fixed seed for stable demo reproduction
    
    for user_id, user_data in interactions_df.groupby('user_id'):
        n_items = len(user_data)
        # Only split if user has more than 2 rated items to avoid continuous cold-start crashing
        if n_items > 2:
            n_test = max(1, int(n_items * test_size))
            test_indices = np.random.choice(user_data.index, n_test, replace=False)
            test_list.append(user_data.loc[test_indices])
            train_list.append(user_data.drop(test_indices))
        else:
            train_list.append(user_data) # Keep in train if too small
            
    train_df = pd.concat(train_list).copy() if train_list else pd.DataFrame(columns=interactions_df.columns)
    test_df = pd.concat(test_list).copy() if test_list else pd.DataFrame(columns=interactions_df.columns)
    
    return train_df, test_df

def evaluate_model(model, test_df, k=3, content_weight=0.5, collab_weight=0.5):
    """
    Calculates primary ML evaluation metrics (Precision, Recall, F1, RMSE)
    Safeguarded mathematically against zero-division and empty dataset crashes.
    """
    # Defensive check for empty test set to prevent structural division by zero crashes
    if test_df.empty:
        return {"Precision@K": 0.0, "Recall@K": 0.0, "F1-Score": 0.0, "RMSE": 0.0}
        
    precisions, recalls, y_true, y_pred = [], [], [], []
    
    for user_id in test_df['user_id'].unique():
        user_test = test_df[test_df['user_id'] == user_id]
        
        # Ground truth: Items the user positively engaged with in the holdout test set (>= 4 rating)
        relevant_items = set(user_test[user_test['rating'] >= 4]['course_id'].tolist())
        
        # VIVA EXPLANATION: We ask the model to predict top K for the user WITHOUT knowing the test set history.
        recs_df = model.get_hybrid_recommendations(user_id, top_n=k, 
                                                   content_weight=content_weight, 
                                                   collab_weight=collab_weight)
        if recs_df.empty:
            continue # Safe-skip if model yields no valid recommendations
            
        rec_items = set(recs_df['course_id'].tolist())
        
        # VIVA EXPLANATION: Hits are items both recommended by the AI and actually liked by the user.
        hits = len(relevant_items.intersection(rec_items))
        
        # Precision@K: Out of K recommendations, what fraction are actually relevant?
        # Safe math: K is minimum 1 via UI slider bounds, so no zero division physically possible.
        precisions.append(hits / k) 
        
        # Recall@K: Out of all relevant items, what fraction did we manage to successfully recommend?
        if len(relevant_items) > 0:
            recalls.append(hits / len(relevant_items))
        else:
            pass # Ignore mathematically if user had 0 relevant items logically
            
        # VIVA EXPLANATION: Calculate deviation (RMSE). Convert similarity score back to 1-5 scale for 1:1 mapping.
        for _, row in user_test.iterrows():
            cid = row['course_id']
            y_true.append(row['rating'])
            
            if cid in recs_df['course_id'].values:
                raw_score = recs_df[recs_df['course_id'] == cid]['final_score'].iloc[0]
                # Map raw 0.0-1.0 sim score loosely back into a 1-5 predicting band mapping regression limits safely
                pred = min(5.0, max(1.0, (raw_score * 4.0) + 1.0))  
                y_pred.append(pred)
            else:
                y_pred.append(2.5)  # Safe penalizing baseline assumption for omitted test courses
                
    # VIVA EXPLANATION: Average out the metrics safely. If completely empty sets, return hard 0.0 mathematically.
    avg_prec = np.mean(precisions) if precisions else 0.0
    avg_rec = np.mean(recalls) if recalls else 0.0
    
    # F1 Score is Harmonic Mean. Safe calculation logic to avert zero division runtime collapse.
    if (avg_prec + avg_rec) > 0:
        f1 = (2 * avg_prec * avg_rec) / (avg_prec + avg_rec)
    else:
        f1 = 0.0
        
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) if y_true else 0.0
    
    return {
        f"Precision@{k}": round(avg_prec, 3),
        f"Recall@{k}": round(avg_rec, 3),
        "F1-Score": round(f1, 3),
        "RMSE": round(rmse, 3)
    }
