import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def train_test_split_interactions(interactions_df, test_size=0.2):
    """
    Simulates an 80/20 Train/Test split for the Recommender Evaluation.
    Ensures that active users preserve history in the training matrix.
    """
    train_list = []
    test_list = []
    np.random.seed(42)  # For reproducible evaluation
    
    for user_id, user_data in interactions_df.groupby('user_id'):
        n_items = len(user_data)
        if n_items > 2:
            n_test = max(1, int(n_items * test_size))
            test_indices = np.random.choice(user_data.index, n_test, replace=False)
            test_list.append(user_data.loc[test_indices])
            train_list.append(user_data.drop(test_indices))
        else:
            train_list.append(user_data)
            
    train_df = pd.concat(train_list).copy() if train_list else pd.DataFrame(columns=interactions_df.columns)
    test_df = pd.concat(test_list).copy() if test_list else pd.DataFrame(columns=interactions_df.columns)
    
    return train_df, test_df

def evaluate_model(model, test_df, k=3, content_weight=0.5, collab_weight=0.5):
    """
    Calculates primary ML evaluation metrics (Precision, Recall, F1, RMSE)
    """
    if test_df.empty:
        return {"Precision@K": 0.0, "Recall@K": 0.0, "F1-Score": 0.0, "RMSE": 0.0}
        
    precisions, recalls, y_true, y_pred = [], [], [], []
    
    for user_id in test_df['user_id'].unique():
        user_test = test_df[test_df['user_id'] == user_id]
        # Ground truth: Items the user positively engaged with (>= 4 rating)
        relevant_items = set(user_test[user_test['rating'] >= 4]['course_id'].tolist())
        
        # Extracted Recommendations based on hyperparams
        recs_df = model.get_hybrid_recommendations(user_id, top_n=k, 
                                                   content_weight=content_weight, 
                                                   collab_weight=collab_weight)
        if recs_df.empty:
            continue
            
        rec_items = set(recs_df['course_id'].tolist())
        
        # --- Classification Metrics ---
        hits = len(relevant_items.intersection(rec_items))
        precisions.append(hits / k) # Precision@K
        
        if len(relevant_items) > 0:
            recalls.append(hits / len(relevant_items)) # Recall@K
            
        # --- Value Metrics (RMSE formulation) ---
        for _, row in user_test.iterrows():
            cid = row['course_id']
            y_true.append(row['rating'])
            
            if cid in recs_df['course_id'].values:
                raw_score = recs_df[recs_df['course_id'] == cid]['final_score'].iloc[0]
                # Map raw 0.0-1.0 sim score loosely back into a 1-5 predicting band
                pred = min(5.0, max(1.0, (raw_score * 4.0) + 1.0))  
                y_pred.append(pred)
            else:
                y_pred.append(2.5)  # Penalizing assumption for omitted test courses
                
    avg_prec = np.mean(precisions) if precisions else 0.0
    avg_rec = np.mean(recalls) if recalls else 0.0
    f1 = (2 * avg_prec * avg_rec) / (avg_prec + avg_rec) if (avg_prec + avg_rec) > 0 else 0.0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) if y_true else 0.0
    
    return {
        f"Precision@{k}": round(avg_prec, 3),
        f"Recall@{k}": round(avg_rec, 3),
        "F1-Score": round(f1, 3),
        "RMSE": round(rmse, 3)
    }
