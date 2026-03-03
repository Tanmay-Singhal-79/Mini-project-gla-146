import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    def __init__(self, courses_df, interactions_df):
        self.courses_df = courses_df.copy()
        self.interactions_df = interactions_df.copy()
        
        # Pipelines & Matrices
        # VIVA EXPLANATION: Removing english stop words (the, a, is) to keep only strict mathematical ML keywords.
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.cosine_sim_content = None
        
        self._build_content_model()
        
    def _build_content_model(self):
        """Builds Course Vector Database."""
        self.courses_df['combined_features'] = (
            self.courses_df['category'] + " " + 
            self.courses_df['difficulty'] + " " + 
            self.courses_df['description']
        )
        # VIVA EXPLANATION: Fit_transform directly converts the raw text into a sparse numeric matrix of weighted vocabulary structures.
        self.tfidf_matrix = self.vectorizer.fit_transform(self.courses_df['combined_features'])
        
        # VIVA EXPLANATION: Computes angular distance natively between all courses. 1 = same, 0 = entirely no overlapping words/mathematical structure.
        self.cosine_sim_content = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def get_user_profile_vector(self, user_id):
        """Constructs an embeded interest vector for a given user based dynamically on their high ratings."""
        user_history = self.interactions_df[self.interactions_df['user_id'] == user_id]
        
        # Safe return if blank user protecting from crashes
        if user_history.empty:
            return np.zeros((1, self.tfidf_matrix.shape[1]))
            
        positive_history = user_history[user_history['rating'] >= 4]
        if positive_history.empty:
            positive_history = user_history # Fallback safely structurally to any rating if specifically no positives exist
            
        course_indices = []
        weights = []
        for _, row in positive_history.iterrows():
            idx_list = self.courses_df.index[self.courses_df['course_id'] == row['course_id']].tolist()
            if idx_list:
                course_indices.append(idx_list[0])
                weights.append(row['rating'])
                
        if not course_indices:
            return np.zeros((1, self.tfidf_matrix.shape[1]))

        # VIVA EXPLANATION: We physically create the User's Persona matrix by mathematically aggregating the exact TF-IDF vectors of the 
        # courses they liked, multiplied dynamically by how aggressively they liked it (weight).
        profile_vector = np.zeros((1, self.tfidf_matrix.shape[1]))
        total_weight = sum(weights)
        for idx, w in zip(course_indices, weights):
            profile_vector += self.tfidf_matrix[idx].toarray() * (w / total_weight)
            
        return profile_vector

    def get_content_recommendations(self, user_id, top_n=5):
        """Finds items specifically identical matching the user's isolated array token geography."""
        profile_vector = self.get_user_profile_vector(user_id)
        
        # VIVA EXPLANATION: Finding geometric cosine angle similarity precisely between the aggregated personalized user node and all global items.
        sim_scores = cosine_similarity(profile_vector, self.tfidf_matrix).flatten()
        
        # Exclude courses the active user has already successfully touched.
        user_rated_courses = self.interactions_df[self.interactions_df['user_id'] == user_id]['course_id'].tolist()
        rated_indices = self.courses_df.index[self.courses_df['course_id'].isin(user_rated_courses)].tolist()
        sim_scores[rated_indices] = -1 # Soft disqualify rated items seamlessly and cleanly logically without reshaping
        
        top_indices = sim_scores.argsort()[::-1][:top_n]
        
        recs = []
        for idx in top_indices:
            if sim_scores[idx] > 0:
                course = self.courses_df.iloc[idx].to_dict()
                course['similarity_score'] = sim_scores[idx]
                course['explainability'] = f"Content Match: Concept geometry naturally matched your history with {sim_scores[idx]:.2f} similarity."
                course['source'] = 'Content'
                recs.append(course)
                
        return pd.DataFrame(recs)

    def get_collaborative_recommendations(self, user_id, top_n=5):
        """Finds active items reliably loved by highly correlated mathematical peer users in array index."""
        # VIVA EXPLANATION: Build user-item isolated sparse matrix for peers to match profiles explicitly mapping on common ratings intersection purely.
        user_item_matrix = self.interactions_df.pivot_table(
            index='user_id', columns='course_id', values='rating'
        ).fillna(0)
        
        if user_id not in user_item_matrix.index:
            return pd.DataFrame()
            
        # VIVA EXPLANATION: Evaluate Users who technically behave remarkably similar to the targeted active user via direct correlation.
        user_sim = cosine_similarity(user_item_matrix)
        user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
        
        similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:4]
        target_user_rated = self.interactions_df[self.interactions_df['user_id'] == user_id]['course_id'].tolist()
        
        recommendations = {}
        explanations = {}
        for su_id, sim_score in similar_users.items():
            if sim_score <= 0.05: continue
            
            su_ratings = self.interactions_df[self.interactions_df['user_id'] == su_id]
            for _, row in su_ratings.iterrows():
                cid = row['course_id']
                if cid not in target_user_rated and row['rating'] >= 4:
                    # VIVA EXPLANATION: Adjust recommendation rating significance accurately scaling based strictly on how identically similar the peer network mathematically natively aligns in history sets.
                    score = sim_score * (row['rating'] / 5.0)
                    if cid not in recommendations or score > recommendations[cid]:
                        recommendations[cid] = score
                        explanations[cid] = f"Collaborative Match: Strongly isolated by community peer with a sharp {sim_score:.2f} behavior correlation."
                        
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        recs = []
        for cid, score in sorted_recs:
            course = self.courses_df[self.courses_df['course_id'] == cid].iloc[0].to_dict()
            course['similarity_score'] = score
            course['explainability'] = explanations[cid]
            course['source'] = 'Collaborative'
            recs.append(course)
            
        return pd.DataFrame(recs)
        
    def get_hybrid_recommendations(self, user_id, top_n=5, content_weight=0.5, collab_weight=0.5):
        """Merges natively the two structurally discrete neural branches."""
        content_df = self.get_content_recommendations(user_id, top_n=top_n)
        collab_df = self.get_collaborative_recommendations(user_id, top_n=top_n)
        
        hybrid_scores = {}
        explanations = {}
        
        # Apply strict dynamically mapped Content Ratio
        if not content_df.empty:
            for _, row in content_df.iterrows():
                cid = row['course_id']
                hybrid_scores[cid] = hybrid_scores.get(cid, 0) + (row['similarity_score'] * content_weight)
                explanations[cid] = row['explainability']
                
        # Apply Collaborative active tuned Ratio map.
        if not collab_df.empty:
            for _, row in collab_df.iterrows():
                cid = row['course_id']
                hybrid_scores[cid] = hybrid_scores.get(cid, 0) + (row['similarity_score'] * collab_weight)
                
                # Intercept explainability string and boost technically
                if cid in explanations and "Collaborative" not in explanations[cid]:
                    explanations[cid] = f"Hybrid Verified Node: Architecturally checked valid by both Content-Feature map structures natively and matching Peer-Community arrays."
                else:
                    explanations[cid] = row['explainability']
                    
        sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        recs = []
        for cid, score in sorted_hybrid:
            course = self.courses_df[self.courses_df['course_id'] == cid].iloc[0].to_dict()
            course['final_score'] = score
            course['explainability'] = explanations[cid]
            recs.append(course)
            
        # VIVA EXPLANATION: Fallback default handler safely deploys global mean popular items logically resolving cold start constraints
        if not recs:
            popular = self.interactions_df.groupby('course_id')['rating'].mean().sort_values(ascending=False).head(top_n).index
            for cid in popular:
                course = self.courses_df[self.courses_df['course_id'] == cid].iloc[0].to_dict()
                course['final_score'] = 1.0 # purely arbitrary stability baseline
                course['explainability'] = "Popularity Anchor Engine: Handled seamlessly via safe dynamic global mean ratings fallback logic directly avoiding cold profile history map issues universally."
                recs.append(course)
                
        return pd.DataFrame(recs)

    def adapt_feedback(self, user_id, course_id, rating):
        """Immediately alters logic states reliably mutating matrix tensors."""
        mask = (self.interactions_df['user_id'] == user_id) & (self.interactions_df['course_id'] == course_id)
        if mask.any():
            self.interactions_df.loc[mask, 'rating'] = rating
        else:
            new_row = pd.DataFrame({'user_id': [user_id], 'course_id': [course_id], 'rating': [rating], 'completed': [True]})
            self.interactions_df = pd.concat([self.interactions_df, new_row], ignore_index=True)
