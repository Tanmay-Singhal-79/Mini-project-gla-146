import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    def __init__(self, courses_df, interactions_df):
        self.courses_df = courses_df.copy()
        self.interactions_df = interactions_df.copy()
        
        # Pipelines & Matrices
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
        # Vectorize Text
        self.tfidf_matrix = self.vectorizer.fit_transform(self.courses_df['combined_features'])
        
        # Sim Matrix
        self.cosine_sim_content = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def get_user_profile_vector(self, user_id):
        """Constructs an embeded interest vector for a given user based on their high ratings."""
        user_history = self.interactions_df[self.interactions_df['user_id'] == user_id]
        if user_history.empty:
            return np.zeros((1, self.tfidf_matrix.shape[1]))
            
        positive_history = user_history[user_history['rating'] >= 4]
        if positive_history.empty:
            positive_history = user_history # Fallback to any history if no positives exist
            
        course_indices = []
        weights = []
        for _, row in positive_history.iterrows():
            idx_list = self.courses_df.index[self.courses_df['course_id'] == row['course_id']].tolist()
            if idx_list:
                course_indices.append(idx_list[0])
                weights.append(row['rating'])
                
        if not course_indices:
            return np.zeros((1, self.tfidf_matrix.shape[1]))

        # construct weighted user vector
        profile_vector = np.zeros((1, self.tfidf_matrix.shape[1]))
        total_weight = sum(weights)
        for idx, w in zip(course_indices, weights):
            profile_vector += self.tfidf_matrix[idx].toarray() * (w / total_weight)
            
        return profile_vector

    def get_content_recommendations(self, user_id, top_n=5):
        profile_vector = self.get_user_profile_vector(user_id)
        
        # Calculate cosine similarity between user embedded vector and all courses
        sim_scores = cosine_similarity(profile_vector, self.tfidf_matrix).flatten()
        
        # Exclude already processed courses
        user_rated_courses = self.interactions_df[self.interactions_df['user_id'] == user_id]['course_id'].tolist()
        rated_indices = self.courses_df.index[self.courses_df['course_id'].isin(user_rated_courses)].tolist()
        
        # Mask out rated courses
        sim_scores[rated_indices] = -1
        
        top_indices = sim_scores.argsort()[::-1][:top_n]
        
        recs = []
        for idx in top_indices:
            if sim_scores[idx] > 0:
                course = self.courses_df.iloc[idx].to_dict()
                course['similarity_score'] = sim_scores[idx]
                course['explainability'] = f"Content Match: Concept geometry matched your history with {sim_scores[idx]:.2f} similarity."
                course['source'] = 'Content'
                recs.append(course)
                
        return pd.DataFrame(recs)

    def get_collaborative_recommendations(self, user_id, top_n=5):
        user_item_matrix = self.interactions_df.pivot_table(
            index='user_id', columns='course_id', values='rating'
        ).fillna(0)
        
        if user_id not in user_item_matrix.index:
            return pd.DataFrame()
            
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
                    score = sim_score * (row['rating'] / 5.0)
                    if cid not in recommendations or score > recommendations[cid]:
                        recommendations[cid] = score
                        explanations[cid] = f"Collaborative Match: Highlighted by a peer with a {sim_score:.2f} rating correlation."
                        
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
        content_df = self.get_content_recommendations(user_id, top_n=top_n)
        collab_df = self.get_collaborative_recommendations(user_id, top_n=top_n)
        
        hybrid_scores = {}
        explanations = {}
        
        if not content_df.empty:
            for _, row in content_df.iterrows():
                cid = row['course_id']
                hybrid_scores[cid] = hybrid_scores.get(cid, 0) + (row['similarity_score'] * content_weight)
                explanations[cid] = row['explainability']
                
        if not collab_df.empty:
            for _, row in collab_df.iterrows():
                cid = row['course_id']
                hybrid_scores[cid] = hybrid_scores.get(cid, 0) + (row['similarity_score'] * collab_weight)
                
                # Boost explainability string if matches both networks
                if cid in explanations and "Collaborative" not in explanations[cid]:
                    explanations[cid] = f"Hybrid Output: Verified by both Content-Features and Peer-Communities."
                else:
                    explanations[cid] = row['explainability']
                    
        sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        recs = []
        for cid, score in sorted_hybrid:
            course = self.courses_df[self.courses_df['course_id'] == cid].iloc[0].to_dict()
            course['final_score'] = score
            course['explainability'] = explanations[cid]
            recs.append(course)
            
        # Fallback mechanism if empty states
        if not recs:
            popular = self.interactions_df.groupby('course_id')['rating'].mean().sort_values(ascending=False).head(top_n).index
            for cid in popular:
                course = self.courses_df[self.courses_df['course_id'] == cid].iloc[0].to_dict()
                course['final_score'] = 1.0 # arbitrary default scale map
                course['explainability'] = "Popularity Anchor: Rendered via global mean ratings due to lack of profile history."
                recs.append(course)
                
        return pd.DataFrame(recs)

    def adapt_feedback(self, user_id, course_id, rating):
        mask = (self.interactions_df['user_id'] == user_id) & (self.interactions_df['course_id'] == course_id)
        if mask.any():
            self.interactions_df.loc[mask, 'rating'] = rating
        else:
            new_row = pd.DataFrame({'user_id': [user_id], 'course_id': [course_id], 'rating': [rating], 'completed': [True]})
            self.interactions_df = pd.concat([self.interactions_df, new_row], ignore_index=True)
