import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    def __init__(self, courses_df, interactions_df):
        self.courses_df = courses_df.copy()
        self.interactions_df = interactions_df.copy()
        
        # State variables for Content-Based filtering
        self.tfidf_matrix = None
        self.cosine_sim_content = None
        self._build_content_model()
        
    def _build_content_model(self):
        """Builds the TF-IDF Matrix and Cosine Similarity for courses."""
        # Combining text features to create a robust profile of the course
        self.courses_df['combined_features'] = (
            self.courses_df['category'] + " " + 
            self.courses_df['difficulty'] + " " + 
            self.courses_df['description']
        )
        
        vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = vectorizer.fit_transform(self.courses_df['combined_features'])
        
        # Calculate similarity between all items in the matrix
        self.cosine_sim_content = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
    def get_content_recommendations(self, course_id, top_n=3):
        """Returns top_n similar courses based on NLP text similarity."""
        idx_list = self.courses_df.index[self.courses_df['course_id'] == course_id].tolist()
        if not idx_list:
            return pd.DataFrame()
        
        idx = idx_list[0]
        sim_scores = list(enumerate(self.cosine_sim_content[idx]))
        
        # Sort courses based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the targeted top_n (skip the first one as it is the queried course itself)
        sim_scores = sim_scores[1:top_n+1]
        course_indices = [i[0] for i in sim_scores]
        
        return self.courses_df.iloc[course_indices]

    def get_collaborative_recommendations(self, user_id, top_n=3):
        """Returns top_n course recommendations based on peer (similar user) behaviors."""
        # Construct User-Item Matrix
        user_item_matrix = self.interactions_df.pivot_table(
            index='user_id', columns='course_id', values='rating'
        ).fillna(0)
        
        if user_id not in user_item_matrix.index:
            return pd.DataFrame() # Handle Cold Start gracefully
            
        # Calculate user cosine similarity
        user_sim = cosine_similarity(user_item_matrix)
        user_sim_df = pd.DataFrame(
            user_sim, 
            index=user_item_matrix.index, 
            columns=user_item_matrix.index
        )
        
        # Find exactly top 3 most similar peers (excluding the user themselves)
        similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:4].index
        
        target_user_rated = self.interactions_df[self.interactions_df['user_id'] == user_id]['course_id'].tolist()
        
        recommendations = {}
        for su_id in similar_users:
            su_ratings = self.interactions_df[self.interactions_df['user_id'] == su_id]
            for _, row in su_ratings.iterrows():
                cid = row['course_id']
                if cid not in target_user_rated:
                    # Weigh rating slightly by peer similarity
                    recommendations[cid] = recommendations.get(cid, 0) + row['rating']
                        
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        rec_course_ids = [x[0] for x in sorted_recs]
        
        return self.courses_df[self.courses_df['course_id'].isin(rec_course_ids)]
        
    def get_hybrid_recommendations(self, user_id, top_n=5):
        """
        Aggregates Content and Collaborative recommendations to build a master learning path.
        Adapts dynamically based on what the user has rated highly.
        """
        user_history = self.interactions_df[self.interactions_df['user_id'] == user_id]
        
        # Filter for explicitly preferred courses (rated >= 4)
        positive_history = user_history[user_history['rating'] >= 4]
        
        content_recs_list = []
        for course_id in positive_history['course_id']:
            content_recs_list.append(self.get_content_recommendations(course_id, top_n=2))
            
        if content_recs_list:
            content_df = pd.concat(content_recs_list).drop_duplicates()
        else:
            content_df = pd.DataFrame()
            
        collab_df = self.get_collaborative_recommendations(user_id, top_n=3)
        
        hybrid_df = pd.concat([content_df, collab_df]).drop_duplicates(subset=['course_id']).head(top_n)
        
        # Fallback mechanism for cold-starts/unresolved profiles
        if hybrid_df.empty:
            popular = self.interactions_df.groupby('course_id')['rating'].mean().sort_values(ascending=False).head(top_n).index
            hybrid_df = self.courses_df[self.courses_df['course_id'].isin(popular)]
            
        return hybrid_df

    def adapt_feedback(self, user_id, course_id, rating):
        """
        Adaptation logic: Registers the user feedback into the DataFrame, triggering 
        a state recalculation for the next collaborative and hybrid filtering cycle.
        """
        mask = (self.interactions_df['user_id'] == user_id) & (self.interactions_df['course_id'] == course_id)
        
        if mask.any():
            # Adjust weight if row already exists
            self.interactions_df.loc[mask, 'rating'] = rating
        else:
            # Create a new graph connection in the matrix
            new_row = pd.DataFrame({
                'user_id': [user_id], 
                'course_id': [course_id], 
                'rating': [rating], 
                'completed': [True]
            })
            self.interactions_df = pd.concat([self.interactions_df, new_row], ignore_index=True)
