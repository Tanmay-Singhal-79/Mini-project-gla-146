import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleRecommender:
    def __init__(self, courses_df):
        self.courses_df = courses_df.copy()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self._build_model()
        
    def _build_model(self):
        self.courses_df['features'] = self.courses_df['tags'] + " " + self.courses_df['desc']
        self.tfidf_matrix = self.vectorizer.fit_transform(self.courses_df['features'])
        
    def recommend(self, user_interest, user_history=None):
        search_query = user_interest
        
        if user_history is not None and not user_history.empty:
            good_courses = user_history[user_history['rating'] >= 4]['course_id']
            for cid in good_courses:
                course_text = self.courses_df[self.courses_df['id'] == cid]['features'].iloc[0]
                search_query += " " + course_text
                
        user_vector = self.vectorizer.transform([search_query])
        
        sim_scores = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        
        top_indices = sim_scores.argsort()[::-1][:3]
        
        results = []
        for idx in top_indices:
            course = self.courses_df.iloc[idx].to_dict()
            course['sim_score'] = sim_scores[idx]
            results.append(course)
            
        return pd.DataFrame(results)
