import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleRecommender:
    def __init__(self, courses_df):
        self.courses_df = courses_df.copy()
        # VIVA: TF-IDF translates standard sentences into mathematical number arrays (vectors).
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self._build_model()
        
    def _build_model(self):
        """Builds combined text column and fits mathematical vectorizer."""
        self.courses_df['features'] = self.courses_df['tags'] + " " + self.courses_df['desc']
        self.tfidf_matrix = self.vectorizer.fit_transform(self.courses_df['features'])
        
    def recommend(self, user_interest, user_history=None):
        """
        Uses explicit user interest (e.g. 'Data Science') combined organically 
        with any courses they liked (4+) in the past to calculate Cosine mathematics.
        """
        # 1. Base interest string (e.g. from dropdown: "Data Science")
        search_query = user_interest
        
        # 2. Append history if they rated a course well (>= 4)
        if user_history is not None and not user_history.empty:
            good_courses = user_history[user_history['rating'] >= 4]['course_id']
            for cid in good_courses:
                course_text = self.courses_df[self.courses_df['id'] == cid]['features'].iloc[0]
                search_query += " " + course_text  # Make the user's focus "heavier" in math
                
        # 3. Vectorize the entire search query mechanically into a new math array
        user_vector = self.vectorizer.transform([search_query])
        
        # 4. Find the angle between the new User Vector and all Course Vectors (1.0 = identical)
        sim_scores = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        
        # 5. Get top 3 indices and format them out cleanly
        top_indices = sim_scores.argsort()[::-1][:3]
        
        results = []
        for idx in top_indices:
            course = self.courses_df.iloc[idx].to_dict()
            course['sim_score'] = sim_scores[idx]
            results.append(course)
            
        return pd.DataFrame(results)
