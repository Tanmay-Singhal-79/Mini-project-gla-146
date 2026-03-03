import streamlit as st
import pandas as pd
from data import get_data
from model import SimpleRecommender

# --- Basic Config & App Setup ---
st.set_page_config(page_title="CogniPath AI", page_icon="📚", layout="wide")

# Premium Dark Styling natively via basic CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    
    .stApp { background-color: #0f1115; }
    h1, h2, h3, h4 { color: #f8fafc !important; }
    
    .course-card {
        background-color: #1e212b;
        border-radius: 12px; border: 1px solid #334155;
        padding: 24px; margin-bottom: 24px;
        transition: transform 0.2s;
    }
    .course-card:hover {
        transform: translateY(-4px);
        border-color: #6366f1;
    }
    .c-title {
        color: #818cf8; font-size: 1.35rem; font-weight: 600; margin-bottom: 8px;
    }
    .c-tags {
        color: #94a3b8; font-size: 0.85rem; margin-bottom: 15px; 
    }
    .sim-badge {
        display: inline-block; background-color: #334155; color: #34d399;
        padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['course_id', 'title', 'rating'])

# Load Model natively
courses_df = get_data()
recommender = SimpleRecommender(courses_df)

# --- Streamlit Basic Layout ---
st.title("📚 CogniPath AI")
st.caption("A Minimalist Content-Based Course Recommender Engine")
st.divider()

# Core UI Flow
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("👤 Step 1: Tell AI your goal")
    user_interest = st.selectbox(
        "Select your primary focus point", 
        ["Data Science", "Web Development", "Cyber Security", "Python Programming", "Machine Learning"]
    )
    
    st.divider()
    
    st.markdown("**📊 Step 2: Track Progress**")
    total_courses_available = len(courses_df)
    completed_courses = len(st.session_state.history)
    progress_val = (completed_courses / total_courses_available) if completed_courses > 0 else 0.0
    
    st.progress(progress_val)
    st.caption(f"Knowledge Database Exhausted: {int(progress_val * 100)}%")
    
    if not st.session_state.history.empty:
        st.markdown("**Your Learning DB:**")
        st.dataframe(st.session_state.history[['title', 'rating']], use_container_width=True)

with col2:
    st.subheader("🤖 AI Extracted Tracks")
    st.markdown("We use Natural Language Processing (**TF-IDF**) to convert your interests into math, and **Cosine Similarity** to match angles against course descriptions.")
    
    # Generate recommendations natively based on State History + String Input
    recs = recommender.recommend(user_interest, st.session_state.history)
    
    if recs.empty:
        st.warning("No matches found in our DB.")
    else:
        for i, row in recs.iterrows():
            st.markdown(f"""
            <div class="course-card">
                <div class="c-title">{row['title']} <span class="sim-badge">Match: {row['sim_score']:.2f}</span></div>
                <div class="c-tags">Categories: {row['tags']}</div>
                <div style="color: #cbd5e1; font-size: 0.95rem;">{row['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Simple Feedback implementation
            rating = st.slider(f"Rate '{row['title']}' quality:", 1, 5, 3, key=f"rate_{row['id']}")
            
            # Hide submit button natively if they already rated it
            has_rated = not st.session_state.history.empty and (row['id'] in st.session_state.history['course_id'].values)
            
            if not has_rated:
                if st.button(f"Submit Log", key=f"btn_{row['id']}", help="Ratings 4+ will dynamically influence your future path geometry."):
                    new_entry = pd.DataFrame({'course_id': [row['id']], 'title': [row['title']], 'rating': [rating]})
                    st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)
                    st.success("Log Saved! System Re-calibrating...")
                    st.rerun()  # Forces immediately visible reload
            else:
                st.info("You possess prior experience intersecting with this module.")
