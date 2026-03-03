import streamlit as st
import pandas as pd
from data import generate_mock_data
from recommender import HybridRecommender
import time

# --- Page Config & Styling ---
st.set_page_config(page_title="CogniPath AI", page_icon="🧠", layout="wide")

# Custom Dark Theme CSS inspired by SaaS
# Using modern typography and premium dark aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background & Text */
    .stApp {
        background-color: #0f1115;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d24;
        border-right: 1px solid #2d3748;
    }
    
    /* Cards for courses */
    .course-card {
        background-color: #1e212b;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .course-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.6);
        border-color: #6366f1;
    }
    .course-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: #818cf8;
        margin-bottom: 12px;
        letter-spacing: -0.025em;
    }
    .course-meta {
        font-size: 0.875rem;
        color: #94a3b8;
        margin-bottom: 12px;
    }
    .tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        background-color: #334155;
        color: #e2e8f0;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 8px;
    }
    
    /* Exander/Accordion tweaks */
    .streamlit-expanderHeader {
        background-color: #1e212b;
        color: #e2e8f0;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization ---
@st.cache_data
def load_data():
    return generate_mock_data()

courses_df, users_df, interactions_df = load_data()

# Inject Session State for Mutability (Learning from Feedback)
if 'interactions' not in st.session_state:
    st.session_state.interactions = interactions_df.copy()

if 'recommender' not in st.session_state:
    st.session_state.recommender = HybridRecommender(courses_df, st.session_state.interactions)

# --- Sidebar: User Profile & Control Panel ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8636/8636883.png", width=50) # simple minimalist brain icon
    st.title("CogniPath AI")
    st.caption("Adaptive Learning Advisory Engine | Version 1.0")
    st.divider()
    
    st.subheader("👤 Architect Profile")
    user_id = st.selectbox("Select Active Learner", users_df['user_id'].tolist(), format_func=lambda x: f"User {x}")
    
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    
    st.markdown(f"**Target Goal:**")
    st.markdown(f"<span style='color: #34d399; font-weight: 600;'>{user_info['target_goal']}</span>", unsafe_allow_html=True)
    
    # Simple Progress Tracking Calculation
    user_history = st.session_state.interactions[st.session_state.interactions['user_id'] == user_id]
    completed = user_history[user_history['completed'] == True]
    
    st.divider()
    st.subheader("📊 System Telemetry")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Path Nodes", len(completed))
    with col2:
        avg_rating = user_history['rating'].mean() if not user_history.empty else 0
        st.metric("Avg Score", f"{avg_rating:.1f}/5")
        
    progress = min(len(completed) * 10, 100) # Mapping len(features) to arbitrary UI progress 
    st.progress(progress / 100.0)
    st.caption(f"Knowledge Track Completion Focus: {progress}%")

# --- Main App Frame ---
st.markdown(f"<h1>Welcome back, {user_info['name']} 👋</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Here is your AI-curated adaptive learning path modeled from your goal trajectory and behavioral feedback.</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Recommended Path", "📚 Enrolled Courses Graph", "⚙️ Neural Architect Log"])

with tab1:
    st.markdown("### AI Recommendations")
    st.caption("Driven by Collaborative Space and TF-IDF Content Similarities.")
    with st.spinner("Processing local heuristic layers & generating tracks..."):
        time.sleep(0.4) # Synthetic delay to give the "AI SaaS" feel
        
        # Trigger predictive logic
        st.session_state.recommender.interactions_df = st.session_state.interactions # sync state
        recs = st.session_state.recommender.get_hybrid_recommendations(user_id, top_n=4)
        
    if recs.empty:
        st.info("System needs more data context. Try exploring generic tracks first!")
    else:
        cols = st.columns(2)
        for i, (_, course) in enumerate(recs.iterrows()):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="course-card">
                    <div class="course-title">{course['title']}</div>
                    <div class="course-meta">
                        <span class="tag">{course['category']}</span>
                        <span class="tag">{course['difficulty']}</span>
                    </div>
                    <p style='color: #cbd5e1; font-size: 0.95rem;'>{course['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Dynamic Feedback Loop UI
                with st.expander(f"Enrol & Calibrate Model ({course['title']})"):
                    rating = st.slider("Assess Course Quality (1-5)", 1, 5, 3, key=f"rate_{course['course_id']}")
                    if st.button("Commit Feedback", key=f"btn_{course['course_id']}", help="This shifts the recommendation vectors."):
                        st.session_state.recommender.adapt_feedback(user_id, course['course_id'], rating)
                        # Sync back to UI state
                        st.session_state.interactions = st.session_state.recommender.interactions_df
                        st.success("Feedback mapped into matrix! Recommendations updated. Refresh view to see adapted paths.")

with tab2:
    st.markdown("### Completed Track Graph")
    history_detailed = pd.merge(st.session_state.interactions[st.session_state.interactions['user_id'] == user_id], courses_df, on="course_id")
    if history_detailed.empty:
        st.info("Your knowledge graph is currently empty.")
    else:
        for _, row in history_detailed.iterrows():
            st.markdown(f"**{row['title']}** - Calibrated Metric: {'⭐'*int(row['rating'])}")
            st.caption(f"{row['category']} • {row['difficulty']}")
            st.divider()

with tab3:
    st.markdown("### Recommender Mathematical State Diagnostics")
    st.markdown("Raw data dump from the underlying Scikit-Learn instances:")
    st.code("""
[INFO] Booting hybrid recommender engine...
[INFO] Instantiating TF-IDF vectorizers (Stop Words=English)
[INFO] Building Item-Item content relationships...
[INFO] Parsing Matrix Sparsity.
[ACTION] Feedback loop active. Listener online.
    """, language="bash")
    
    st.markdown("#### Course Metadata Vector Tensors (TF-IDF Fragment)")
    if st.session_state.recommender.tfidf_matrix is not None:
        # Show a slice of the mathematical matrix
        df_slice = pd.DataFrame(st.session_state.recommender.tfidf_matrix.toarray()[:5, :5])
        df_slice.columns = [f"Token_{i}" for i in range(5)]
        st.dataframe(df_slice, use_container_width=True)
        st.caption("Visual proof of vectorized description fields.")
