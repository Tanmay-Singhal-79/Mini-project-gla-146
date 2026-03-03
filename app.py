import streamlit as st
import pandas as pd
import time
import numpy as np

# Use updated modular imports
from data import generate_mock_data
from model import HybridRecommender
from evaluation import train_test_split_interactions, evaluate_model
from utils import apply_custom_css

# --- Page Config & Styling ---
st.set_page_config(page_title="CogniPath AI", page_icon="🧠", layout="wide")
apply_custom_css()

# --- Initialization ---
@st.cache_data
def load_data():
    return generate_mock_data()

courses_df, users_df, interactions_df = load_data()

if 'interactions' not in st.session_state:
    st.session_state.interactions = interactions_df.copy()

if 'recommender' not in st.session_state:
    st.session_state.recommender = HybridRecommender(courses_df, st.session_state.interactions)

# --- Sidebar: User Profile & Control Panel ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8636/8636883.png", width=50) 
    st.title("CogniPath AI")
    st.caption("Adaptive Learning Advisory Engine | Version 2.0")
    st.divider()
    
    st.subheader("👤 Architect Profile")
    user_id = st.selectbox("Select Active Learner", users_df['user_id'].tolist(), format_func=lambda x: f"User {x}")
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    st.markdown(f"**Target Goal:** <span style='color: #34d399; font-weight: 600;'>{user_info['target_goal']}</span>", unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("⚙️ Hyperparameter Tuning")
    k_val = st.slider("Top K Recommendations", 1, 10, 4)
    st.caption("Adjust Model Vector Weightings:")
    c_weight = st.slider("Content Filter Ratio %", 0, 100, 60, help="100% means purely NLP text matching. 0% means pure User collaborative filtering.") 
    
    content_weight = c_weight / 100.0
    collab_weight = (100 - c_weight) / 100.0
    
    # Progress Tracking
    user_history = st.session_state.interactions[st.session_state.interactions['user_id'] == user_id]
    completed = user_history[user_history['completed'] == True]
    progress = min(len(completed) * 10, 100) 
    
    st.divider()
    st.progress(progress / 100.0)
    st.caption(f"Knowledge Track Completion Focus: {progress}%")


# --- Main App Frame ---
st.markdown(f"<h1>Welcome back, {user_info['name']} 👋</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Explore your personalized, academically verified learning trajectories below.</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Recommended Path", "📉 System Evaluation", "🔬 ML Demonstration Mode"])

with tab1:
    st.markdown("### AI Recommendations")
    st.caption("Algorithmically blended targets. Rate courses to instantly shift trajectory embedding tensors.")
    
    # Sync DB state to recommender
    st.session_state.recommender.interactions_df = st.session_state.interactions 
    
    with st.spinner("Processing vector logic..."):
        recs = st.session_state.recommender.get_hybrid_recommendations(
            user_id, top_n=k_val, content_weight=content_weight, collab_weight=collab_weight
        )
        
    if recs.empty:
        st.info("System needs more context.")
    else:
        # Lay out results dynamically
        cols = st.columns(min(len(recs), 2) if len(recs) > 0 else 1)
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
                    <span class="sim-score">Sim Score: {course['final_score']:.3f}</span>
                    <div class="explain-text">{course['explainability']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Feedback loop UI
                with st.expander(f"Enrol & Calibrate Model ({course['title']})"):
                    rating = st.slider("Assess Quality (1-5)", 1, 5, 3, key=f"rate_{course['course_id']}")
                    if st.button("Commit Feedback", key=f"btn_{course['course_id']}"):
                        st.session_state.recommender.adapt_feedback(user_id, course['course_id'], rating)
                        st.session_state.interactions = st.session_state.recommender.interactions_df
                        st.success("Matrix updated! Re-render tabs to see new trajectory.")


with tab2:
    st.markdown("### Model Evaluation Metrics (Train/Test Split Strategy)")
    st.markdown("Executing simulated 80/20 offline evaluation protocol on historical data.")
    
    if st.button("▶ Run Full Corpus Evaluation Test"):
        with st.spinner("Splitting matrices and generating simulated loss maps..."):
            train_df, test_df = train_test_split_interactions(st.session_state.interactions)
            
            # Spin up evaluation model subset
            eval_model = HybridRecommender(courses_df, train_df)
            metrics = evaluate_model(eval_model, test_df, k=k_val, content_weight=content_weight, collab_weight=collab_weight)
            
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric(f"Precision@{k_val}", f"{metrics[f'Precision@{k_val}']}")
            mc2.metric(f"Recall@{k_val}", f"{metrics[f'Recall@{k_val}']}")
            mc3.metric("F1-Score", f"{metrics['F1-Score']}")
            mc4.metric("RMSE Penalty", f"{metrics['RMSE']}")
            
            st.caption("Note: Since we are using an extremely sparse mock dataset mapping 20 courses to 10 users, metrics will be inherently volatile. Real-world implementations utilize thousands of samples for stable regression.")


with tab3:
    st.markdown("### Explainable AI & Vector Space Validation")
    st.markdown("This diagnostic tab exposes the underlying Scikit-Learn structures in real-time, functioning as an academic project demonstrator.")
    
    uv = st.session_state.recommender.get_user_profile_vector(user_id)
    n_features = st.session_state.recommender.tfidf_matrix.shape[1]
    
    st.code(f"""
[DIAGNOSTICS - USER {user_id}]
> Vocabulary Features Extracted: {n_features} Words
> User Profile Vector Shape: {uv.shape}
> Model Base Weight Configuration: Content ({(content_weight)*100:.0f}%) / Collaborative ({(collab_weight)*100:.0f}%)
    """)
    
    st.markdown("#### User Mathematical Trace Embedding")
    st.caption("Displaying the top 15 Non-Zero active vector weights representing user's underlying knowledge features.")
    
    flat_uv = uv.flatten()
    top_feature_indices = flat_uv.argsort()[::-1][:15]
    feature_names = st.session_state.recommender.vectorizer.get_feature_names_out()
    
    vector_df = pd.DataFrame({
        "TF-IDF Token": [feature_names[i] for i in top_feature_indices],
        "Token Significance Weight": [flat_uv[i] for i in top_feature_indices]
    })
    
    st.dataframe(vector_df.style.background_gradient(cmap="viridis"), use_container_width=True)
    
    st.markdown("#### Cosine Formulations")
    st.latex(r"Sim(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|\|\mathbf{B}\|}")
    st.caption("Cosine similarity measures the angle between the user's vector array (shown above) and the available course description vector arrays, driving the content filtering output.")
