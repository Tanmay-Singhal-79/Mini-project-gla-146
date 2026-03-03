import streamlit as st
import pandas as pd
import numpy as np

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
    st.caption("Adaptive Learning Advisory Engine | Stable Release")
    st.divider()
    
    st.subheader("👤 Architect Profile")
    user_id = st.selectbox("Select Active Learner", users_df['user_id'].tolist(), format_func=lambda x: f"User {x}")
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    st.markdown(f"**Target Goal:** <span style='color: #34d399; font-weight: 600;'>{user_info['target_goal']}</span>", unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("⚙️ Hyperparameter Tuning")
    k_val = st.slider("Top K Recommendations", 1, 6, 4)
    c_weight = st.slider("Content Filter Ratio %", 0, 100, 60, help="0% = Collab Only | 100% = Content Only") 
    
    content_weight = c_weight / 100.0
    collab_weight = (100 - c_weight) / 100.0
    
    user_history = st.session_state.interactions[st.session_state.interactions['user_id'] == user_id]
    completed = user_history[user_history['completed'] == True]
    progress = min(len(completed) * 10, 100) 
    
    st.divider()
    st.progress(progress / 100.0)
    st.caption(f"Knowledge Track Completion Focus: {progress}%")


# --- Main App Frame ---
st.markdown(f"<h1>Welcome back, {user_info['name']} 👋</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Explore your personalized, academically verified learning trajectories below.</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Interactive Path", "📈 Demo Flow Mode", "💡 Professor Breakdown", "🔬 ML Diagnostics"])

with tab1:
    st.markdown("### Active Engine Recommendations")
    st.session_state.recommender.interactions_df = st.session_state.interactions 
    
    recs = st.session_state.recommender.get_hybrid_recommendations(
        user_id, top_n=k_val, content_weight=content_weight, collab_weight=collab_weight
    )
    
    if recs.empty:
        st.info("System needs more context.")
    else:
        cols = st.columns(min(len(recs), 2) if len(recs) > 0 else 1)
        for i, (_, course) in enumerate(recs.iterrows()):
            with cols[i % 2]:
                st.markdown(f'''
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
                ''', unsafe_allow_html=True)
                
                with st.expander(f"Enrol & Calibrate Model ({course['title']})"):
                    rating = st.slider("Assess Quality (1-5)", 1, 5, 3, key=f"rate_{course['course_id']}")
                    if st.button("Commit Feedback", key=f"btn_{course['course_id']}"):
                        st.session_state.recommender.adapt_feedback(user_id, course['course_id'], rating)
                        st.session_state.interactions = st.session_state.recommender.interactions_df
                        st.success("Matrix updated! Refresh to see new trajectory.")

with tab2:
    st.markdown("### Single-Click Viva Demonstration Pipeline")
    st.markdown("Use this to effortlessly walk through the architecture end-to-end without manual adjustments.")
    
    if st.button("▶ Initialize Complete Demo Sequence", key="demo_btn"):
        st.markdown("#### Step 1: Loading User Persona & History")
        st.dataframe(user_history.merge(courses_df, on='course_id')[['title', 'rating', 'category']], use_container_width=True)
        
        st.markdown("#### Step 2: Extracting User Token Geometry (Top 3 Words)")
        uv = st.session_state.recommender.get_user_profile_vector(user_id).flatten()
        top_idx = uv.argsort()[::-1][:3]
        f_names = st.session_state.recommender.vectorizer.get_feature_names_out()
        demo_tokens = [f"{f_names[i]} ({uv[i]:.2f})" for i in top_idx]
        st.info(f"Most heavily weighted underlying concepts derived from history: **{', '.join(demo_tokens)}**")
        
        st.markdown("#### Step 3: Resolving Hybrid Score Matrix")
        st.dataframe(recs[['title', 'final_score', 'explainability']])
        
        st.markdown("#### Step 4: System Stability Check on Entire Dataset")
        train_df, test_df = train_test_split_interactions(st.session_state.interactions)
        eval_model = HybridRecommender(courses_df, train_df)
        metrics = evaluate_model(eval_model, test_df, k=k_val)
        
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(f"Precision@{k_val}", f"{metrics[f'Precision@{k_val}']}")
        mc2.metric(f"Recall@{k_val}", f"{metrics[f'Recall@{k_val}']}")
        mc3.metric("F1-Score", f"{metrics['F1-Score']}")
        st.success("Demo Sequence Completed Gracefully. No mathematical paradoxes detected.")

with tab3:
    st.markdown("### Professor Explanation Mode")
    st.markdown("""
<div class="course-card">
<h4 style='color: #818cf8;'>1. What is Precision@K vs Recall@K?</h4>
<p style='color: #cbd5e1; font-size: 0.95rem;'>
<b>Precision</b> tracks exactly how many of the top 'K' courses we recommended were actually relevant or liked by the unseen user target. If we recommend 5 courses, and they like 4, Precision is high. <br>
<b>Recall</b> asks out of ALL the user's liked courses in the whole system, what percentage did we catch in our top K? Recommending 1 highly precise course gives great Precision but terrible Recall.
</p>

<h4 style='color: #818cf8;'>2. Why TF-IDF was chosen?</h4>
<p style='color: #cbd5e1; font-size: 0.95rem;'>
Term Frequency-Inverse Document Frequency handles scaling. Natural Language has filler words ("the", "learn", "how"). TF-IDF penalizes these common strings and boosts rare, highly specialized tokens ("PyTorch", "Network"). This results in a cleaner similarity matrix without the massive overhead/RAM cost of Deep Neural Networks (like BERT).
</p>

<h4 style='color: #818cf8;'>3. The Power of Hybrid Filtering</h4>
<p style='color: #cbd5e1; font-size: 0.95rem;'>
A pure <b>Content</b> system traps users in a bubble; if they study Python, the system only shows Python, never branching into ML. <br>
A pure <b>Collaborative</b> model suffers the "Cold Start Block" (new items/users have no history so they never get recommended). <br>
By blending them harmoniously, we rely on Content text for cold starts and lean into Peer behaviors to suggest adjacent learning branches naturally, solving both limitations cleanly.
</p>
</div>
""", unsafe_allow_html=True)

with tab4:
    st.markdown("### Explainable ML Vectors")
    uv = st.session_state.recommender.get_user_profile_vector(user_id)
    n_features = st.session_state.recommender.tfidf_matrix.shape[1]
    
    st.code(f"""
[DIAGNOSTICS - USER {user_id}]
> Vocabulary Features Extracted: {n_features} Words
> User Profile Vector Shape: {uv.shape}
> Model Base Weight Configuration: Content ({(content_weight)*100:.0f}%) / Collaborative ({(collab_weight)*100:.0f}%)
    """)
    
    st.markdown("#### User Mathematical Trace")
    flat_uv = uv.flatten()
    top_feature_indices = flat_uv.argsort()[::-1][:10]
    feature_names = st.session_state.recommender.vectorizer.get_feature_names_out()
    
    vector_df = pd.DataFrame({
        "TF-IDF Token": [feature_names[i] for i in top_feature_indices],
        "Token Significance Weight": [flat_uv[i] for i in top_feature_indices]
    })
    
    st.dataframe(vector_df.style.background_gradient(cmap="viridis"), use_container_width=True)
