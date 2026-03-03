# CogniPath: An AI-Assisted Learning Advisory Platform

## 1. Project Title
**CogniPath: An AI-Driven Adaptive Learning Advisory Platform using Hybrid Recommendation and Feedback Loops**

## 2. Abstract
The rapid expansion of e-learning platforms has democratized education but simultaneously introduced an overwhelming abundance of choices, leading to cognitive overload and sub-optimal learning trajectories. Currently, existing educational environments lack personalized, dynamic, and organized learning guidance, relying heavily on static curricula or generic popularity-based suggestions. This project proposes **CogniPath**, an AI-assisted learning advisory platform designed to deliver personalized and adaptive educational pathways. By integrating a hybrid recommendation algorithm (combining Content-Based Filtering and Collaborative Filtering), the system analyzes user profiles, goals, and historic interactions to suggest highly relevant courses. Furthermore, CogniPath incorporates a real-time feedback-driven adaptation logic, ensuring the recommendation weights dynamically shift as a user grades courses or modifies their target learning goals. Progress tracking mechanisms provide continuous state assessment, cultivating a self-directed, goal-oriented learning environment. The platform is deployed via a modern, scalable web architecture built using Python, Scikit-learn, and Streamlit, resulting in an enterprise-grade demonstration of AI in EdTech.

## 3. Problem Definition
Learners on modern digital platforms face the "Paradox of Choice." Without structured tutoring, individuals struggle to piece together coherent learning paths from thousands of fragmented courses. 
**Core Problem Statement:** "To develop an AI-assisted learning advisory platform that addresses the lack of personalized and organized learning guidance in existing educational environments through recommendation algorithms, community knowledge sharing, and feedback-driven adaptation."

## 4. Existing System & Limitations
**Existing Systems:**
- Standard MOOC platforms (Coursera, Udemy) primarily rely on global popularity, simple tag-based filtering, or collaborative filtering alone.
- Curricula are static and linear (e.g., standard degree tracks).

**Limitations:**
- **Static Pathways:** Learning paths do not adapt if a student finds a topic too difficult or too easy.
- **Cold Start Problem:** New users face a steep curve to find relevant content.
- **Lack of Feedback Integration:** Course ratings rarely immediately adapt the user's localized sequential path.

## 5. Proposed System
CogniPath proposes a dynamic, intelligent system:
1. **User Profiling Engine:** Captures the learner's initial goal (e.g., Data Scientist) and skill baseline.
2. **Hybrid Recommendation Engine:** Uses NLP (TF-IDF) on course descriptions for content similarity, and user-item matrix cosine similarity for collaborative filtering.
3. **Adaptive Feedback Loop:** Re-calculates user embedding vectors instantly based on post-course ratings (1-5 scale).
4. **Progress Tracker:** Visualizes completed tracks versus master goals to gamify and encourage completion.

## 6. System Architecture Diagram

```text
[ User Interface (Streamlit) ]
        |       ^
 (Input Goals)  (Interactive Dash / Progress)
        v       |
[ API / Controller Layer (app.py) ]
        |--------------------------|
        v                          v
[ Recommender Engine ] <---- [ Data & State Manager ]
   |-- Content Filter             |-- User Profiles (data.py)
   |-- Collab. Filter             |-- Course DB (Mock Data)
   |-- Hybrid Aggregator          |-- Interaction Logs
        |
        v
[ Machine Learning Models ]
   |-- TF-IDF Vectorizer
   |-- Cosine Similarity Matrix
```

*Explanation:* The user interacts with the UI, inputting their goals and rating courses. This state updates the Data Manager. The Recommender Engine queries the Data Manager, processes the state through its TF-IDF and Cosine Similarity matrices, and pushes tailored recommendations back to the UI.

## 7. Tech Stack
- **Language:** Python 3.10+
- **Machine Learning / Math:** Scikit-learn, Pandas, NumPy
- **Frontend / Fullstack:** Streamlit (UI Framework with custom CSS orchestration)
- **Deployment Topology:** Monolithic prototype (expandable to microservices)

## 8. Dataset Explanation 
Due to institutional constraints, a highly controlled simulated dataset is used to map specific edge-cases:
- **Courses Dataset:** 20 granular courses tagged with categories (Data Science, Cloud, Web Dev), difficulty levels, and textual descriptions.
- **Users Dataset:** Profiles defined by unique target goals.
- **Interactions Dataset:** A user-item matrix containing boolean completion status and numeric ratings (1-5), facilitating collaborative filtering.

## 9. Machine Learning Model Used
**1. Content-Based Filtering:**
- Algorithm: TF-IDF (Term Frequency-Inverse Document Frequency)
- Usage: Converts course descriptions and tags into vector space. It finds courses mathematically similar to what the user explicitly likes.

**2. Collaborative Filtering:**
- Algorithm: User-User Collaborative Filtering (Cosine Similarity)
- Usage: Identifies users with similar historical rating patterns and recommends courses highly rated by peers ("community knowledge sharing").

**3. Feedback Loop Adaptation:**
- Logic: When a rating occurs, the system's `interactions_df` receives an immediate mutation. Subsequent calls to `get_hybrid_recommendations` re-factor the user's vector average.

## 10. Mathematical Intuition
**TF-IDF:**
`W(t, d) = TF(t, d) * log(N / DF(t))`
Where `t` is a keyword (e.g., "Neural Networks"), `d` is the course, `N` is total courses. This punishes generic words and highlights unique technical terms.

**Cosine Similarity:**
`Sim(A, B) = (A · B) / (||A|| * ||B||)`
Calculates the cosine of the angle between two mathematical vectors (representing either course texts or user rating histories). A similarity of 1 means exactly identical context.

*(Implementation details mapped in code artifacts `recommender.py` and `app.py`)*

## 13. Sample Output Demonstration
1. **Welcome Dashboard:** User sees current progress (e.g., "Completed: 3, Avg Score: 4.2").
2. **AI Engine:** Based on user rating 5 on "Intro to Python", the TF-IDF engine highly ranks "Advanced Python" and user-user engine pairs "Data Science 101".
3. **Feedback Update:** The user clicks "Complete & Submit Feedback (2/5)" on a DevOps course. The system instantly realizes the user dislikes DevOps, altering the hybrid recommendation matrix to prioritize Data Science over Cloud infrastructure in the next render.

## 14. Future Enhancements
- **Knowledge Graphs:** Integrating Neo4j to map prerequisite chains (e.g., Python -> ML -> Deep Learning).
- **LLM Integration:** Using OpenAI/Llama-2 to generate a conversational tutor that explains *why* a course was recommended.
- **Automated Assessments:** Replacing subjective 1-5 ratings with objective post-course quiz scores to adjust the feedback loop.

## 15. Conclusion
The proposed CogniPath platform successfully demonstrates a proof-of-concept AI-assisted learning environment. By combining content logic with community behavioral filtering, it mitigates static-curriculum limitations. The integrated feedback loops ensure a scalable, adaptable system that evolves organically alongside the learner's journey.

## 16. References
[1] P. Brusilovsky, "Adaptive hypermedia: From intelligent tutoring systems to Web-based education," *Int. J. Artif. Intell. Educ.*, vol. 18, no. 1, pp. 87-96, 2008.
[2] J. Bobadilla et al., "Recommender systems survey: Challenges, evolution and expands," *Knowledge-Based Systems*, 2013.
[3] IEEE Xplore Digital Recommendations in Educational Tech.
