# CogniPath: An AI-Assisted Learning Advisory Platform

## 1. Project Title
**CogniPath: An AI-Driven Adaptive Learning Advisory Platform using Hybrid Recommendation and Feedback Loops**

## 2. Abstract
The rapid expansion of e-learning platforms has democratized education but simultaneously introduced an overwhelming abundance of choices, leading to cognitive overload and sub-optimal learning trajectories. Currently, existing educational environments lack personalized, dynamic, and organized learning guidance. This project proposes **CogniPath**, an AI-assisted learning advisory platform designed to deliver personalized and adaptive educational pathways. By integrating a hybrid recommendation algorithm (combining Content-Based Filtering and Collaborative Filtering), the system analyzes user profiles, goals, and historic interactions to suggest highly relevant courses. Furthermore, CogniPath incorporates a real-time feedback-driven adaptation logic, ensuring the recommendation weights dynamically shift as a user grades courses. Progress tracking mechanisms provide continuous state assessment, cultivating a self-directed, goal-oriented learning environment. The platform features an explainability mode and full offline ML metric evaluation.

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
- **Opacity (Black Box):** Users do not know *why* specific courses are recommended.

## 5. Proposed System
CogniPath proposes a dynamic, intelligent system:
1. **User Profiling Engine:** Captures the learner's initial goal and skill baseline into arrays.
2. **Hybrid Recommendation Engine:** Uses machine learning text vectorization and matrix geometry to calculate overlapping knowledge spheres.
3. **Adaptive Feedback Loop:** Re-calculates user embedding vectors instantly based on post-course numeric ratings.
4. **Offline Evaluation Loop:** Incorporates strict train/test validation methodologies measuring algorithmic accuracy natively.

## 6. System Architecture Diagram

```text
[ User Interface (Streamlit) ]
        |       ^
 (Hyperparameters, Preferences, Feedback)
        v       |
[ API / Controller Layer (app.py) ]
        |---------------------------------------|
        v                                       v
[ Recommender Logic (model.py) ] <---- [ Evaluation & Data ]
   |-- Vectors Builder                   |-- Train/Test Split
   |-- Hybrid Aggregator                 |-- Interactions DB
        |
        v
[ Machine Learning Pipeline Execution ]
   |-- 1. TF-IDF Course Vectorization
   |-- 2. User Interest Mathematical Embedding
   |-- 3. Cosine Similarity Geometric Mapping
```

### 6.1 Architecture Explanation
- **User Interface Layer:** A robust Streamlit frontend collecting user targets and course ratings while projecting progress tracking modules.
- **API / Controller Layer:** Orchestrates data flow. Processes hyperparameter adjustments (K value, Model weights) into tensor re-calculations.
- **Recommender Engine:** The core application logic. Blends Collaborative (user-user) and Content-based (item-feature) models according to real-time adjustable parameter ratios.
- **ML Pipeline & Evaluation Layer:** Responsible for vectorized math operations and running Precision/Recall tests on dataset splits.

## 7. Tech Stack
- **Language:** Python 3.10+
- **Machine Learning Layer:** Scikit-learn (TF-IDF, Cosine Similarity Matrices, Evaluation functions)
- **Data Engineering:** Pandas, NumPy
- **Frontend / Fullstack:** Streamlit (Custom Dark CSS theme orchestration)

## 8. Dataset Explanation 
Due to institutional constraints, a highly controlled simulated dataset is configured:
- **Courses Dataset:** 20 granular courses tagged with categories, difficulty vectors, and textual description nodes.
- **Users Dataset:** Profiles mapped tightly to abstract target goals.
- **Interactions Dataset:** A user-item history matrix facilitating collaborative filtering methodologies.

## 9. Machine Learning Pipeline & Vectorization
CogniPath employs a scalable vectorization pipeline:
1. **TF-IDF Course Vectorization:** Translates textual data (course abstracts, difficulty, tags) into numerical feature matrices. TF-IDF punishes generic words (like "the" or "learn") and isolates high-weight technical vocabulary (like "PyTorch" or "Neural").
2. **User Interest Embedding:** Instead of analyzing a user as a static text profile point, users are represented dynamically by aggregating the geometric TF-IDF vectors of the courses they've explicitly rated positively. 
3. **Cosine Similarity Computation:** Calculates the geometric angular deviation mapped between user vectors and all available course vectors to produce similarity scores between `0.0` and `1.0`.

## 10. Hyperparameter Tuning
The system architecture directly embeds UI logic to adjust configuration parameters, offering immense analytical flexibility:
- **K-Value Parameter:** Controls `Precision@K` retrieval ceilings. Isolating top-3 vs top-10 yields drastically different semantic focuses.
- **Weight Threshold Balancing:** Enables the engine to drift elegantly from `100% Collaborative Filtering` (crowdsourced peer behaviors) over to `100% Content Filtering` (rigid topic text analysis) depending on system state and dataset sparsity.

## 11. Model Evaluation & Metrics
To systematically measure algorithmic quality computationally, the `interactions_df` is passed through a reproducible `80/20` Train/Test simulation.
- **Precision@K:** Measures the proportion of recommended items in the top `K` predicted output that the user actually historically evaluated positively. 
- **Recall@K:** Determines the fraction of positively associated testing items that successfully appeared inside the recommendation set. 
- **F1-Score:** The harmonic integration of Precision and Recall, producing a structurally balanced health check of the algorithm preventing bias scenarios (e.g., retrieving everything to cheat Recall to 100%).
- **RMSE (Root Mean Square Error):** Investigates the severity of numerical deviation between the model's mapped prediction strength and real recorded input scores in the holdout set.

## 12. Mathematical Intuition
**Cosine Similarity Expression:**
`Sim(A, B) = (A · B) / (||A|| * ||B||)`
Provides the projection strength of Vector B intersecting Vector A. An angle of 0° outputs 1.0 similarity (perfect thematic match).

**F1-Score Formula:**
`F1 = 2 * (Precision * Recall) / (Precision + Recall)`

## 13. Implementation Code Structure
To emulate modular enterprise standards, repository logic is strictly isolated:
- `model.py`: Domain host for the mathematical `HybridRecommender` and array construction logic. 
- `evaluation.py`: Mathematical layer executing the dataset chunking and prediction analysis metrics.
- `app.py`: Top-level Controller logic managing UI rendering sequences and state synchronization.
- `utils.py`: Design injections, CSS compilation, and string utilities.
- `data.py`: Bootstrap generation layer for DataFrame objects.

## 14. Sample Output & "Explainable AI Mode"
Recognizing the risks of "Black Box" AI in educational technology, CogniPath institutes a foundational explainability layer:
- **Context Prints:** Every predicted recommendation forces a real-time logical trace directly in the UI (e.g., *"Content Match: Concept geometry matched your history with 0.84 similarity."*).
- **ML Demonstration Mode:** An integrated backend diagnostic UI exposes active variable geometries, plotting the User arrays (`1x314` dimension vectors, specifically detailing active tokens like "python" or "matrix") to prove the application is mathematically sound and not functioning on randomization.

## 15. Conclusion
CogniPath comprehensively verifies the viability of complex hybrid recommendation chains applied to unstructured educational nodes. By mathematically translating qualitative preferences into geometric coordinates, while verifying performance against stringent analytical criteria (F1/RMSE metrics), the ecosystem serves as a highly resilient and robust digital advisory framework.

## 16. References
[1] P. Brusilovsky, "Adaptive hypermedia: From intelligent tutoring systems to Web-based education," *Int. J. Artif. Intell. Educ.*, vol. 18, pp. 87-96, 2008.
[2] J. Bobadilla et al., "Recommender systems survey: Challenges, evolution and expands," *Knowledge-Based Systems*, 2013.
[3] F. Maxwell Harper and Joseph A. Konstan, "The MovieLens Datasets: History and Context," *ACM Transactions on Interactive Intelligent Systems (TiiS)*, 2015.
