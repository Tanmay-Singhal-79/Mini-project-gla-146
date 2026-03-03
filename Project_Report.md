# AI-Assisted Learning Advisory Platform

## 1. Project Title
**CogniPath: A Content-Based Course Recommendation Engine using TF-IDF and Cosine Similarity**

## 2. Abstract
The rapid growth of e-learning platforms has created an overwhelming abundance of educational content, making it difficult for students to find courses that align with their specific goals. Existing systems often rely on generic popularity rather than analyzing the actual content of the courses. This project proposes an AI-assisted learning platform that provides personalized course recommendations. Using a Content-Based Filtering approach, the system vectorizes course descriptions and student interests using Natural Language Processing (TF-IDF). It then calculates the Cosine Similarity to recommend courses mathematically closest to the user's goals. The system also tracks progress and allows users to rate courses, dynamically shifting future recommendations as they learn. It is built entirely in Python using Streamlit for a clean, modern user interface.

## 3. Problem Statement
"To design a simple, content-driven AI learning platform that recommends personalized courses to students based on their technical interests and intelligently verified past ratings, solving the problem of unstructured learning paths."

## 4. Existing System
- Most basic educational platforms suggest courses manually categorized by administrators.
- Static lists do not adapt over time.
- Often missing a mathematical way to check if two courses actually share the same internal topic focus.

## 5. Proposed System
We propose an intelligent, logic-driven advisory system:
1. Users select their primary domain of interest (e.g., Data Science).
2. The AI reads all available course descriptions and tags using TF-IDF.
3. It recommends courses most similar to the user's interest using Cosine Geometry.
4. When a user rates a course highly (4+), the system adds that course's text features to the user's hidden "Interest Profile," making future recommendations more accurate.

## 6. System Architecture Diagram
```text
[ User Interface (Streamlit) ]
      |
 (Selects Interest & Rates Courses)
      |
      v
[ Content-Based Recommender ]
      |-- TF-IDF Vectorizer (Reads text)
      |-- Cosine Similarity (Finds matches)
      |
      v
[ Course Database (Mock Data) ]
```

## 7. Tech Stack
- **Language:** Python
- **Machine Learning:** Scikit-learn (TF-IDF, Cosine Similarity)
- **Data Handling:** Pandas, NumPy
- **Frontend App:** Streamlit
