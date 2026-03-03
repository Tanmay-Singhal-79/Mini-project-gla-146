# CogniPath AI - Content-Based Course Recommender Engine

CogniPath is a minimalist, content-driven AI learning platform designed to recommend personalized courses to students. It solves the problem of unstructured learning paths by mathematically correlating user interests with available course material.

## Features
- **Pure Content-Based Filtering**: The core engine strictly uses analyzing text without relying on other user data.
- **TF-IDF Vectorization**: Reads course descriptions and tags to form numeric mathematical arrays.
- **Cosine Similarity Math**: Analyzes the geometric angles between the student's text interests and courses to identify precise matches.
- **Dynamic Feedback**: If a student rates a course highly (4 or 5 stars), the internal mathematical wording from that course is permanently appended to their future searches. 
- **Modern Minimal UI**: Encased seamlessly in a sleek, scalable Streamlit interface utilizing a custom dark theme.

## Architecture
The application is purely local and stateless. 
1. `app.py`: Contains the Streamlit controller components, frontend CSS styling logic, and user input capture mechanisms.
2. `model.py`: Houses the Scikit-Learn instances (TF-IDF mapping arrays and cosine similarity engine). 
3. `data.py`: A simple mock DataFrame simulation handling the course repository elements.

## Installation & Setup

Ensure you have Python 3.10+ installed on your system.

**1. Clone the repository and navigate inside the folder**
```sh
cd mini-project-146
```

**2. Install dependencies mapping the requirements file**
```sh
pip install -r requirements.txt
```

**3. Launch the Application**
```sh
python -m streamlit run app.py
```

*Note: Streamlit serves natively at `localhost:8501`. Navigate to this URL in any standard web browser if it does not automatically open.* 

## Built For
This implementation is designed as a streamlined, hyper-focused Machine Learning project artifact suitable for second or third-year collegiate defense, prioritizing stability and core mathematical logic validation over vast complex system architectures.
