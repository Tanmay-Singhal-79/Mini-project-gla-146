import pandas as pd
import numpy as np

def generate_mock_data():
    """Generates a mock dataset for the CogniPath recommendation engine."""
    np.random.seed(42)
    
    # 1. Courses Dataset
    courses = {
        'course_id': range(1, 21),
        'title': [
            'Intro to Python', 'Advanced Python', 'Data Science 101', 'Machine Learning Basics',
            'Deep Learning with PyTorch', 'NLP Fundamentals', 'Computer Vision Pro', 
            'Web Dev with React', 'Backend with Node.js', 'Cloud Computing AWS',
            'DevOps Essentials', 'Kubernetes Mastery', 'Data Engineering', 'SQL for Beginners',
            'Advanced SQL', 'Cybersecurity 101', 'Ethical Hacking', 'UI/UX Design',
            'Agile Project Management', 'Product Management'
        ],
        'category': [
            'Programming', 'Programming', 'Data Science', 'Data Science',
            'Data Science', 'Data Science', 'Data Science',
            'Web Dev', 'Web Dev', 'Cloud',
            'DevOps', 'DevOps', 'Data Science', 'Database',
            'Database', 'Security', 'Security', 'Design',
            'Management', 'Management'
        ],
        'difficulty': [
            'Beginner', 'Advanced', 'Beginner', 'Intermediate',
            'Advanced', 'Intermediate', 'Advanced',
            'Beginner', 'Intermediate', 'Intermediate',
            'Beginner', 'Advanced', 'Intermediate', 'Beginner',
            'Advanced', 'Beginner', 'Intermediate', 'Beginner',
            'Beginner', 'Intermediate'
        ],
        'description': [
            'Learn the basics of Python programming language.',
            'Master advanced Python concepts like decorators and generators.',
            'Introduction to data manipulation with pandas and numpy.',
            'Learn fundamental ML algorithms and Scikit-learn.',
            'Build neural networks using PyTorch framework.',
            'Text processing, sentiment analysis, and basic transformers.',
            'Image classification and object detection with CNNs.',
            'Build modern frontend applications with React.',
            'Develop robust backend APIs using Node.js and Express.',
            'Deploy architecture on Amazon Web Services.',
            'CI/CD pipelines, Docker, and deployment basics.',
            'Container orchestration and microservices with K8s.',
            'ETL pipelines and big data handling.',
            'Relational database concepts and basic queries.',
            'Complex joins, window functions, and optimization.',
            'Foundations of network security and cryptography.',
            'Penetration testing and vulnerability assessment.',
            'User research, wireframing, and Figma basics.',
            'Scrum, Kanban, and agile methodologies.',
            'Product lifecycle, roadmapping, and user stories.'
        ]
    }
    df_courses = pd.DataFrame(courses)

    # 2. Users Profile Dataset
    users = {
        'user_id': range(1, 11),
        'name': [f'User {i}' for i in range(1, 11)],
        'target_goal': np.random.choice(['Data Scientist', 'Web Developer', 'DevOps Engineer', 'Security Analyst'], 10)
    }
    df_users = pd.DataFrame(users)

    # 3. Interactions Dataset (User-Course matrix for Collaborative Filtering)
    interactions = []
    for u in users['user_id']:
        # Each user has completed and rated 3-5 courses
        num_rated = np.random.randint(3, 6)
        rated_courses = np.random.choice(courses['course_id'], num_rated, replace=False)
        for c in rated_courses:
            interactions.append({
                'user_id': u,
                'course_id': c,
                'rating': np.random.randint(3, 6), # mostly positive ratings (3-5) for initial history
                'completed': np.random.choice([True])
            })
    df_interactions = pd.DataFrame(interactions)

    return df_courses, df_users, df_interactions

if __name__ == "__main__":
    c_df, u_df, i_df = generate_mock_data()
    print("Generated mock dataset successfully.")
    print(f"Courses: {len(c_df)}, Users: {len(u_df)}, Interactions: {len(i_df)}")
