import pandas as pd

def get_data():
    courses = [
        {'id': 1, 'title': 'Python Basics', 'tags': 'Python, Programming, Beginner', 'desc': 'Learn variables, loops, and functions.'},
        {'id': 2, 'title': 'Machine Learning 101', 'tags': 'ML, Python, Data Science', 'desc': 'Learn linear regression and basic classification.'},
        {'id': 3, 'title': 'Deep Learning', 'tags': 'Deep Learning, AI, Neural Networks', 'desc': 'Advanced neural networks with PyTorch.'},
        {'id': 4, 'title': 'Web Development', 'tags': 'HTML, CSS, Web', 'desc': 'Build responsive frontend websites.'},
        {'id': 5, 'title': 'React Fundamentals', 'tags': 'React, JavaScript, Web', 'desc': 'Modern frontend state management.'},
        {'id': 6, 'title': 'Cyber Security 101', 'tags': 'Security, Network, Linux', 'desc': 'Learn ethical hacking and network defense.'},
        {'id': 7, 'title': 'Advanced Python', 'tags': 'Python, Programming, Advanced', 'desc': 'Object oriented programming and decorators.'},
        {'id': 8, 'title': 'Data Analysis', 'tags': 'Pandas, Data Science, Python', 'desc': 'Data cleaning and basic analytics.'},
    ]
    return pd.DataFrame(courses)

def get_users():
    return pd.DataFrame([
        {'id': 1, 'name': 'Alice', 'interest': 'Data Science'},
        {'id': 2, 'name': 'Bob', 'interest': 'Web Development'},
        {'id': 3, 'name': 'Charlie', 'interest': 'Security'}
    ])
