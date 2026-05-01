import os
import random
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def populate():
    # Initialize Firebase
    cred_path = os.getenv("FIREBASE_CREDENTIALS", "firebase-credentials.json")
    if not os.path.exists(cred_path):
        # Try backend folder
        alternate_path = os.path.join("backend", cred_path)
        if os.path.exists(alternate_path):
            cred_path = alternate_path
        else:
            print(f"Error: {cred_path} not found.")
            return

    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    # Clear existing resources
    docs = db.collection('resources').stream()
    for doc in docs:
        doc.reference.delete()
    print("Cleared existing resources.")

    resources = [
        {"title": "Mastering React Hooks - A Comprehensive Guide", "creator": "Sarah J.", "upvotes": 124},
        {"title": "Python for Data Science: From Scratch to Advanced", "creator": "David Chen", "upvotes": 89},
        {"title": "The Ultimate CSS Grid Layout Tutorial", "creator": "Emma Wilson", "upvotes": 56},
        {"title": "Understanding Transformers in Machine Learning", "creator": "Alex Rivera", "upvotes": 210},
        {"title": "Building Scalable Microservices with Go", "creator": "Michael Scott", "upvotes": 45},
        {"title": "Design Patterns in JavaScript - Clean Code Principles", "creator": "Jessica Lee", "upvotes": 132},
        {"title": "Docker & Kubernetes: The Practical Guide", "creator": "Robert Brown", "upvotes": 78},
        {"title": "Next.js 14 Server Components Deep Dive", "creator": "Kevin Zhang", "upvotes": 167},
        {"title": "Modern UI/UX Trends for 2024", "creator": "Chloe Adams", "upvotes": 93},
        {"title": "Introduction to Rust for C++ Developers", "creator": "Tom Harris", "upvotes": 112},
        {"title": "SQL Performance Tuning for Web Developers", "creator": "Maria Garcia", "upvotes": 34},
        {"title": "OAuth2 and OpenID Connect Explained", "creator": "Ryan Gosling", "upvotes": 88},
        {"title": "Building Real-time Apps with WebSockets", "creator": "Justin Bieber", "upvotes": 21},
        {"title": "Functional Programming in TypeScript", "creator": "Linus Torvalds", "upvotes": 245},
        {"title": "AWS Lambda & Serverless Best Practices", "creator": "Jeff Bezos", "upvotes": 56},
    ]

    print(f"Populating {len(resources)} resources...")

    for res in resources:
        # Random date within last 30 days
        days_ago = random.randint(0, 30)
        created_at = datetime.utcnow() - timedelta(days=days_ago)
        
        doc_data = {
            "title": res["title"],
            "creator_name": res["creator"],
            "upvotes": res["upvotes"],
            "comments_count": random.randint(0, 15),
            "created_at": created_at,
            "created_by": "system_gen"
        }
        
        db.collection('resources').add(doc_data)
        print(f"Added: {res['title']}")

    print("Successfully populated community resources!")

if __name__ == "__main__":
    populate()
