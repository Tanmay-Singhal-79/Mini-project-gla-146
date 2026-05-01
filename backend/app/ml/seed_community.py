import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
import random

# --- Configuration ---
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRED_PATH = os.path.join(BACKEND_DIR, "firebase-credentials.json")

# Initialize Firebase if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(CRED_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()

MOCK_RESOURCES = [
    {"title": "FreeCodeCamp: Full Stack Web Development Roadmap", "upvotes": 125, "created_by": "user_101"},
    {"title": "Awesome Machine Learning - Curated List of Resources", "upvotes": 89, "created_by": "user_102"},
    {"title": "Next.js 14 Official Documentation & Tutorial", "upvotes": 210, "created_by": "user_103"},
    {"title": "Tailwind CSS Component Gallery - Design Inspiration", "upvotes": 56, "created_by": "user_104"},
    {"title": "Python for Data Science - Hand-drawn Cheat Sheets", "upvotes": 142, "created_by": "user_105"},
    {"title": "OWASP Top 10 Security Risks - 2024 Guide", "upvotes": 77, "created_by": "user_106"},
    {"title": "Docker Mastery: Containerize Any Application", "upvotes": 64, "created_by": "user_107"},
    {"title": "React Query vs SWR: Choosing the Right Fetching Library", "upvotes": 45, "created_by": "user_108"},
    {"title": "Introduction to Neural Networks - 3Blue1Brown Series", "upvotes": 320, "created_by": "user_109"},
    {"title": "Clean Code Handbook: Principles of Agile Software Craft", "upvotes": 198, "created_by": "user_110"},
    {"title": "Cloud Architecture: AWS Well-Architected Framework", "upvotes": 34, "created_by": "user_111"},
    {"title": "Flutter vs React Native in 2024 - Performance Review", "upvotes": 52, "created_by": "user_112"},
    {"title": "Advanced SQL: Performance Tuning for PostgreSQL", "upvotes": 81, "created_by": "user_113"},
    {"title": "System Design Primer: Preparing for Tech Interviews", "upvotes": 275, "created_by": "user_114"},
    {"title": "Generative AI with LLMs - Stanford Online Course", "upvotes": 156, "created_by": "user_115"},
    {"title": "TypeScript Deep Dive - GitBook by Basarat", "upvotes": 93, "created_by": "user_116"},
    {"title": "Kubernetes: Up and Running - Interactive Playground", "upvotes": 41, "created_by": "user_117"},
    {"title": "Mastering the VIM Editor - Productivity Guide", "upvotes": 68, "created_by": "user_118"},
    {"title": "Ethical Hacking: The Complete Pentesting Course", "upvotes": 112, "created_by": "user_119"},
    {"title": "Zero to Mastery: Data Structures and Algorithms", "upvotes": 305, "created_by": "user_120"},
]

def seed_community():
    print(f"[*] Seeding {len(MOCK_RESOURCES)} community resources...")
    
    # Delete existing resources first (optional, but good for clean seed)
    # docs = db.collection('resources').stream()
    # for doc in docs:
    #     doc.reference.delete()
    
    batch = db.batch()
    for res in MOCK_RESOURCES:
        res_ref = db.collection('resources').document()
        # Randomize timestamp within the last 30 days
        days_ago = random.randint(0, 30)
        created_at = datetime.utcnow() - timedelta(days=days_ago)
        
        data = {
            "title": res["title"],
            "upvotes": res["upvotes"],
            "created_by": res["created_by"],
            "created_at": created_at
        }
        batch.set(res_ref, data)
    
    batch.commit()
    print("[OK] Community resources seeded successfully.")

if __name__ == "__main__":
    seed_community()
