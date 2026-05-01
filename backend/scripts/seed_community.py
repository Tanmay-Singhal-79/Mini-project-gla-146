import os
import sys
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Add backend to path to use existing setup
sys.path.append(os.path.join(os.getcwd(), 'backend'))

def seed_community():
    print("[*] Starting Neural Community Seed (Bharat Edition)...")
    
    # Initialize Firebase if not already
    cred_path = os.path.join(os.getcwd(), 'backend', 'firebase-credentials.json')
    if not os.path.exists(cred_path):
        print(f"[✗] Firebase credentials not found at {cred_path}")
        return

    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    
    # Realistic Indian Tech Resources
    resources = [
        {
            "title": "Cracking the 2024 SWE Placements: My Journey to Google India (Bangalore)",
            "upvotes": 342,
            "comments_count": 15,
            "creator_name": "Ananya_IITD",
            "link": "https://ananyasharma.dev/placements",
            "created_at": datetime.utcnow()
        },
        {
            "title": "Optimizing Razorpay Webhook Reliability for Indian Fintech Apps",
            "upvotes": 215,
            "comments_count": 8,
            "creator_name": "Rohan_Zomato_SDE",
            "link": "https://engineering.zomato.com",
            "created_at": datetime.utcnow()
        },
        {
            "title": "Next.js vs Flutter: What should Indian Startups choose for MVP in 2024?",
            "upvotes": 187,
            "comments_count": 22,
            "creator_name": "Kavya_Nair_Architect",
            "link": "https://kavyanair.com/blog",
            "created_at": datetime.utcnow()
        },
        {
            "title": "Low-Latency System Design for a Real-time Cricket Score App (IPL Edition)",
            "upvotes": 456,
            "comments_count": 31,
            "creator_name": "CricCoder_99",
            "link": "https://github.com/criccoder/ipl-realtime",
            "created_at": datetime.utcnow()
        }
    ]

    for res in resources:
        try:
            # Check if it already exists by title
            docs = db.collection('resources').where('title', '==', res['title']).get()
            if not docs:
                doc_ref = db.collection('resources').add(res)
                print(f"[✓] Seeded: {res['title']}")
                
                # Add some comments for the first few
                if res['creator_name'] == "Ananya_IITD":
                    db.collection('comments').add({
                        "resource_id": doc_ref[1].id,
                        "content": "Did you focus more on LeetCode Medium or Hard for the Bangalore office rounds?",
                        "creator_name": "Aman_NIT_Trichy",
                        "created_at": datetime.utcnow()
                    })
                    db.collection('comments').add({
                        "resource_id": doc_ref[1].id,
                        "content": "Mostly Medium, but make sure your System Design (LLD) is super strong.",
                        "creator_name": "Ananya_IITD",
                        "created_at": datetime.utcnow()
                    })
            else:
                print(f"[-] Already exists: {res['title']}")
        except Exception as e:
            print(f"[✗] Failed to seed {res['title']}: {e}")

    print("[✓] Community Seeding Complete.")

if __name__ == "__main__":
    seed_community()
