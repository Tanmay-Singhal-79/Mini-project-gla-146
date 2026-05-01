import os
from firebase_admin import firestore
from dotenv import load_dotenv

load_dotenv()

def get_db():
    """Dependency that provides a Firestore database client."""
    db = firestore.client()
    try:
        yield db
    finally:
        pass
