import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

try:
    from app.services import recommendation_service
    print("Model loaded successfully.")
    path = recommendation_service.generate_learning_path("Web Development")
    print(f"Generated path with {len(path)} steps.")
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
