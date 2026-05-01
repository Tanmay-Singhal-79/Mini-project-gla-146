import os
import sys
# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from backend.app.services import recommendation_service

path = recommendation_service.generate_learning_path("Web Development")
print(f"Generated {len(path)} steps.")
if len(path) > 0:
    print(f"First step: {path[0]['title']}")
    print(f"Description: {path[0].get('description')}")
