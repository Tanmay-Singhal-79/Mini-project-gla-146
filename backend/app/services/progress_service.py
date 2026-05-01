import logging
from datetime import datetime, timezone
from typing import List

from app import schemas

logger = logging.getLogger(__name__)


def update_progress(db, user: dict, step_title: str) -> dict:
    """Mark a learning step as completed. Idempotent."""
    user_id = user['id']
    step_title = step_title.strip()
    # Create a safe document ID: user_id + slugified title
    safe_title = "".join(c if c.isalnum() else "_" for c in step_title)
    progress_id = f"{user_id}_{safe_title}"
    progress_ref = db.collection('progress').document(progress_id)
    
    doc = progress_ref.get()
    now = datetime.now(timezone.utc)
    
    if doc.exists:
        data = doc.to_dict()
        if data.get('status') != "completed":
            progress_ref.update({
                "status": "completed",
                "completed_at": now
            })
            data['status'] = 'completed'
            data['completed_at'] = now
        data['id'] = progress_ref.id
        return data

    progress = {
        "user_id": user_id,
        "step_title": step_title,
        "status": "completed",
        "completed_at": now
    }
    progress_ref.set(progress)
    progress['id'] = progress_ref.id
    logger.info(f"User {user_id} completed step: '{step_title}' in Firestore")
    return progress


def get_progress(db, user: dict) -> schemas.ProgressResponse:
    """Return all completed steps and a percentage score."""
    user_id = user['id']
    query = db.collection('progress').where('user_id', '==', user_id).where('status', '==', 'completed').stream()
    
    completed = []
    for doc in query:
        data = doc.to_dict()
        data['id'] = doc.id
        completed.append(data)

    # Sort by completed_at descending (most recent first)
    completed.sort(key=lambda x: x.get('completed_at') or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    # Total steps across 5 domains × 100 steps each
    TOTAL_STEPS = 500
    percentage = round(min(len(completed) / TOTAL_STEPS * 100, 100), 1)

    return schemas.ProgressResponse(
        completedItems=[
            schemas.ProgressItem(
                id=p['id'],
                step_title=p['step_title'].strip(),
                status=p['status'],
                completed_at=p['completed_at']
            )
            for p in completed
        ],
        percentage=percentage,
    )


def toggle_favorite(db, user: dict, step_title: str) -> bool:
    """Toggle a step as a favorite. Returns True if now favorited, False if removed."""
    user_id = user['id']
    fav_id = f"{user_id}_{step_title.replace(' ', '_')}"
    fav_ref = db.collection('favorites').document(fav_id)
    
    doc = fav_ref.get()
    if doc.exists:
        fav_ref.delete()
        logger.info(f"User {user_id} removed favorite: '{step_title}'")
        return False
    else:
        fav_ref.set({
            "user_id": user_id,
            "step_title": step_title,
            "created_at": datetime.utcnow()
        })
        logger.info(f"User {user_id} added favorite: '{step_title}'")
        return True


def get_favorites(db, user: dict) -> List[dict]:
    """Get all favorite steps for a user."""
    user_id = user['id']
    query = db.collection('favorites').where('user_id', '==', user_id).stream()
    
    favorites = []
    for doc in query:
        data = doc.to_dict()
        data['id'] = doc.id
        favorites.append(data)
    
    return favorites
