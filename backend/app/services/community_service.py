import logging
from datetime import datetime
from fastapi import HTTPException, status
from app import schemas

logger = logging.getLogger(__name__)


def get_resources(db):
    """Return all resources ordered by upvotes descending."""
    docs = db.collection('resources').order_by('upvotes', direction='DESCENDING').stream()
    resources = []
    for doc in docs:
        data = doc.to_dict()
        data['id'] = doc.id
        resources.append(data)
    return resources


def add_resource(db, user: dict, title: str) -> dict:
    """Create a new community resource."""
    resource = {
        "title": title,
        "created_by": user['id'],
        "creator_name": user.get('name', 'Community Member'),
        "upvotes": 0,
        "comments_count": 0,
        "created_at": datetime.utcnow()
    }
    _, ref = db.collection('resources').add(resource)
    resource['id'] = ref.id
    logger.info(f"User {user['id']} added resource: '{title}'")
    return resource


def upvote_resource(db, resource_id: str) -> dict:
    """Increment upvote count for a resource."""
    ref = db.collection('resources').document(resource_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resource with id={resource_id} not found.",
        )
    
    # Firestore increment
    from google.cloud.firestore import Increment
    ref.update({"upvotes": Increment(1)})
    
    updated_doc = ref.get()
    data = updated_doc.to_dict()
    data['id'] = updated_doc.id
    return data


def get_comments(db, resource_id: str) -> list:
    """Return all comments for a specific resource."""
    docs = db.collection('resources').document(resource_id).collection('comments').order_by('created_at', direction='ASCENDING').stream()
    comments = []
    for doc in docs:
        data = doc.to_dict()
        data['id'] = doc.id
        comments.append(data)
    return comments


def add_comment(db, user: dict, resource_id: str, content: str) -> dict:
    """Add a new comment to a resource."""
    resource_ref = db.collection('resources').document(resource_id)
    if not resource_ref.get().exists:
        raise HTTPException(status_code=404, detail="Resource not found")

    comment = {
        "resource_id": resource_id,
        "content": content,
        "creator_id": user['id'],
        "creator_name": user.get('name', 'Community Member'),
        "created_at": datetime.utcnow()
    }
    
    # Add to subcollection
    _, ref = resource_ref.collection('comments').add(comment)
    comment['id'] = ref.id
    
    # Increment comments_count on resource
    from google.cloud.firestore import Increment
    resource_ref.update({"comments_count": Increment(1)})
    
    return comment
