import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from fastapi import APIRouter, Depends, HTTPException

from app import schemas
from app.database import get_db
from app.auth import get_current_user
from app.services import community_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Community"])


@router.get("/resource", response_model=List[schemas.ResourceOut])
def get_resources(
    db = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    try:
        return community_service.get_resources(db)
    except Exception as e:
        logger.error(f"Error fetching resources: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch resources.")


@router.post("/resource", response_model=schemas.ResourceOut, status_code=201)
def add_resource(
    body: schemas.AddResourceRequest,
    db = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    try:
        if not body.title or not body.title.strip():
            raise HTTPException(status_code=422, detail="Title cannot be empty.")
        return community_service.add_resource(db, current_user, body.title.strip())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding resource: {e}")
        raise HTTPException(status_code=500, detail="Failed to add resource.")


@router.post("/upvote", response_model=schemas.ResourceOut)
def upvote(
    body: schemas.UpvoteRequest,
    db = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    try:
        return community_service.upvote_resource(db, body.resource_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error upvoting resource {body.resource_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to upvote resource.")


@router.get("/resource/{resource_id}/comments", response_model=List[schemas.CommentOut])
def get_comments(
    resource_id: str,
    db = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    try:
        return community_service.get_comments(db, resource_id)
    except Exception as e:
        logger.error(f"Error fetching comments for {resource_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch comments.")


@router.post("/resource/{resource_id}/comments", response_model=schemas.CommentOut, status_code=201)
def add_comment(
    resource_id: str,
    body: schemas.AddCommentRequest,
    db = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    try:
        if not body.content or not body.content.strip():
            raise HTTPException(status_code=422, detail="Comment cannot be empty.")
        return community_service.add_comment(db, current_user, resource_id, body.content.strip())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding comment to {resource_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to add comment.")
