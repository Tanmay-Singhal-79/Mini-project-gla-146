import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException

from app import schemas
from app.database import get_db
from app.auth import get_current_user
from app.services import progress_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Progress"])


@router.get("/progress", response_model=schemas.ProgressResponse)
def get_progress(
    db = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    try:
        return progress_service.get_progress(db, current_user)
    except Exception as e:
        logger.error(f"Error fetching progress for user {current_user['id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch progress.")


@router.post("/progress", response_model=schemas.ProgressItem)
def update_progress(
    body: schemas.ProgressUpdateRequest,
    db = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    try:
        if not body.step_title or not body.step_title.strip():
            raise HTTPException(status_code=422, detail="step_title cannot be empty.")
        return progress_service.update_progress(db, current_user, body.step_title.strip())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating progress for user {current_user['id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update progress.")


@router.get("/favorites", response_model=List[schemas.FavoriteItem])
def get_favorites(
    db = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    try:
        return progress_service.get_favorites(db, current_user)
    except Exception as e:
        logger.error(f"Error fetching favorites for user {current_user['id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch favorites.")


@router.post("/favorites")
def toggle_favorite(
    body: schemas.FavoriteUpdateRequest,
    db = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    try:
        if not body.step_title or not body.step_title.strip():
            raise HTTPException(status_code=422, detail="step_title cannot be empty.")
        is_favorited = progress_service.toggle_favorite(db, current_user, body.step_title.strip())
        return {"step_title": body.step_title, "is_favorited": is_favorited}
    except Exception as e:
        logger.error(f"Error toggling favorite for user {current_user['id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle favorite.")
