import logging
from fastapi import APIRouter, Depends
from app import schemas
from app.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Authentication"])


@router.get("/me", response_model=schemas.UserOut)
def get_me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user based on the Firebase token."""
    return current_user
