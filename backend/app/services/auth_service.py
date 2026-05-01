import logging
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app import models, schemas
from app.auth import hash_password, verify_password, create_access_token

logger = logging.getLogger(__name__)


def signup_user(db: Session, data: schemas.SignupRequest) -> models.User:
    """Register a new user, raising 400 if email already exists."""
    existing = db.query(models.User).filter(models.User.email == data.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists.",
        )
    user = models.User(
        name=data.name,
        email=data.email,
        password_hash=hash_password(data.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"New user registered: {user.email} (id={user.id})")
    return user


def login_user(db: Session, data: schemas.LoginRequest) -> dict:
    """Authenticate user and return JWT token + user data."""
    user = db.query(models.User).filter(models.User.email == data.email).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    token = create_access_token({"sub": str(user.id)})
    logger.info(f"User logged in: {user.email}")
    return {"token": token, "user": user}
