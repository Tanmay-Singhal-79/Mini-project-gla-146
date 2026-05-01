import os
import logging
from typing import Optional
from datetime import datetime

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials

from app.database import get_db

load_dotenv()

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer()

def get_current_user(
    credentials_header: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db = Depends(get_db),
) -> dict:
    """FastAPI dependency to get the currently authenticated user from Firebase JWT."""
    token = credentials_header.credentials

    import time
    start_auth = time.time()
    try:
        # Verify the Firebase ID token
        decoded_token = firebase_auth.verify_id_token(token)
        logger.info(f"[Perf] Token verification took: {time.time() - start_auth:.4f}s")
    except Exception as e:
        logger.warning(f"Invalid or expired Firebase token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    uid = decoded_token.get("uid")
    email = decoded_token.get("email")
    name = decoded_token.get("name", "User") # Sometimes name isn't set depending on provider

    if not uid or not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload is missing user identity.",
        )

    # Use Firestore
    db_fetch_start = time.time()
    user_ref = db.collection('users').document(uid)
    user_doc = user_ref.get()
    logger.info(f"[Perf] Auth User Doc Fetch took: {time.time() - db_fetch_start:.4f}s")

    if user_doc.exists:
        user_data = user_doc.to_dict()
        user_data['id'] = uid
        return user_data
    else:
        # Auto-create user in Firestore
        create_start = time.time()
        new_user = {
            "name": name,
            "email": email,
            "created_at": datetime.utcnow()
        }
        user_ref.set(new_user)
        new_user['id'] = uid
        logger.info(f"[Perf] Auth User Auto-create took: {time.time() - create_start:.4f}s")
        logger.info(f"Auto-created new user from Firebase token: {email}")
        return new_user
