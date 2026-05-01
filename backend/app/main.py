import os
import logging

# ─── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Triggering reload for new ML model

import firebase_admin
from firebase_admin import credentials

from app.routes import auth_routes, ai_routes, progress_routes, community_routes

# ─── Firebase Initialization ──────────────────────────────────────────────────
# Look for the service account key in the backend folder or from environment variable
cred_path = os.getenv("FIREBASE_CREDENTIALS", "firebase-credentials.json")
if not os.path.exists(cred_path):
    logger.warning(f"⚠️ Firebase credentials file '{cred_path}' not found! Authentication will fail.")
    logger.warning("Please download your service account JSON from Firebase Console and place it as backend/firebase-adminsdk.json")
else:
    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        logger.info("[✓] Firebase Admin SDK initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {e}")



# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="LearnPath AI API",
    description="AI-powered learning advisory platform backend",
    version="1.0.0",
)

# ─── CORS Middleware ──────────────────────────────────────────────────────────
# Using a more robust configuration to prevent preflight failures
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # More permissive for local development stability
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"GLOBAL CRASH: {exc}", exc_info=True)
    return {
        "detail": "Internal Server Error",
        "message": str(exc)
    }, 500

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(auth_routes.router)
app.include_router(ai_routes.router)
app.include_router(progress_routes.router)
app.include_router(community_routes.router)


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "app": "LearnPath AI API", "version": "1.0.0"}
