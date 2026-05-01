import logging
from fastapi import APIRouter, Depends, HTTPException
from app import schemas
from app.database import get_db
from app.auth import get_current_user
from app.services import recommendation_service, lesson_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["AI Recommendations"])


@router.post("/recommend", response_model=schemas.RecommendResponse)
def recommend(
    body: schemas.RecommendRequest,
    current_user: dict = Depends(get_current_user),
):
    """Returns ML-based topic recommendations for a given interest string."""
    try:
        if not body.interest or not body.interest.strip():
            raise HTTPException(status_code=422, detail="Interest field cannot be empty.")

        topics = recommendation_service.recommend_topics(body.interest.strip())
        return schemas.RecommendResponse(topics=topics)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /recommend: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations.")


@router.post("/generate-path", response_model=schemas.GeneratePathResponse)
def generate_path(
    body: schemas.RecommendRequest,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
):
    """Generates a personalized, sequential learning roadmap from the ML model."""
    import time
    start_time = time.time()
    try:
        interest = body.interest.strip() if body.interest else "Web Development"
        logger.info(f"[Perf] Starting generation for: {interest}")
        
        path_start = time.time()
        steps = recommendation_service.generate_learning_path(interest)
        logger.info(f"[Perf] AI Path Generation took: {time.time() - path_start:.4f}s")
        
        if not steps:
             logger.warning(f"No steps generated for interest: {interest}")
        
        # Save active roadmap interest to user profile - wrap in safety
        db_start = time.time()
        try:
            user_id = current_user.get('id')
            if user_id:
                db.collection('users').document(user_id).update({
                    "current_interest": interest
                })
                logger.info(f"[Perf] Firestore update took: {time.time() - db_start:.4f}s")
        except Exception as db_err:
            logger.error(f"Non-critical Firestore update failed: {db_err}")
        
        logger.info(f"[Perf] Total request time: {time.time() - start_time:.4f}s")
        return schemas.GeneratePathResponse(steps=steps)

    except Exception as e:
        logger.error(f"Critical error in /generate-path: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate learning path: {str(e)}")


@router.get("/path/{interest}", response_model=schemas.GeneratePathResponse)
def get_path(
    interest: str,
    current_user: dict = Depends(get_current_user),
):
    """Retrieves a generated learning path for a specific interest."""
    try:
        steps = recommendation_service.generate_learning_path(interest)
        return schemas.GeneratePathResponse(steps=steps)
    except Exception as e:
        logger.error(f"Error in GET /path: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve learning path.")


@router.get("/lesson/{step_title:path}", response_model=schemas.LessonResponse)
def get_lesson(
    step_title: str,
    current_user: dict = Depends(get_current_user),
):
    """Returns structured W3Schools-style lesson content for a specific step."""
    try:
        lesson = lesson_service.get_lesson(step_title)
        if not lesson:
            raise HTTPException(status_code=404, detail="Lesson not found.")
        return lesson
    except Exception as e:
        logger.error(f"Error in /lesson: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch lesson content.")
