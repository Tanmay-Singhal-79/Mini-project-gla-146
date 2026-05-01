from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr


# ─── Auth Schemas ─────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    token: str
    user: "UserOut"


class UserOut(BaseModel):
    id: str
    name: str
    email: str
    current_interest: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Progress Schemas ─────────────────────────────────────────────────────────

class ProgressUpdateRequest(BaseModel):
    step_title: str


class ProgressItem(BaseModel):
    id: str
    step_title: str
    status: str
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class ProgressResponse(BaseModel):
    completedItems: List[ProgressItem]
    percentage: float


# ─── Favorite Schemas ─────────────────────────────────────────────────────────

class FavoriteUpdateRequest(BaseModel):
    step_title: str

class FavoriteItem(BaseModel):
    id: str
    step_title: str
    created_at: datetime

    class Config:
        from_attributes = True


# ─── AI / Recommendation Schemas ──────────────────────────────────────────────

class RecommendRequest(BaseModel):
    interest: str


class TopicItem(BaseModel):
    id: str
    title: str
    difficulty: str


class RecommendResponse(BaseModel):
    topics: List[TopicItem]


class PathStep(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    status: str  # "completed" or "pending"


class GeneratePathResponse(BaseModel):
    steps: List[PathStep]


# ─── Lesson Content Schemas ─────────────────────────────────────────────────

class LessonSection(BaseModel):
    heading: str
    content: str
    code: Optional[str] = None
    language: Optional[str] = "javascript"


class LessonResponse(BaseModel):
    title: str
    introduction: str
    sections: List[LessonSection]
    tip: Optional[str] = None
    try_it: Optional[str] = None
    color: Optional[str] = "#4f46e5"
    icon: Optional[str] = "📚"


# ─── Community Schemas ────────────────────────────────────────────────────────

class AddResourceRequest(BaseModel):
    title: str


class ResourceOut(BaseModel):
    id: str
    title: str
    creator_name: Optional[str] = "Community Member"
    upvotes: int
    comments_count: Optional[int] = 0
    created_at: datetime

    class Config:
        from_attributes = True


class UpvoteRequest(BaseModel):
    resource_id: str


class AddCommentRequest(BaseModel):
    content: str


class CommentOut(BaseModel):
    id: str
    resource_id: str
    content: str
    creator_name: str
    creator_id: str
    created_at: datetime

    class Config:
        from_attributes = True


# Update forward references
TokenResponse.model_rebuild()
