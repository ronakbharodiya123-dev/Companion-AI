"""
Pydantic models for API request and response validation.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# Authentication Models
# ============================================================================

class UserCreate(BaseModel):
    """Request model for user registration."""
    email: EmailStr
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    """Request model for user login."""
    email: EmailStr
    password: str


class Token(BaseModel):
    """Response model for authentication token."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Data extracted from JWT token."""
    email: Optional[str] = None


class UserResponse(BaseModel):
    """Response model for user data."""
    user_id: str
    email: str
    is_active: bool
    created_at: datetime


class UserProfileResponse(BaseModel):
    """Extended response model including profile fields."""
    user_id: str
    email: str
    full_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_color: Optional[str] = None
    is_active: bool
    is_admin: bool = False
    created_at: datetime
    updated_at: datetime


class UserUpdate(BaseModel):
    """Request model for updating user profile."""
    full_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=300)
    avatar_color: Optional[str] = Field(None, pattern=r'^#[0-9A-Fa-f]{6}$')


class ChangePasswordRequest(BaseModel):
    """Request model for changing password."""
    current_password: str
    new_password: str = Field(..., min_length=8)


# ============================================================================
# Chat Models
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str = Field(..., min_length=1, max_length=1000)
    device_type: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    conversation_id: Optional[str] = None
    ai_model: Optional[str] = Field(default="gemini", description="AI model to use: 'gemini' or 'groq'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "My refrigerator is not cooling properly",
                "device_type": "Refrigerator",
                "brand": "Samsung",
                "model": "RF28R7351SR",
                "conversation_id": None,
                "ai_model": "gemini"
            }
        }


class SourceCitation(BaseModel):
    """Source citation for an answer."""
    content: str
    source_file: str
    page_number: Optional[int] = None
    section_name: Optional[str] = None
    relevance_score: float


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    sources: List[SourceCitation] = []
    conversation_id: str
    message_id: str
    timestamp: datetime
    title: Optional[str] = None          # Populated only on the first message
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Based on your Samsung refrigerator manual, here are troubleshooting steps...",
                "sources": [
                    {
                        "content": "If the refrigerator is not cooling, check the temperature settings...",
                        "source_file": "Samsung_RF28R7351SR_Manual.pdf",
                        "page_number": 45,
                        "section_name": "Troubleshooting",
                        "relevance_score": 0.92
                    }
                ],
                "conversation_id": "abc123",
                "message_id": "msg456",
                "timestamp": "2026-02-09T22:19:15Z"
            }
        }


# ============================================================================
# Document Upload Models
# ============================================================================

class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str
    filename: str
    device_type: str
    brand: str
    model: Optional[str]
    status: str
    message: str


class DocumentMetadata(BaseModel):
    """Metadata for document upload."""
    device_type: str = Field(..., min_length=1)
    brand: str = Field(..., min_length=1)
    model: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    document_id: str
    filename: str
    device_type: str
    brand: str
    model: Optional[str]
    status: str
    chunks_count: int
    uploaded_at: datetime
    processed_at: Optional[datetime]


# ============================================================================
# Device Models
# ============================================================================

class DeviceInfo(BaseModel):
    """Device information."""
    device_type: str
    brands: List[str]
    models: Dict[str, List[str]]


class DeviceListResponse(BaseModel):
    """Response model for device list."""
    devices: List[DeviceInfo]
    total_count: int


# ============================================================================
# Conversation Models
# ============================================================================

class ConversationResponse(BaseModel):
    """Response model for conversation details."""
    conversation_id: str
    title: Optional[str] = None
    device_type: Optional[str]
    brand: Optional[str]
    model: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int


class MessageResponse(BaseModel):
    """Response model for a single message."""
    message_id: str
    role: str
    content: str
    sources: List[SourceCitation] = []
    created_at: datetime


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    conversation: ConversationResponse
    messages: List[MessageResponse]


# ============================================================================
# Feedback Models
# ============================================================================

class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    message_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = Field(None, max_length=500)
    
    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "msg456",
                "rating": 5,
                "comment": "Very helpful troubleshooting steps!"
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    feedback_id: str
    message: str


# ============================================================================
# Health Check Models
# ============================================================================

class HealthStatus(str, Enum):
    """Health status enum."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ServiceHealth(BaseModel):
    """Health status of a service."""
    status: HealthStatus
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: HealthStatus
    timestamp: datetime
    services: Dict[str, ServiceHealth]
    version: str = "1.0.0"


# ============================================================================
# Error Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid request",
                "detail": "Query parameter is required",
                "timestamp": "2026-02-09T22:19:15Z"
            }
        }
