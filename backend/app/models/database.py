"""
MongoDB document models using Beanie ODM.
These models represent collections in MongoDB.
"""

from beanie import Document, Indexed, Link
from pydantic import Field, EmailStr
from typing import Optional, List, Dict, Any, Annotated
from datetime import datetime, timezone
from enum import Enum
import uuid
from pymongo import ASCENDING


class DocumentStatus(str, Enum):
    """Status of document processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class MessageRole(str, Enum):
    """Role of message sender."""
    USER = "user"
    ASSISTANT = "assistant"


class User(Document):
    """User account document."""
    
    user_id: Indexed(str) = Field(default_factory=lambda: str(uuid.uuid4()))
    email: Indexed(EmailStr, unique=True)
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False
    # Profile fields
    full_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_color: Optional[str] = None   # hex color chosen by user
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Settings:
        name = "users"
        indexes = [
            "email",
            "user_id",
        ]


class Conversation(Document):
    """Conversation document storing chat sessions."""
    
    conversation_id: Indexed(str) = Field(default_factory=lambda: str(uuid.uuid4()))
    user: Link[User]
    title: Optional[str] = None          # LLM-generated, set after first message
    device_type: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Settings:
        name = "conversations"
        indexes = [
            "conversation_id",
            "user",
            "device_type",
            "created_at",
        ]


class Source(Document):
    """Embedded source citation."""
    
    content: str
    source_file: str
    page_number: Optional[int] = None
    section_name: Optional[str] = None
    relevance_score: float
    
    class Settings:
        is_embedded = True


class Message(Document):
    """Message document in a conversation."""
    
    message_id: Indexed(str) = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation: Link[Conversation]
    role: MessageRole
    content: str
    sources: Optional[List[Dict[str, Any]]] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Settings:
        name = "messages"
        indexes = [
            "message_id",
            "conversation",
            "created_at",
        ]


class Feedback(Document):
    """User feedback on assistant responses."""
    
    feedback_id: Indexed(str) = Field(default_factory=lambda: str(uuid.uuid4()))
    message: Link[Message]
    rating: int  # 1 (thumbs down) or 5 (thumbs up)
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Settings:
        name = "feedback"
        indexes = [
            "feedback_id",
            "message",
            "rating",
            "created_at",
        ]


class ManualDocument(Document):
    """Document metadata for uploaded manuals."""
    
    document_id: Indexed(str) = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    device_type: Indexed(str)
    brand: Indexed(str)
    model: Optional[str] = None
    file_path: str
    file_size: int  # in bytes
    status: DocumentStatus = DocumentStatus.PENDING
    error_message: Optional[str] = None
    chunks_count: int = 0
    uploaded_by: Link[User]
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None
    
    class Settings:
        name = "documents"
        indexes = [
            "document_id",
            "device_type",
            "brand",
            "model",
            "status",
            "uploaded_at",
        ]


class DeviceCategory(Document):
    """Device category and supported models."""
    
    category_id: Indexed(str) = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str  # e.g., "Refrigerator", "Washing Machine"
    brands: List[str] = []
    models: Dict[str, List[str]] = {}  # {brand: [model1, model2]}
    icon: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Settings:
        name = "device_categories"
        indexes = [
            "category_id",
            "name",
        ]
