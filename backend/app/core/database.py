"""
MongoDB database connection and initialization.
Uses Motor for async MongoDB operations and Beanie ODM for document models.
"""

from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from typing import Optional
import logging

from app.core.config import settings
from app.models.database import User, Conversation, Message, Feedback, ManualDocument, DeviceCategory

logger = logging.getLogger(__name__)


class Database:
    """MongoDB database manager."""
    
    client: Optional[AsyncIOMotorClient] = None
    
    @classmethod
    async def connect_db(cls):
        """Connect to MongoDB and initialize Beanie ODM."""
        try:
            logger.info(f"Connecting to MongoDB at {settings.mongodb_url}")

            cls.client = AsyncIOMotorClient(
                settings.mongodb_url,
                maxPoolSize=settings.mongodb_max_pool_size,
                minPoolSize=settings.mongodb_min_pool_size,
                serverSelectionTimeoutMS=30000,   # 30s — Atlas needs time for replica set discovery
                connectTimeoutMS=30000,           # 30s — TLS + network handshake
                socketTimeoutMS=30000,            # 30s — per-operation socket timeout
            )

            # Get database
            db = cls.client[settings.mongodb_db_name]

            # Initialize Beanie with document models
            await init_beanie(
                database=db,
                document_models=[
                    User,
                    Conversation,
                    Message,
                    Feedback,
                    ManualDocument,
                    DeviceCategory,
                ]
            )

            logger.info("Successfully connected to MongoDB and initialized Beanie")

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    @classmethod
    async def close_db(cls):
        """Close MongoDB connection."""
        if cls.client:
            cls.client.close()
            logger.info("MongoDB connection closed")


# Dependency for FastAPI
async def get_database():
    """Dependency to get database connection."""
    return Database.client[settings.mongodb_db_name]
