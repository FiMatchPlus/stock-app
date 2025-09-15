"""MongoDB 연결 및 클라이언트 관리"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional
import os
from app.utils.logger import get_logger

logger = get_logger(__name__)

class MongoDBClient:
    """MongoDB 비동기 클라이언트"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
    
    async def connect(self, connection_string: str, database_name: str):
        """MongoDB 연결"""
        try:
            self.client = AsyncIOMotorClient(connection_string)
            self.database = self.client[database_name]
            
            # 연결 테스트
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB database: {database_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    async def disconnect(self):
        """MongoDB 연결 해제"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    def get_collection(self, collection_name: str):
        """컬렉션 가져오기"""
        if self.database is None:
            raise RuntimeError("Database not connected")
        return self.database[collection_name]

# 전역 MongoDB 클라이언트 인스턴스
mongodb_client = MongoDBClient()

from app.config import settings

# MongoDB 연결 설정
MONGODB_URL = settings.mongodb_url
MONGODB_DATABASE = settings.mongodb_database

async def get_mongodb_client() -> MongoDBClient:
    """MongoDB 클라이언트 의존성"""
    return mongodb_client
