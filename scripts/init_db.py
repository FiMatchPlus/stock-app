#!/usr/bin/env python3
"""데이터베이스 초기화 스크립트"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.models.database import engine, Base
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def init_database():
    """데이터베이스 초기화"""
    try:
        logger.info("Initializing database...")
        
        # 데이터베이스 테이블 생성
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def main():
    """메인 함수"""
    try:
        await init_database()
        print("✅ Database initialized successfully!")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


