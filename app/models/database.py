"""데이터베이스 연결 및 세션 관리"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import settings


class Base(DeclarativeBase):
    """SQLAlchemy Base 클래스"""
    pass


# 설정에서 데이터베이스 URL 및 디버그 모드 가져오기
database_url = settings.database_url
debug_mode = settings.debug

# 비동기 엔진 생성
engine = create_async_engine(
    database_url,
    echo=debug_mode,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# 비동기 세션 팩토리
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncSession:
    """비동기 데이터베이스 세션 의존성"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

