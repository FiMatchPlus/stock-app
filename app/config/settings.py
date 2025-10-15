"""애플리케이션 설정 관리"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    app_name: str = Field(default="Stock Data Collection Server", description="애플리케이션 이름")
    app_version: str = Field(default="1.0.0", description="애플리케이션 버전")
    debug: bool = Field(default=False, description="디버그 모드")
    log_level: str = Field(default="INFO", description="로그 레벨")
    
    host: str = Field(default="0.0.0.0", description="서버 호스트")
    port: int = Field(default=8000, description="서버 포트")
    
    database_url: str = Field(..., description="PostgreSQL 데이터베이스 URL")
    
    redis_url: str = Field(..., description="Redis URL")
    
    mongodb_url: str = Field(default="mongodb://localhost:27017", description="MongoDB URL")
    mongodb_database: str = Field(default="stockone19", description="MongoDB 데이터베이스 이름")
    

    kis_base_url: str = Field(default="https://openapi.koreainvestment.com:9443", description="한국투자증권 API 기본 URL")
    kis_app_key: str = Field(..., description="한국투자증권 APP_KEY")
    kis_app_secret: str = Field(..., description="한국투자증권 APP_SECRET")
    kis_token_url: str = Field(default="/oauth2/tokenP", description="한국투자증권 토큰 발급 URL")
    
    max_workers: int = Field(default=4, description="최대 워커 수")
    chunk_size: int = Field(default=1000, description="청크 크기")
    
    cache_ttl: int = Field(default=300, description="캐시 TTL(초)")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()

