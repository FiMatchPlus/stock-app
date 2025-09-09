"""Redis 클라이언트 관리"""

import json
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import redis.asyncio as redis
from app.config import settings


class RedisClient:
    """Redis 클라이언트 래퍼"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
    
    async def connect(self):
        """Redis 연결"""
        if not self._redis:
            # redis.asyncio.from_url is synchronous and returns an async Redis client
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True
            )
    
    async def disconnect(self):
        """Redis 연결 해제"""
        if self._redis:
            # In redis-py asyncio, aclose is the async close method
            try:
                await self._redis.aclose()  # type: ignore[attr-defined]
            except AttributeError:
                # Fallback for older redis versions
                await self._redis.close()
            self._redis = None
    
    async def get(self, key: str) -> Optional[str]:
        """값 조회"""
        await self.connect()
        return await self._redis.get(key)
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """값 저장"""
        await self.connect()
        
        if isinstance(value, (dict, list)):
            value = json.dumps(value, default=str)
        
        if ttl:
            return await self._redis.setex(key, ttl, value)
        else:
            return await self._redis.set(key, value)
    
    async def delete(self, key: str) -> bool:
        """키 삭제"""
        await self.connect()
        return await self._redis.delete(key) > 0
    
    async def exists(self, key: str) -> bool:
        """키 존재 여부 확인"""
        await self.connect()
        return await self._redis.exists(key) > 0
    
    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """JSON 값 조회"""
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None
    
    async def set_json(
        self, 
        key: str, 
        value: Dict[str, Any], 
        ttl: Optional[int] = None
    ) -> bool:
        """JSON 값 저장"""
        return await self.set(key, value, ttl)
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """해시 필드 값 조회"""
        await self.connect()
        return await self._redis.hget(name, key)
    
    async def hset(self, name: str, key: str, value: Any) -> bool:
        """해시 필드 값 저장"""
        await self.connect()
        
        if isinstance(value, (dict, list)):
            value = json.dumps(value, default=str)
        
        return await self._redis.hset(name, key, value) >= 0
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """해시 모든 필드 조회"""
        await self.connect()
        return await self._redis.hgetall(name)
    
    async def hdel(self, name: str, key: str) -> bool:
        """해시 필드 삭제"""
        await self.connect()
        return await self._redis.hdel(name, key) > 0
    
    async def lpush(self, key: str, value: Any) -> int:
        """리스트 앞에 값 추가"""
        await self.connect()
        
        if isinstance(value, (dict, list)):
            value = json.dumps(value, default=str)
        
        return await self._redis.lpush(key, value)
    
    async def rpop(self, key: str) -> Optional[str]:
        """리스트 뒤에서 값 제거"""
        await self.connect()
        return await self._redis.rpop(key)
    
    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """리스트 범위 조회"""
        await self.connect()
        return await self._redis.lrange(key, start, end)
    
    async def expire(self, key: str, ttl: int) -> bool:
        """키 만료 시간 설정"""
        await self.connect()
        return await self._redis.expire(key, ttl)
    
    async def ttl(self, key: str) -> int:
        """키 남은 만료 시간 조회"""
        await self.connect()
        return await self._redis.ttl(key)
    
    async def publish(self, channel: str, message: Any) -> int:
        """채널에 메시지 발행"""
        await self.connect()
        
        if isinstance(message, (dict, list)):
            message = json.dumps(message, default=str)
        
        return await self._redis.publish(channel, message)
    
    async def subscribe(self, *channels: str):
        """채널 구독"""
        await self.connect()
        return self._redis.pubsub().subscribe(*channels)


# 전역 Redis 클라이언트 인스턴스
_redis_client: Optional[RedisClient] = None


async def get_redis_client() -> RedisClient:
    """Redis 클라이언트 의존성"""
    global _redis_client
    
    if not _redis_client:
        _redis_client = RedisClient(settings.redis_url)
        await _redis_client.connect()
    
    return _redis_client


async def close_redis_client():
    """Redis 클라이언트 연결 해제"""
    global _redis_client
    
    if _redis_client:
        await _redis_client.disconnect()
        _redis_client = None

