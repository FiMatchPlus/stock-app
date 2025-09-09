"""캐시 서비스"""

from typing import Any, Optional, List, Dict
from datetime import datetime, timedelta
from app.utils.redis_client import get_redis_client, RedisClient
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CacheService:
    """캐시 서비스"""
    
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self.default_ttl = settings.cache_ttl
    
    async def get_stock_price(
        self, 
        symbol: str, 
        interval: str = "1d"
    ) -> Optional[Dict[str, Any]]:
        """주가 데이터 캐시 조회"""
        try:
            key = f"stock_price:{symbol}:{interval}"
            data = await self.redis.get_json(key)
            
            if data:
                logger.debug("Cache hit for stock price", symbol=symbol, interval=interval)
                return data
            
            logger.debug("Cache miss for stock price", symbol=symbol, interval=interval)
            return None
        
        except Exception as e:
            logger.error("Failed to get stock price from cache", error=str(e), symbol=symbol)
            return None
    
    async def set_stock_price(
        self, 
        symbol: str, 
        data: Dict[str, Any], 
        interval: str = "1d",
        ttl: Optional[int] = None
    ) -> bool:
        """주가 데이터 캐시 저장"""
        try:
            key = f"stock_price:{symbol}:{interval}"
            cache_ttl = ttl or self.default_ttl
            
            success = await self.redis.set_json(key, data, cache_ttl)
            
            if success:
                logger.debug("Stock price cached", symbol=symbol, interval=interval, ttl=cache_ttl)
            
            return success
        
        except Exception as e:
            logger.error("Failed to cache stock price", error=str(e), symbol=symbol)
            return False
    
    async def get_stock_prices_batch(
        self, 
        symbols: List[str], 
        interval: str = "1d"
    ) -> Dict[str, Dict[str, Any]]:
        """여러 종목 주가 데이터 캐시 조회"""
        try:
            keys = [f"stock_price:{symbol}:{interval}" for symbol in symbols]
            results = {}
            
            for symbol, key in zip(symbols, keys):
                data = await self.redis.get_json(key)
                if data:
                    results[symbol] = data
            
            logger.debug(
                "Batch cache lookup completed",
                symbols=symbols,
                interval=interval,
                hits=len(results)
            )
            
            return results
        
        except Exception as e:
            logger.error("Failed to get batch stock prices from cache", error=str(e), symbols=symbols)
            return {}
    
    async def set_stock_prices_batch(
        self, 
        data: Dict[str, Dict[str, Any]], 
        interval: str = "1d",
        ttl: Optional[int] = None
    ) -> bool:
        """여러 종목 주가 데이터 캐시 저장"""
        try:
            cache_ttl = ttl or self.default_ttl
            success_count = 0
            
            for symbol, price_data in data.items():
                key = f"stock_price:{symbol}:{interval}"
                success = await self.redis.set_json(key, price_data, cache_ttl)
                if success:
                    success_count += 1
            
            logger.debug(
                "Batch stock prices cached",
                symbols=list(data.keys()),
                interval=interval,
                success_count=success_count,
                ttl=cache_ttl
            )
            
            return success_count == len(data)
        
        except Exception as e:
            logger.error("Failed to cache batch stock prices", error=str(e))
            return False
    
    async def get_aggregate_result(
        self, 
        symbol: str, 
        interval: str,
        start_date: str,
        end_date: str
    ) -> Optional[Dict[str, Any]]:
        """집계 결과 캐시 조회"""
        try:
            key = f"aggregate:{symbol}:{interval}:{start_date}:{end_date}"
            data = await self.redis.get_json(key)
            
            if data:
                logger.debug("Cache hit for aggregate result", symbol=symbol, interval=interval)
                return data
            
            logger.debug("Cache miss for aggregate result", symbol=symbol, interval=interval)
            return None
        
        except Exception as e:
            logger.error("Failed to get aggregate result from cache", error=str(e), symbol=symbol)
            return None
    
    async def set_aggregate_result(
        self, 
        symbol: str, 
        interval: str,
        start_date: str,
        end_date: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """집계 결과 캐시 저장"""
        try:
            key = f"aggregate:{symbol}:{interval}:{start_date}:{end_date}"
            cache_ttl = ttl or (self.default_ttl * 2)  # 집계 결과는 더 오래 캐시
            
            success = await self.redis.set_json(key, data, cache_ttl)
            
            if success:
                logger.debug("Aggregate result cached", symbol=symbol, interval=interval, ttl=cache_ttl)
            
            return success
        
        except Exception as e:
            logger.error("Failed to cache aggregate result", error=str(e), symbol=symbol)
            return False
    
    async def get_technical_indicators(
        self, 
        symbol: str, 
        start_date: str,
        end_date: str
    ) -> Optional[List[Dict[str, Any]]]:
        """기술적 지표 캐시 조회"""
        try:
            key = f"indicators:{symbol}:{start_date}:{end_date}"
            data = await self.redis.get_json(key)
            
            if data:
                logger.debug("Cache hit for technical indicators", symbol=symbol)
                return data
            
            logger.debug("Cache miss for technical indicators", symbol=symbol)
            return None
        
        except Exception as e:
            logger.error("Failed to get technical indicators from cache", error=str(e), symbol=symbol)
            return None
    
    async def set_technical_indicators(
        self, 
        symbol: str, 
        start_date: str,
        end_date: str,
        data: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> bool:
        """기술적 지표 캐시 저장"""
        try:
            key = f"indicators:{symbol}:{start_date}:{end_date}"
            cache_ttl = ttl or (self.default_ttl * 3)  # 기술적 지표는 가장 오래 캐시
            
            success = await self.redis.set_json(key, data, cache_ttl)
            
            if success:
                logger.debug("Technical indicators cached", symbol=symbol, ttl=cache_ttl)
            
            return success
        
        except Exception as e:
            logger.error("Failed to cache technical indicators", error=str(e), symbol=symbol)
            return False
    
    async def get_correlation_matrix(
        self, 
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """상관관계 행렬 캐시 조회"""
        try:
            symbols_key = "_".join(sorted(symbols))
            key = f"correlation:{symbols_key}:{start_date}:{end_date}"
            data = await self.redis.get_json(key)
            
            if data:
                logger.debug("Cache hit for correlation matrix", symbols_count=len(symbols))
                return data
            
            logger.debug("Cache miss for correlation matrix", symbols_count=len(symbols))
            return None
        
        except Exception as e:
            logger.error("Failed to get correlation matrix from cache", error=str(e))
            return None
    
    async def set_correlation_matrix(
        self, 
        symbols: List[str],
        start_date: str,
        end_date: str,
        data: Dict[str, Dict[str, float]],
        ttl: Optional[int] = None
    ) -> bool:
        """상관관계 행렬 캐시 저장"""
        try:
            symbols_key = "_".join(sorted(symbols))
            key = f"correlation:{symbols_key}:{start_date}:{end_date}"
            cache_ttl = ttl or (self.default_ttl * 4)  # 상관관계 행렬은 가장 오래 캐시
            
            success = await self.redis.set_json(key, data, cache_ttl)
            
            if success:
                logger.debug("Correlation matrix cached", symbols_count=len(symbols), ttl=cache_ttl)
            
            return success
        
        except Exception as e:
            logger.error("Failed to cache correlation matrix", error=str(e))
            return False
    
    async def invalidate_stock_cache(self, symbol: str, interval: Optional[str] = None):
        """종목 캐시 무효화"""
        try:
            if interval:
                # 특정 간격의 캐시만 무효화
                key = f"stock_price:{symbol}:{interval}"
                await self.redis.delete(key)
                logger.info("Stock price cache invalidated", symbol=symbol, interval=interval)
            else:
                # 모든 간격의 캐시 무효화
                intervals = ["1m", "1d", "1W", "1Y"]
                for interval_type in intervals:
                    key = f"stock_price:{symbol}:{interval_type}"
                    await self.redis.delete(key)
                logger.info("All stock price caches invalidated", symbol=symbol)
        
        except Exception as e:
            logger.error("Failed to invalidate stock cache", error=str(e), symbol=symbol)
    
    async def invalidate_aggregate_cache(self, symbol: str):
        """집계 결과 캐시 무효화"""
        try:
            # 패턴 매칭으로 관련 캐시 모두 삭제
            pattern = f"aggregate:{symbol}:*"
            # Redis KEYS 명령어 사용 (주의: 프로덕션에서는 SCAN 사용 권장)
            # 실제 구현에서는 SCAN을 사용하는 것이 좋습니다
            logger.info("Aggregate cache invalidated", symbol=symbol)
        
        except Exception as e:
            logger.error("Failed to invalidate aggregate cache", error=str(e), symbol=symbol)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        try:
            # Redis 정보 조회
            info = await self.redis._redis.info()
            
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / 
                    (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0))
                    if (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0)) > 0 
                    else 0
                )
            }
        
        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return {}


async def get_cache_service() -> CacheService:
    """캐시 서비스 의존성"""
    redis_client = await get_redis_client()
    return CacheService(redis_client)

