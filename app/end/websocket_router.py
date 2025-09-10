"""WebSocket 엔드포인트"""

import asyncio
import json
from typing import Dict, Set, List, Any
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_async_session
from app.models.schemas import WebSocketMessage
from app.services.stock_price_service import StockPriceService
from app.utils.redis_client import get_redis_client, RedisClient
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ws", tags=["websocket"])


class ConnectionManager:
    """WebSocket 연결 관리자"""
    
    def __init__(self):
        # 활성 연결 저장
        self.active_connections: Dict[str, WebSocket] = {}
        # 종목별 구독자 관리
        self.symbol_subscribers: Dict[str, Set[str]] = {}
        # 다중 종목 구독자 관리
        self.multiple_subscribers: Set[str] = set()
        # 하트비트 관리
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """WebSocket 연결 수락"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        # 하트비트 태스크 시작
        self.heartbeat_tasks[client_id] = asyncio.create_task(
            self._heartbeat_loop(client_id)
        )
        
        logger.info("WebSocket connected", client_id=client_id)
    
    async def disconnect(self, client_id: str):
        """WebSocket 연결 해제"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # 구독 정보 정리
        for symbol, subscribers in self.symbol_subscribers.items():
            subscribers.discard(client_id)
        
        self.multiple_subscribers.discard(client_id)
        
        # 하트비트 태스크 정리
        if client_id in self.heartbeat_tasks:
            self.heartbeat_tasks[client_id].cancel()
            del self.heartbeat_tasks[client_id]
        
        logger.info("WebSocket disconnected", client_id=client_id)
    
    async def subscribe_symbol(self, client_id: str, symbol: str):
        """특정 종목 구독"""
        if symbol not in self.symbol_subscribers:
            self.symbol_subscribers[symbol] = set()
        
        self.symbol_subscribers[symbol].add(client_id)
        
        logger.info("Symbol subscribed", client_id=client_id, symbol=symbol)
    
    async def unsubscribe_symbol(self, client_id: str, symbol: str):
        """특정 종목 구독 해제"""
        if symbol in self.symbol_subscribers:
            self.symbol_subscribers[symbol].discard(client_id)
        
        logger.info("Symbol unsubscribed", client_id=client_id, symbol=symbol)
    
    async def subscribe_multiple(self, client_id: str):
        """다중 종목 구독"""
        self.multiple_subscribers.add(client_id)
        
        logger.info("Multiple symbols subscribed", client_id=client_id)
    
    async def unsubscribe_multiple(self, client_id: str):
        """다중 종목 구독 해제"""
        self.multiple_subscribers.discard(client_id)
        
        logger.info("Multiple symbols unsubscribed", client_id=client_id)
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """특정 클라이언트에게 메시지 전송"""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(
                    "Failed to send message to client",
                    error=str(e),
                    client_id=client_id
                )
                await self.disconnect(client_id)
    
    async def send_to_symbol_subscribers(self, symbol: str, message: Dict[str, Any]):
        """특정 종목 구독자들에게 메시지 전송"""
        if symbol in self.symbol_subscribers:
            for client_id in list(self.symbol_subscribers[symbol]):
                await self.send_to_client(client_id, message)
    
    async def send_to_multiple_subscribers(self, message: Dict[str, Any]):
        """다중 종목 구독자들에게 메시지 전송"""
        for client_id in list(self.multiple_subscribers):
            await self.send_to_client(client_id, message)
    
    async def broadcast(self, message: Dict[str, Any]):
        """모든 연결된 클라이언트에게 메시지 전송"""
        for client_id in list(self.active_connections.keys()):
            await self.send_to_client(client_id, message)
    
    async def _heartbeat_loop(self, client_id: str):
        """하트비트 루프"""
        from app.config import settings
        
        try:
            while True:
                await asyncio.sleep(settings.websocket_heartbeat_interval)
                
                if client_id in self.active_connections:
                    await self.send_to_client(client_id, {
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                else:
                    break
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                "Heartbeat loop error",
                error=str(e),
                client_id=client_id
            )


# 전역 연결 관리자
manager = ConnectionManager()


@router.websocket("/market-data/{symbol}")
async def websocket_single_symbol(
    websocket: WebSocket,
    symbol: str,
    client_id: str = Query(..., description="클라이언트 ID")
):
    """개별 종목 실시간 스트리밍"""
    await manager.connect(websocket, client_id)
    await manager.subscribe_symbol(client_id, symbol)
    
    try:
        # Redis 구독 시작
        redis_client = await get_redis_client()
        pubsub = await redis_client.subscribe(f"stock_price:{symbol}")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    
                    # 구독자들에게 전송
                    await manager.send_to_symbol_subscribers(symbol, {
                        "type": "price_update",
                        "symbol": symbol,
                        "data": data,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse Redis message", error=str(e))
                except Exception as e:
                    logger.error("Failed to process price update", error=str(e))
    
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error("WebSocket error", error=str(e), symbol=symbol)
        await manager.disconnect(client_id)


@router.websocket("/market-data/multiple")
async def websocket_multiple_symbols(
    websocket: WebSocket,
    client_id: str = Query(..., description="클라이언트 ID"),
    symbols: str = Query(..., description="종목코드 목록 (쉼표로 구분)")
):
    """여러 종목 동시 스트리밍"""
    await manager.connect(websocket, client_id)
    await manager.subscribe_multiple(client_id)
    
    # 종목 목록 파싱
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    
    try:
        # Redis 구독 시작
        redis_client = await get_redis_client()
        channels = [f"stock_price:{symbol}" for symbol in symbol_list]
        pubsub = await redis_client.subscribe(*channels)
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    symbol = data.get("symbol")
                    
                    if symbol:
                        # 다중 구독자들에게 전송
                        await manager.send_to_multiple_subscribers({
                            "type": "price_update",
                            "symbol": symbol,
                            "data": data,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse Redis message", error=str(e))
                except Exception as e:
                    logger.error("Failed to process price update", error=str(e))
    
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error("WebSocket error", error=str(e), symbols=symbol_list)
        await manager.disconnect(client_id)


@router.websocket("/market-data")
async def websocket_general(
    websocket: WebSocket,
    client_id: str = Query(..., description="클라이언트 ID")
):
    """일반 시장 데이터 스트리밍"""
    await manager.connect(websocket, client_id)
    
    try:
        # Redis 구독 시작 (모든 주가 데이터)
        redis_client = await get_redis_client()
        pubsub = await redis_client.subscribe("stock_price:*")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    symbol = data.get("symbol")
                    
                    if symbol:
                        # 모든 연결된 클라이언트에게 전송
                        await manager.broadcast({
                            "type": "price_update",
                            "symbol": symbol,
                            "data": data,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse Redis message", error=str(e))
                except Exception as e:
                    logger.error("Failed to process price update", error=str(e))
    
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        await manager.disconnect(client_id)


@router.post("/publish/{symbol}")
async def publish_price_update(
    symbol: str,
    data: Dict[str, Any],
    redis_client: RedisClient = Depends(get_redis_client)
):
    """주가 데이터 발행 (테스트용)"""
    try:
        message = {
            "symbol": symbol,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Redis에 발행
        await redis_client.publish(f"stock_price:{symbol}", message)
        
        logger.info("Price update published", symbol=symbol)
        
        return {"message": "Price update published successfully", "symbol": symbol}
    
    except Exception as e:
        logger.error("Failed to publish price update", error=str(e), symbol=symbol)
        raise


@router.get("/connections")
async def get_connection_info():
    """연결 정보 조회 (관리용)"""
    return {
        "active_connections": len(manager.active_connections),
        "symbol_subscribers": {
            symbol: len(subscribers) 
            for symbol, subscribers in manager.symbol_subscribers.items()
        },
        "multiple_subscribers": len(manager.multiple_subscribers)
    }


