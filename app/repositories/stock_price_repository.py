"""주가 데이터 Repository"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc, asc
from app.models.stock import StockPrice, Stock
from app.repositories.base import BaseRepository


class StockPriceRepository(BaseRepository[StockPrice]):
    """주가 데이터 Repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(StockPrice, session)
    
    async def get_by_stock_and_interval(
        self, 
        stock_id: str, 
        interval_unit: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[StockPrice]:
        """종목별, 간격별 주가 데이터 조회"""
        query = select(StockPrice).where(
            and_(
                StockPrice.stock_id == stock_id,
                StockPrice.interval_unit == interval_unit
            )
        )
        
        if start_date:
            query = query.where(StockPrice.timestamp >= start_date)
        if end_date:
            query = query.where(StockPrice.timestamp <= end_date)
        
        query = query.order_by(desc(StockPrice.timestamp)).offset(offset).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_latest_price(self, stock_id: str, interval_unit: str) -> Optional[StockPrice]:
        """최신 주가 데이터 조회"""
        query = select(StockPrice).where(
            and_(
                StockPrice.stock_id == stock_id,
                StockPrice.interval_unit == interval_unit
            )
        ).order_by(desc(StockPrice.timestamp)).limit(1)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_price_range(
        self,
        stock_id: str,
        interval_unit: str,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """기간별 최고가, 최저가 조회"""
        query = select(
            func.min(StockPrice.low_price),
            func.max(StockPrice.high_price)
        ).where(
            and_(
                StockPrice.stock_id == stock_id,
                StockPrice.interval_unit == interval_unit,
                StockPrice.timestamp >= start_date,
                StockPrice.timestamp <= end_date
            )
        )
        
        result = await self.session.execute(query)
        min_price, max_price = result.first()
        return min_price, max_price
    
    async def get_volume_sum(
        self,
        stock_id: str,
        interval_unit: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """기간별 총 거래량 조회"""
        query = select(func.sum(StockPrice.volume)).where(
            and_(
                StockPrice.stock_id == stock_id,
                StockPrice.interval_unit == interval_unit,
                StockPrice.timestamp >= start_date,
                StockPrice.timestamp <= end_date
            )
        )
        
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def upsert_price(self, price_data: Dict[str, Any]) -> StockPrice:
        """주가 데이터 UPSERT"""
        # 기존 데이터 확인
        existing_price = await self.session.execute(
            select(StockPrice).where(
                and_(
                    StockPrice.stock_id == price_data["stock_id"],
                    StockPrice.interval_unit == price_data["interval_unit"],
                    StockPrice.timestamp == price_data["timestamp"]
                )
            )
        )
        existing_price = existing_price.scalar_one_or_none()
        
        if existing_price:
            # 업데이트
            for field, value in price_data.items():
                if hasattr(existing_price, field):
                    setattr(existing_price, field, value)
            await self.session.commit()
            await self.session.refresh(existing_price)
            return existing_price
        else:
            # 생성
            return await self.create(price_data)
    
    async def bulk_upsert_prices(self, prices_data: List[Dict[str, Any]]) -> List[StockPrice]:
        """대량 주가 데이터 UPSERT"""
        upserted_prices = []
        
        for price_data in prices_data:
            upserted_price = await self.upsert_price(price_data)
            upserted_prices.append(upserted_price)
        
        return upserted_prices
    
    async def get_prices_with_stock_info(
        self,
        stock_id: str,
        interval_unit: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Tuple[StockPrice, Stock]]:
        """종목 정보와 함께 주가 데이터 조회"""
        query = select(StockPrice, Stock).join(
            Stock, StockPrice.stock_id == Stock.ticker
        ).where(
            and_(
                StockPrice.stock_id == stock_id,
                StockPrice.interval_unit == interval_unit
            )
        )
        
        if start_date:
            query = query.where(StockPrice.timestamp >= start_date)
        if end_date:
            query = query.where(StockPrice.timestamp <= end_date)
        
        query = query.order_by(desc(StockPrice.timestamp)).offset(offset).limit(limit)
        
        result = await self.session.execute(query)
        return result.all()
    
    async def get_multiple_stocks_latest_prices(
        self, 
        stock_ids: List[str], 
        interval_unit: str
    ) -> List[StockPrice]:
        """여러 종목의 최신 주가 조회"""
        # 서브쿼리로 각 종목별 최신 시간 조회
        subquery = select(
            StockPrice.stock_id,
            func.max(StockPrice.timestamp).label("max_timestamp")
        ).where(
            and_(
                StockPrice.stock_id.in_(stock_ids),
                StockPrice.interval_unit == interval_unit
            )
        ).group_by(StockPrice.stock_id).subquery()
        
        # 메인 쿼리로 최신 데이터 조회
        query = select(StockPrice).join(
            subquery,
            and_(
                StockPrice.stock_id == subquery.c.stock_id,
                StockPrice.timestamp == subquery.c.max_timestamp
            )
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_price_statistics(
        self,
        stock_id: str,
        interval_unit: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """주가 통계 정보 조회"""
        query = select(
            func.count(StockPrice.id).label("count"),
            func.avg(StockPrice.close_price).label("avg_price"),
            func.min(StockPrice.low_price).label("min_price"),
            func.max(StockPrice.high_price).label("max_price"),
            func.sum(StockPrice.volume).label("total_volume"),
            func.stddev(StockPrice.close_price).label("volatility")
        ).where(
            and_(
                StockPrice.stock_id == stock_id,
                StockPrice.interval_unit == interval_unit,
                StockPrice.timestamp >= start_date,
                StockPrice.timestamp <= end_date
            )
        )
        
        result = await self.session.execute(query)
        stats = result.first()
        
        return {
            "count": stats.count or 0,
            "avg_price": stats.avg_price or Decimal('0'),
            "min_price": stats.min_price or Decimal('0'),
            "max_price": stats.max_price or Decimal('0'),
            "total_volume": stats.total_volume or 0,
            "volatility": stats.volatility or Decimal('0')
        }
