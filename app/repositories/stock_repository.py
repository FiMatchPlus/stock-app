"""종목 정보 Repository"""

from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from app.models.stock import Stock
from app.repositories.base import BaseRepository


class StockRepository(BaseRepository[Stock]):
    """종목 정보 Repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(Stock, session)
    
    async def get_by_ticker(self, ticker: str) -> Optional[Stock]:
        """단축코드로 종목 조회"""
        return await self.get_by_field("ticker", ticker)
    
    async def get_by_isin(self, isin: str) -> Optional[Stock]:
        """ISIN으로 종목 조회"""
        return await self.get_by_field("isin", isin)
    
    async def get_by_market(self, market: str, skip: int = 0, limit: int = 100) -> List[Stock]:
        """시장별 종목 조회"""
        return await self.get_many(
            skip=skip,
            limit=limit,
            filters={"market": market}
        )
    
    async def get_active_stocks(self, skip: int = 0, limit: int = 100) -> List[Stock]:
        """활성 종목 조회"""
        return await self.get_many(
            skip=skip,
            limit=limit,
            filters={"is_active": "Y"}
        )
    
    async def search_by_name(self, name: str, skip: int = 0, limit: int = 100) -> List[Stock]:
        """종목명으로 검색"""
        query = select(Stock).where(
            Stock.name.ilike(f"%{name}%")
        ).offset(skip).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_by_industry(self, industry_code: int, skip: int = 0, limit: int = 100) -> List[Stock]:
        """업종별 종목 조회"""
        return await self.get_many(
            skip=skip,
            limit=limit,
            filters={"industry_code": industry_code}
        )
    
    async def upsert_stock(self, stock_data: Dict[str, Any]) -> Stock:
        """종목 정보 UPSERT (중복 시 업데이트)"""
        existing_stock = await self.get_by_ticker(stock_data.get("ticker"))
        
        if existing_stock:
            for field, value in stock_data.items():
                if hasattr(existing_stock, field):
                    setattr(existing_stock, field, value)
            await self.session.commit()
            await self.session.refresh(existing_stock)
            return existing_stock
        else:
            return await self.create(stock_data)
    
    async def get_stocks_by_tickers(self, tickers: List[str]) -> List[Stock]:
        """여러 단축코드로 종목 조회"""
        query = select(Stock).where(Stock.ticker.in_(tickers))
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_stock_count_by_market(self) -> Dict[str, int]:
        """시장별 종목 수 조회"""
        query = select(Stock.market, func.count(Stock.id)).group_by(Stock.market)
        result = await self.session.execute(query)
        return dict(result.all())
