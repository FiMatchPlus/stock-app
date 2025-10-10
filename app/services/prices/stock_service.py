"""종목 서비스"""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.stock_repository import StockRepository
from app.models.schemas import StockCreate, StockResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)


class StockService:
    """종목 서비스"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.stock_repo = StockRepository(session)
    
    async def create_stock(self, stock_data: StockCreate) -> StockResponse:
        """종목 생성"""
        try:
            stock_dict = stock_data.model_dump()
            stock = await self.stock_repo.create(stock_dict)
            
            logger.info("Stock created", stock_id=stock.id, ticker=stock.ticker)
            
            return StockResponse.model_validate(stock)
        
        except Exception as e:
            logger.error("Failed to create stock", error=str(e), ticker=stock_data.ticker)
            raise
    
    async def get_stock_by_ticker(self, ticker: str) -> Optional[StockResponse]:
        """단축코드로 종목 조회"""
        try:
            stock = await self.stock_repo.get_by_ticker(ticker)
            
            if stock:
                return StockResponse.model_validate(stock)
            
            return None
        
        except Exception as e:
            logger.error("Failed to get stock by ticker", error=str(e), ticker=ticker)
            raise
    
    async def get_stock_by_id(self, stock_id: int) -> Optional[StockResponse]:
        """ID로 종목 조회"""
        try:
            stock = await self.stock_repo.get_by_id(stock_id)
            
            if stock:
                return StockResponse.model_validate(stock)
            
            return None
        
        except Exception as e:
            logger.error("Failed to get stock by id", error=str(e), stock_id=stock_id)
            raise
    
    async def get_stocks_by_market(
        self, 
        market: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[StockResponse]:
        """시장별 종목 조회"""
        try:
            stocks = await self.stock_repo.get_by_market(market, skip, limit)
            
            return [StockResponse.model_validate(stock) for stock in stocks]
        
        except Exception as e:
            logger.error("Failed to get stocks by market", error=str(e), market=market)
            raise
    
    async def search_stocks(self, name: str, skip: int = 0, limit: int = 100) -> List[StockResponse]:
        """종목명으로 검색"""
        try:
            stocks = await self.stock_repo.search_by_name(name, skip, limit)
            
            return [StockResponse.model_validate(stock) for stock in stocks]
        
        except Exception as e:
            logger.error("Failed to search stocks", error=str(e), name=name)
            raise
    
    async def get_active_stocks(self, skip: int = 0, limit: int = 100) -> List[StockResponse]:
        """활성 종목 조회"""
        try:
            stocks = await self.stock_repo.get_active_stocks(skip, limit)
            
            return [StockResponse.model_validate(stock) for stock in stocks]
        
        except Exception as e:
            logger.error("Failed to get active stocks", error=str(e))
            raise
    
    async def upsert_stock(self, stock_data: Dict[str, Any]) -> StockResponse:
        """종목 정보 UPSERT"""
        try:
            stock = await self.stock_repo.upsert_stock(stock_data)
            
            logger.info("Stock upserted", stock_id=stock.id, ticker=stock.ticker)
            
            return StockResponse.model_validate(stock)
        
        except Exception as e:
            logger.error("Failed to upsert stock", error=str(e), ticker=stock_data.get("ticker"))
            raise
    
    async def get_stocks_by_tickers(self, tickers: List[str]) -> List[StockResponse]:
        """여러 단축코드로 종목 조회"""
        try:
            stocks = await self.stock_repo.get_stocks_by_tickers(tickers)
            
            return [StockResponse.model_validate(stock) for stock in stocks]
        
        except Exception as e:
            logger.error("Failed to get stocks by tickers", error=str(e), tickers=tickers)
            raise
    
    async def get_stock_count_by_market(self) -> Dict[str, int]:
        """시장별 종목 수 조회"""
        try:
            return await self.stock_repo.get_stock_count_by_market()
        
        except Exception as e:
            logger.error("Failed to get stock count by market", error=str(e))
            raise

