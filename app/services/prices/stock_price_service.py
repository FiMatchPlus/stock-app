"""주가 데이터 서비스"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.stock_price_repository import StockPriceRepository
from app.models.schemas import (
    StockPriceCreate, 
    StockPriceResponse, 
    StockPriceCollectionRequest,
    StockPriceQueryRequest
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class StockPriceService:
    """주가 데이터 서비스"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.stock_price_repo = StockPriceRepository(session)
    
    async def create_stock_price(self, price_data: StockPriceCreate) -> StockPriceResponse:
        """주가 데이터 생성"""
        try:
            price_dict = price_data.model_dump(by_alias=True)
            price = await self.stock_price_repo.create(price_dict)
            
            logger.info(
                "Stock price created",
                price_id=price.id,
                stock_code=price.stock_code,
                datetime=price.datetime
            )
            
            return StockPriceResponse.model_validate(price, from_attributes=True)
        
        except Exception as e:
            logger.error(
                "Failed to create stock price",
                error=str(e),
                stock_code=price_data.stock_code
            )
            raise
    
    async def upsert_stock_price(self, price_data: Dict[str, Any]) -> StockPriceResponse:
        """주가 데이터 UPSERT"""
        try:
            price = await self.stock_price_repo.upsert_price(price_data)
            
            logger.info(
                "Stock price upserted",
                price_id=price.id,
                stock_code=price.stock_code,
                datetime=price.datetime
            )
            
            return StockPriceResponse.model_validate(price, from_attributes=True)
        
        except Exception as e:
            logger.error(
                "Failed to upsert stock price",
                error=str(e),
                stock_code=price_data.get("stock_code")
            )
            raise
    
    async def bulk_upsert_stock_prices(self, prices_data: List[Dict[str, Any]]) -> List[StockPriceResponse]:
        """대량 주가 데이터 UPSERT"""
        try:
            prices = await self.stock_price_repo.bulk_upsert_prices(prices_data)
            
            logger.info(
                "Bulk stock prices upserted",
                count=len(prices),
                stock_codes=list(set(price.stock_code for price in prices))
            )
            
            return [StockPriceResponse.model_validate(price) for price in prices]
        
        except Exception as e:
            logger.error("Failed to bulk upsert stock prices", error=str(e))
            raise
    
    async def get_stock_prices(
        self, 
        stock_code: str, 
        interval_unit: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[StockPriceResponse]:
        """주가 데이터 조회"""
        try:
            prices = await self.stock_price_repo.get_by_stock_and_interval(
                stock_code=stock_code,
                interval_unit=interval_unit,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                offset=offset
            )
            
            return [StockPriceResponse.model_validate(price) for price in prices]
        
        except Exception as e:
            logger.error(
                "Failed to get stock prices",
                error=str(e),
                stock_code=stock_code,
                interval_unit=interval_unit
            )
            raise
    
    async def get_latest_stock_price(
        self, 
        stock_code: str, 
        interval_unit: str
    ) -> Optional[StockPriceResponse]:
        """최신 주가 데이터 조회"""
        try:
            price = await self.stock_price_repo.get_latest_price(stock_code, interval_unit)
            
            if price:
                return StockPriceResponse.model_validate(price, from_attributes=True)
            
            return None
        
        except Exception as e:
            logger.error(
                "Failed to get latest stock price",
                error=str(e),
                stock_code=stock_code,
                interval_unit=interval_unit
            )
            raise
    
    async def get_multiple_stocks_latest_prices(
        self, 
        stock_codes: List[str], 
        interval_unit: str
    ) -> List[StockPriceResponse]:
        """여러 종목의 최신 주가 조회"""
        try:
            prices = await self.stock_price_repo.get_multiple_stocks_latest_prices(
                stock_codes, interval_unit
            )
            
            return [StockPriceResponse.model_validate(price) for price in prices]
        
        except Exception as e:
            logger.error(
                "Failed to get multiple stocks latest prices",
                error=str(e),
                stock_codes=stock_codes,
                interval_unit=interval_unit
            )
            raise
    
    async def get_price_statistics(
        self,
        stock_code: str,
        interval_unit: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """주가 통계 정보 조회"""
        try:
            stats = await self.stock_price_repo.get_price_statistics(
                stock_code=stock_code,
                interval_unit=interval_unit,
                start_date=start_date,
                end_date=end_date
            )
            
            return stats
        
        except Exception as e:
            logger.error(
                "Failed to get price statistics",
                error=str(e),
                stock_code=stock_code,
                interval_unit=interval_unit
            )
            raise
    
    async def collect_stock_prices(
        self, 
        collection_request: StockPriceCollectionRequest
    ) -> Dict[str, List[StockPriceResponse]]:
        """주가 데이터 수집"""
        try:
            from app.services.prices.external_api_service import ExternalAPIService
            
            api_service = ExternalAPIService()
            
            if len(collection_request.symbols) == 1:
                symbol = collection_request.symbols[0]
                prices_data = await api_service.collect_stock_prices(
                    symbol=symbol,
                    interval=collection_request.interval,
                    start_date=collection_request.start_date,
                    end_date=collection_request.end_date
                )
                
                upserted_prices = await self.bulk_upsert_stock_prices(prices_data)
                
                return {symbol: upserted_prices}
            
            else:
                prices_data_dict = await api_service.collect_multiple_stock_prices(
                    symbols=collection_request.symbols,
                    interval=collection_request.interval,
                    start_date=collection_request.start_date,
                    end_date=collection_request.end_date
                )
                
                result = {}
                for symbol, prices_data in prices_data_dict.items():
                    upserted_prices = await self.bulk_upsert_stock_prices(prices_data)
                    result[symbol] = upserted_prices
                
                return result
        
        except Exception as e:
            logger.error(
                "Failed to collect stock prices",
                error=str(e),
                symbols=collection_request.symbols
            )
            raise
    
    async def query_stock_prices(
        self, 
        query_request: StockPriceQueryRequest
    ) -> List[StockPriceResponse]:
        """주가 데이터 조회 (쿼리)"""
        try:
            prices = await self.get_stock_prices(
                stock_code=query_request.symbol,
                interval_unit=query_request.interval or "1d",
                start_date=query_request.start_date,
                end_date=query_request.end_date,
                limit=query_request.limit or 100,
                offset=query_request.offset or 0
            )
            
            return prices
        
        except Exception as e:
            logger.error(
                "Failed to query stock prices",
                error=str(e),
                symbol=query_request.symbol
            )
            raise

