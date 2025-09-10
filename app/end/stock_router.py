"""주가 데이터 HTTP 엔드포인트"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_async_session
from app.models.schemas import (
    StockPriceCollectionRequest,
    StockPriceQueryRequest,
    StockPriceResponse,
    StockResponse,
    ErrorResponse
)
from app.services.stock_service import StockService
from app.services.stock_price_service import StockPriceService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/stock-price", tags=["stock-price"])


@router.post(
    "/collect/{symbol}",
    response_model=dict,
    summary="특정 종목 주가 데이터 수집",
    description="특정 종목의 연월봉 데이터를 수집합니다."
)
async def collect_stock_prices_by_symbol(
    symbol: str = Path(..., description="종목코드"),
    interval: str = Query("1Y", description="시간간격 (1Y/1M)"),
    start_date: Optional[datetime] = Query(None, description="시작일"),
    end_date: Optional[datetime] = Query(None, description="종료일"),
    session: AsyncSession = Depends(get_async_session)
):
    """특정 종목의 주가 데이터 수집"""
    try:
        # 시간간격 검증
        if interval not in ["1Y", "1M", "1d", "1W"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid interval. Must be one of: 1Y, 1M, 1d, 1W"
            )
        
        # 서비스 초기화
        stock_price_service = StockPriceService(session)
        
        # 수집 요청 생성
        collection_request = StockPriceCollectionRequest(
            symbols=[symbol],
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )
        
        # 데이터 수집
        result = await stock_price_service.collect_stock_prices(collection_request)
        
        logger.info(
            "Stock prices collected by symbol",
            symbol=symbol,
            interval=interval,
            count=len(result.get(symbol, []))
        )
        
        return {
            "message": "Stock prices collected successfully",
            "symbol": symbol,
            "interval": interval,
            "count": len(result.get(symbol, [])),
            "data": result
        }
    
    except Exception as e:
        logger.error(
            "Failed to collect stock prices by symbol",
            error=str(e),
            symbol=symbol
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/collect",
    response_model=dict,
    summary="특정 날짜 일봉 데이터 수집",
    description="특정 날짜의 일봉 데이터를 수집합니다."
)
async def collect_daily_prices_by_date(
    date: datetime = Query(..., description="날짜 (yyyyMMdd 형식)"),
    session: AsyncSession = Depends(get_async_session)
):
    """특정 날짜의 일봉 데이터 수집"""
    try:
        # 서비스 초기화
        stock_price_service = StockPriceService(session)
        
        # 수집 요청 생성 (일봉 데이터)
        collection_request = StockPriceCollectionRequest(
            symbols=[],  # 빈 리스트는 전체 종목을 의미
            interval="1d",
            start_date=date,
            end_date=date
        )
        
        # 데이터 수집
        result = await stock_price_service.collect_stock_prices(collection_request)
        
        total_count = sum(len(prices) for prices in result.values())
        
        logger.info(
            "Daily prices collected by date",
            date=date,
            total_count=total_count
        )
        
        return {
            "message": "Daily prices collected successfully",
            "date": date,
            "total_count": total_count,
            "data": result
        }
    
    except Exception as e:
        logger.error(
            "Failed to collect daily prices by date",
            error=str(e),
            date=date
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/collect/batch",
    response_model=dict,
    summary="배치 주가 데이터 수집",
    description="여러 종목의 주가 데이터를 배치로 수집합니다."
)
async def collect_stock_prices_batch(
    request: StockPriceCollectionRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """배치 주가 데이터 수집"""
    try:
        # 최대 100개 종목 제한
        if len(request.symbols) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 symbols allowed per batch"
            )
        
        # 서비스 초기화
        stock_price_service = StockPriceService(session)
        
        # 데이터 수집
        result = await stock_price_service.collect_stock_prices(request)
        
        total_count = sum(len(prices) for prices in result.values())
        
        logger.info(
            "Batch stock prices collected",
            symbols=request.symbols,
            interval=request.interval,
            total_count=total_count
        )
        
        return {
            "message": "Batch stock prices collected successfully",
            "symbols": request.symbols,
            "interval": request.interval,
            "total_count": total_count,
            "data": result
        }
    
    except Exception as e:
        logger.error(
            "Failed to collect batch stock prices",
            error=str(e),
            symbols=request.symbols
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{symbol}",
    response_model=List[StockPriceResponse],
    summary="주가 데이터 조회",
    description="저장된 주가 데이터를 조회합니다."
)
async def get_stock_prices(
    symbol: str = Path(..., description="종목코드"),
    interval: Optional[str] = Query(None, description="시간간격"),
    start_date: Optional[datetime] = Query(None, description="시작일"),
    end_date: Optional[datetime] = Query(None, description="종료일"),
    limit: int = Query(100, ge=1, le=1000, description="조회 개수"),
    offset: int = Query(0, ge=0, description="오프셋"),
    session: AsyncSession = Depends(get_async_session)
):
    """주가 데이터 조회"""
    try:
        # 서비스 초기화
        stock_price_service = StockPriceService(session)
        
        # 쿼리 요청 생성
        query_request = StockPriceQueryRequest(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )
        
        # 데이터 조회
        prices = await stock_price_service.query_stock_prices(query_request)
        
        logger.info(
            "Stock prices retrieved",
            symbol=symbol,
            count=len(prices)
        )
        
        return prices
    
    except Exception as e:
        logger.error(
            "Failed to get stock prices",
            error=str(e),
            symbol=symbol
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{symbol}/latest",
    response_model=StockPriceResponse,
    summary="최신 주가 데이터 조회",
    description="특정 종목의 최신 주가 데이터를 조회합니다."
)
async def get_latest_stock_price(
    symbol: str = Path(..., description="종목코드"),
    interval: str = Query("1d", description="시간간격"),
    session: AsyncSession = Depends(get_async_session)
):
    """최신 주가 데이터 조회"""
    try:
        # 서비스 초기화
        stock_price_service = StockPriceService(session)
        
        # 최신 데이터 조회
        price = await stock_price_service.get_latest_stock_price(symbol, interval)
        
        if not price:
            raise HTTPException(
                status_code=404,
                detail=f"No price data found for symbol {symbol} with interval {interval}"
            )
        
        logger.info(
            "Latest stock price retrieved",
            symbol=symbol,
            interval=interval
        )
        
        return price
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get latest stock price",
            error=str(e),
            symbol=symbol
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{symbol}/statistics",
    response_model=dict,
    summary="주가 통계 정보 조회",
    description="특정 종목의 주가 통계 정보를 조회합니다."
)
async def get_stock_price_statistics(
    symbol: str = Path(..., description="종목코드"),
    interval: str = Query("1d", description="시간간격"),
    start_date: datetime = Query(..., description="시작일"),
    end_date: datetime = Query(..., description="종료일"),
    session: AsyncSession = Depends(get_async_session)
):
    """주가 통계 정보 조회"""
    try:
        # 서비스 초기화
        stock_price_service = StockPriceService(session)
        
        # 통계 정보 조회
        stats = await stock_price_service.get_price_statistics(
            stock_code=symbol,
            interval_unit=interval,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(
            "Stock price statistics retrieved",
            symbol=symbol,
            interval=interval
        )
        
        return {
            "symbol": symbol,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "statistics": stats
        }
    
    except Exception as e:
        logger.error(
            "Failed to get stock price statistics",
            error=str(e),
            symbol=symbol
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/stocks/search",
    response_model=List[StockResponse],
    summary="종목 검색",
    description="종목명으로 종목을 검색합니다."
)
async def search_stocks(
    q: str = Query(..., description="검색어"),
    skip: int = Query(0, ge=0, description="오프셋"),
    limit: int = Query(100, ge=1, le=1000, description="조회 개수"),
    session: AsyncSession = Depends(get_async_session)
):
    """종목 검색"""
    try:
        # 서비스 초기화
        stock_service = StockService(session)
        
        # 종목 검색
        stocks = await stock_service.search_stocks(q, skip, limit)
        
        logger.info(
            "Stocks searched",
            query=q,
            count=len(stocks)
        )
        
        return stocks
    
    except Exception as e:
        logger.error(
            "Failed to search stocks",
            error=str(e),
            query=q
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/stocks/market/{market}",
    response_model=List[StockResponse],
    summary="시장별 종목 조회",
    description="특정 시장의 종목 목록을 조회합니다."
)
async def get_stocks_by_market(
    market: str = Path(..., description="시장구분"),
    skip: int = Query(0, ge=0, description="오프셋"),
    limit: int = Query(100, ge=1, le=1000, description="조회 개수"),
    session: AsyncSession = Depends(get_async_session)
):
    """시장별 종목 조회"""
    try:
        # 서비스 초기화
        stock_service = StockService(session)
        
        # 시장별 종목 조회
        stocks = await stock_service.get_stocks_by_market(market, skip, limit)
        
        logger.info(
            "Stocks retrieved by market",
            market=market,
            count=len(stocks)
        )
        
        return stocks
    
    except Exception as e:
        logger.error(
            "Failed to get stocks by market",
            error=str(e),
            market=market
        )
        raise HTTPException(status_code=500, detail=str(e))
