"""백테스트 데이터 조회 및 전처리 서비스"""

from datetime import datetime, timedelta
from typing import Tuple
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.stock import StockPrice
from app.models.schemas import BacktestRequest
from app.exceptions import MissingStockPriceDataException
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestDataService:
    """백테스트 데이터 조회 및 전처리 서비스"""
    
    def _adjust_to_trading_days(
        self, 
        stock_prices: pd.DataFrame, 
        start_date: datetime, 
        end_date: datetime
    ) -> Tuple[datetime, datetime]:
        """실제 거래일 기준으로 시작/종료 날짜 조정"""
        if stock_prices.empty:
            return start_date, end_date
            
        trading_dates = stock_prices['datetime'].unique()
        trading_dates = pd.to_datetime(trading_dates)
        
        actual_start = trading_dates[trading_dates >= start_date]
        if len(actual_start) > 0:
            adjusted_start = actual_start.min()
        else:
            adjusted_start = start_date
            
        actual_end = trading_dates[trading_dates <= end_date]
        if len(actual_end) > 0:
            adjusted_end = actual_end.max()
        else:
            adjusted_end = end_date
            
        return adjusted_start, adjusted_end

    async def _get_stock_prices_optimized(
        self, 
        request: BacktestRequest, 
        session: AsyncSession
    ) -> pd.DataFrame:
        """최적화된 주가 데이터 조회 (휴장일 고려)"""
        stock_codes = [holding.code for holding in request.holdings]
        
        extended_start = request.start - timedelta(days=10)
        extended_end = request.end + timedelta(days=10)
        
        query = select(StockPrice).where(
            and_(
                StockPrice.stock_code.in_(stock_codes),
                StockPrice.datetime >= extended_start,
                StockPrice.datetime <= extended_end,
                StockPrice.interval_unit == "1d"
            )
        ).order_by(StockPrice.datetime)
        
        result = await session.execute(query)
        stock_prices = result.scalars().all()
        
        if stock_prices is None or len(stock_prices) == 0:
            return pd.DataFrame()
        
        data = []
        for price in stock_prices:
            data.append({
                'stock_code': price.stock_code,
                'datetime': price.datetime,
                'close_price': price.close_price,
                'open_price': price.open_price,
                'high_price': price.high_price,
                'low_price': price.low_price,
                'volume': price.volume,
                'change_rate': price.change_rate
            })
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            adjusted_start, adjusted_end = self._adjust_to_trading_days(df, request.start, request.end)
            
            df = df[(df['datetime'] >= adjusted_start) & (df['datetime'] <= adjusted_end)]
            
            logger.info(f"Adjusted trading period: {adjusted_start.strftime('%Y-%m-%d')} ~ {adjusted_end.strftime('%Y-%m-%d')} "
                       f"(original: {request.start.strftime('%Y-%m-%d')} ~ {request.end.strftime('%Y-%m-%d')})")
        
        if not df.empty:
            available_stocks = set(df['stock_code'].unique())
            missing_stocks = []
            
            for stock_code in stock_codes:
                if stock_code not in available_stocks:
                    available_range_query = select(StockPrice).where(
                        and_(
                            StockPrice.stock_code == stock_code,
                            StockPrice.interval_unit == "1d"
                        )
                    ).order_by(StockPrice.datetime)
                    
                    range_result = await session.execute(available_range_query)
                    available_prices = range_result.scalars().all()
                    
                    available_range = None
                    if available_prices:
                        start_date = min(price.datetime for price in available_prices).strftime('%Y-%m-%d')
                        end_date = max(price.datetime for price in available_prices).strftime('%Y-%m-%d')
                        available_range = f"{start_date} ~ {end_date}"
                    
                    missing_stocks.append({
                        'stock_code': stock_code,
                        'start_date': request.start.strftime('%Y-%m-%d'),
                        'end_date': request.end.strftime('%Y-%m-%d'),
                        'available_date_range': available_range
                    })
            
            if missing_stocks:
                requested_period = f"{request.start.strftime('%Y-%m-%d')} ~ {request.end.strftime('%Y-%m-%d')}"
                raise MissingStockPriceDataException(
                    missing_stocks=missing_stocks,
                    requested_period=requested_period,
                    total_stocks=len(stock_codes)
                )
        
        df = await self._preprocess_stock_data(df)
        
        return df
    
    async def _preprocess_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """주가 데이터 전처리"""
        if df is None or df.empty:
            return df
        
        df = df.sort_values(['datetime', 'stock_code'])
        
        df['close_price'] = df.groupby('stock_code')['close_price'].ffill()
        
        df['returns'] = df.groupby('stock_code')['close_price'].pct_change()
        
        df['returns'] = df['returns'].fillna(0)
        
        return df

