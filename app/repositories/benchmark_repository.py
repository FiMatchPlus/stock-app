"""벤치마크 가격 데이터 리포지토리"""

from typing import List, Optional, Dict
from datetime import datetime
from decimal import Decimal
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, asc

from app.models.stock import BenchmarkPrice
from app.repositories.base import BaseRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkRepository(BaseRepository[BenchmarkPrice]):
    """벤치마크 가격 데이터 리포지토리"""

    def __init__(self, session: AsyncSession):
        super().__init__(BenchmarkPrice, session)

    async def get_benchmark_prices(
        self,
        index_codes: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """지정된 기간의 벤치마크 가격 데이터 조회"""
        try:
            stmt = (
                select(BenchmarkPrice)
                .where(
                    and_(
                        BenchmarkPrice.index_code.in_(index_codes),
                        BenchmarkPrice.datetime >= start_date,
                        BenchmarkPrice.datetime <= end_date
                    )
                )
                .order_by(asc(BenchmarkPrice.datetime))
            )
            
            result = await self.session.execute(stmt)
            benchmark_prices = result.scalars().all()
            
            if not benchmark_prices:
                logger.warning(f"No benchmark data found for {index_codes} between {start_date} and {end_date}")
                return pd.DataFrame()

            # DataFrame으로 변환
            data = []
            for price in benchmark_prices:
                data.append({
                    'index_code': price.index_code,
                    'datetime': price.datetime,
                    'close_price': float(price.close_price),
                    'open_price': float(price.open_price),
                    'high_price': float(price.high_price),
                    'low_price': float(price.low_price),
                    'change_amount': float(price.change_amount),
                    'change_rate': float(price.change_rate),
                    'volume': price.volume,
                    'trading_value': float(price.trading_value) if price.trading_value else 0,
                    'market_cap': float(price.market_cap) if price.market_cap else None
                })

            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            logger.info(f"Retrieved {len(df)} benchmark price records for {len(index_codes)} indices")
            return df

        except Exception as e:
            logger.error(f"Error retrieving benchmark prices: {str(e)}")
            raise

    async def get_available_benchmarks(self) -> List[str]:
        """사용 가능한 벤치마크 지수 목록 조회"""
        try:
            stmt = select(BenchmarkPrice.index_code).distinct()
            result = await self.session.execute(stmt)
            return [code for (code,) in result.fetchall()]

        except Exception as e:
            logger.error(f"Error retrieving available benchmarks: {str(e)}")
            raise

    async def get_benchmark_returns_series(
        self,
        index_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.Series:
        """특정 벤치마크의 수익률 시계열 조회"""
        try:
            df = await self.get_benchmark_prices([index_code], start_date, end_date)
            
            if df.empty:
                return pd.Series(dtype=float)
            
            # 벤치마크별로 수익률 계산
            df = df[df['index_code'] == index_code].sort_values('datetime')
            df['returns'] = df['close_price'].pct_change().fillna(0)
            
            # 날짜를 인덱스로 하는 수익률 시리즈 반환
            returns_series = pd.Series(
                data=df['returns'].values,
                index=df['datetime'],
                name=f'{index_code}_returns'
            )
            
            return returns_series

        except Exception as e:
            logger.error(f"Error calculating benchmark returns for {index_code}: {str(e)}")
            raise
