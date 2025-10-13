from __future__ import annotations

from datetime import datetime
from typing import Tuple
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, asc

from app.models.stock import StockPrice
from app.utils.logger import get_logger


logger = get_logger(__name__)


class DataService:
    async def _load_daily_prices(
        self,
        session: AsyncSession,
        request,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """선택 종목들의 일별 종가를 날짜 x 종목 형태의 표로 만듭니다."""
        symbols = [h.code for h in request.holdings]
        if not symbols:
            return pd.DataFrame()

        stmt = (
            select(StockPrice)
            .where(
                and_(
                    StockPrice.stock_code.in_(symbols),
                    StockPrice.interval_unit == "1d",
                    StockPrice.datetime >= start_dt,
                    StockPrice.datetime <= end_dt,
                )
            )
            .order_by(asc(StockPrice.datetime))
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            [
                {
                    "datetime": r.datetime,
                    "stock_code": r.stock_code,
                    "close": float(r.close_price),
                }
                for r in rows
            ]
        )

        pivot = df.pivot_table(index="datetime", columns="stock_code", values="close", aggfunc="first")
        pivot = pivot.sort_index()
        
        # NaN 처리: forward fill (가장 최근 유효 가격으로 채우기)
        # 거래정지/상장폐지 등으로 최신 날짜 데이터가 없을 수 있음
        pivot = pivot.ffill()
        
        # 여전히 NaN이 있다면 (처음부터 데이터가 없는 경우) 경고
        if pivot.isnull().any().any():
            null_stocks = pivot.columns[pivot.isnull().any()].tolist()
            logger.warning(f"Stocks with remaining NaN values after forward fill: {null_stocks}")
        
        return pivot

    def _synchronize_time_series(
        self, 
        prices_df: pd.DataFrame, 
        benchmark_returns: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """포트폴리오와 벤치마크 시계열 동기화"""
        if benchmark_returns.empty:
            logger.warning("Empty benchmark returns, returning original portfolio prices")
            return prices_df, pd.Series(dtype=float)
        
        # 공통 날짜 찾기
        common_dates = prices_df.index.intersection(benchmark_returns.index)
        
        if len(common_dates) == 0:
            logger.warning("No common dates between portfolio and benchmark")
            return prices_df, pd.Series(dtype=float)
        
        # 공통 날짜로 필터링
        synced_prices = prices_df.loc[common_dates]
        synced_benchmark = benchmark_returns.loc[common_dates]
        
        logger.info(f"Synchronized {len(common_dates)} trading days between portfolio and benchmark")
        return synced_prices, synced_benchmark

