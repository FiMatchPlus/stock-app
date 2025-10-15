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
        
        pivot = pivot.ffill()
        
        if pivot.isnull().any().any():
            null_stocks = pivot.columns[pivot.isnull().any()].tolist()
            logger.warning(f"Forward fill 후에도 NaN 값이 남아있는 종목: {null_stocks}")
        
        return pivot

    def _synchronize_time_series(
        self, 
        prices_df: pd.DataFrame, 
        benchmark_returns: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """포트폴리오와 벤치마크 시계열 동기화"""
        if benchmark_returns.empty:
            logger.warning("벤치마크 수익률이 비어있음, 원본 포트폴리오 가격 반환")
            return prices_df, pd.Series(dtype=float)
        
        common_dates = prices_df.index.intersection(benchmark_returns.index)
        
        if len(common_dates) == 0:
            logger.warning("포트폴리오와 벤치마크 간 공통 날짜가 없음")
            return prices_df, pd.Series(dtype=float)
        
        synced_prices = prices_df.loc[common_dates]
        synced_benchmark = benchmark_returns.loc[common_dates]
        
        logger.info(f"포트폴리오와 벤치마크 간 {len(common_dates)}개 거래일 동기화 완료")
        return synced_prices, synced_benchmark

