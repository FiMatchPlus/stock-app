"""백테스트 벤치마크 관련 서비스"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.schemas import BacktestRequest, BenchmarkMetrics
from app.services.benchmark_service import BenchmarkService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestBenchmarkService:
    """백테스트 벤치마크 관련 서비스"""
    
    async def _execute_benchmark_operations(
        self,
        session: AsyncSession,
        benchmark_service: BenchmarkService,
        benchmark_code: str,
        start_date: datetime,
        end_date: datetime
    ):
        """벤치마크 관련 연산 수행"""
        benchmark_returns = await benchmark_service.get_benchmark_returns(
            benchmark_code=benchmark_code,
            start_date=start_date,
            end_date=end_date,
            session=session
        )
        
        benchmark_info = await benchmark_service.get_benchmark_info(
            benchmark_code=benchmark_code,
            session=session
        )
        
        return benchmark_returns, benchmark_info

    async def _get_benchmark_returns_for_period(
        self,
        request: BacktestRequest,
        session: AsyncSession
    ) -> Tuple[Optional[pd.Series], Optional[Dict[str, Any]]]:
        """벤치마크 수익률 조회 - 트랜잭션 에러 시 독립 세션으로 재시도"""
        try:
            benchmark_service = BenchmarkService()
            
            benchmark_code = request.benchmark_code or "KOSPI"
            
            benchmark_returns, benchmark_info = await self._retry_with_new_session_if_transaction_failed(
                operation_func=self._execute_benchmark_operations,
                session=session,
                service_name="benchmark",
                benchmark_service=benchmark_service,
                benchmark_code=benchmark_code,
                start_date=request.start,
                end_date=request.end
            )
            
            logger.info(
                "Benchmark returns retrieved for backtest",
                benchmark_code=benchmark_code,
                data_points=len(benchmark_returns),
                portfolio_size=len(request.holdings)
            )
            
            return benchmark_returns, benchmark_info
            
        except Exception as e:
            logger.error(f"Failed to get benchmark returns: {str(e)}")
            raise Exception(f"Database error during benchmark retrieval: {str(e)}")

    async def _calculate_benchmark_metrics(
        self,
        request: BacktestRequest,
        result_summary: List[Dict[str, Any]],
        session: AsyncSession
    ) -> Optional[BenchmarkMetrics]:
        """벤치마크 성과 지표 계산"""
        try:
            benchmark_service = BenchmarkService()
            
            benchmark_code = request.benchmark_code or "KOSPI"
            
            benchmark_returns = await benchmark_service.get_benchmark_returns(
                benchmark_code=benchmark_code,
                start_date=request.start,
                end_date=request.end,
                session=session
            )
            
            if benchmark_returns is None or benchmark_returns.empty:
                return None
            
            benchmark_total_return = float(((1 + benchmark_returns).cumprod()[-1] - 1.0) * 100)
            benchmark_volatility = float(benchmark_returns.std() * np.sqrt(252.0) * 100)
            
            benchmark_prices = await self._get_benchmark_prices_for_period(
                benchmark_code, request.start, request.end, session
            )
            
            benchmark_max_price = 0.0
            benchmark_min_price = 0.0
            if benchmark_prices is not None and not benchmark_prices.empty:
                benchmark_max_price = float(benchmark_prices.max())
                benchmark_min_price = float(benchmark_prices.min())
            
            portfolio_returns = []
            for rs in result_summary:
                daily_return = sum(stock['portfolio_contribution'] for stock in rs['stocks'])
                portfolio_returns.append(daily_return)
            
            portfolio_returns = np.array(portfolio_returns)
            
            if len(portfolio_returns) == len(benchmark_returns):
                alpha = float((portfolio_returns.mean() - benchmark_returns.mean()) * 252.0 * 100)
            else:
                alpha = 0.0
            
            benchmark_daily_average = float(benchmark_returns.mean() * 100)
            
            return BenchmarkMetrics(
                benchmark_total_return=benchmark_total_return,
                benchmark_volatility=benchmark_volatility,
                benchmark_max_price=benchmark_max_price,
                benchmark_min_price=benchmark_min_price,
                alpha=alpha,
                benchmark_daily_average=benchmark_daily_average
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate benchmark metrics: {str(e)}")
            return None
    
    async def _get_benchmark_prices_for_period(
        self,
        benchmark_code: str,
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession
    ) -> Optional[pd.Series]:
        """벤치마크 지수 가격 조회"""
        try:
            from app.models.stock import BenchmarkPrice
            
            query = select(BenchmarkPrice).where(
                and_(
                    BenchmarkPrice.index_code == benchmark_code,
                    BenchmarkPrice.datetime >= start_date,
                    BenchmarkPrice.datetime <= end_date
                )
            ).order_by(BenchmarkPrice.datetime)
            
            result = await session.execute(query)
            benchmark_prices = result.scalars().all()
            
            if not benchmark_prices:
                return None
            
            dates = []
            prices = []
            for price in benchmark_prices:
                dates.append(price.datetime)
                prices.append(float(price.close_price))
            
            return pd.Series(prices, index=dates)
            
        except Exception as e:
            logger.error(f"Failed to get benchmark prices: {str(e)}")
            return None

