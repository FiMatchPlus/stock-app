"""백테스트 서비스 - 최적화된 포트폴리오 백테스트 엔진"""

import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import BacktestRequest, BacktestResponse
from app.exceptions import MissingStockPriceDataException, InsufficientDataException
from app.utils.logger import get_logger

from app.services.backtest.data_service import BacktestDataService
from app.services.backtest.calculation_service import BacktestCalculationService
from app.services.backtest.metrics_service import BacktestMetricsService
from app.services.backtest.benchmark_service import BacktestBenchmarkService
from app.services.backtest.risk_free_rate_service import BacktestRiskFreeRateService
from app.services.backtest.execution_service import BacktestExecutionService
from app.services.backtest.analysis_service import BacktestAnalysisService

logger = get_logger(__name__)


class BacktestService(
    BacktestDataService,
    BacktestCalculationService,
    BacktestMetricsService,
    BacktestBenchmarkService,
    BacktestRiskFreeRateService,
    BacktestExecutionService,
    BacktestAnalysisService
):
    """최적화된 백테스트 서비스"""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    async def _retry_with_new_session_if_transaction_failed(
        self, 
        operation_func, 
        session: AsyncSession,
        service_name: str,
        **kwargs
    ):
        """
        트랜잭션 에러 감지시 새 독립 세션으로 재시도
        
        Args:
            operation_func: 실행할 함수 (session을 첫 번째 인자로 받음)
            session: 원본 세션 
            service_name: 로깅용 서비스명
            **kwargs: operation_func에 전달할 추가 인자들
        """
        try:
            return await operation_func(session, **kwargs)
        except Exception as e:
            if "InFailedSQLTransaction" in str(e) or "transaction is aborted" in str(e):
                logger.warning(f"Primary session failed for {service_name}, retrying with fresh session", error=str(e))
                from app.models.database import AsyncSessionLocal
                async with AsyncSessionLocal() as new_session:
                    return await operation_func(new_session, **kwargs)
            else:
                raise
        
    async def run_backtest(
        self, 
        request: BacktestRequest, 
        session: Optional[AsyncSession] = None,
        portfolio_id: Optional[int] = None
    ) -> BacktestResponse:
        """백테스트 실행"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting backtest for {len(request.holdings)} holdings", 
                       start=request.start.isoformat(), end=request.end.isoformat())
            
            if session is None:
                raise ValueError("Session is required for stock price data retrieval")
            stock_prices = await self._get_stock_prices_optimized(
                request, session
            )
            
            if stock_prices is None or stock_prices.empty:
                missing_stocks = []
                for holding in request.holdings:
                    missing_stocks.append({
                        'stock_code': holding.code,
                        'start_date': request.start.strftime('%Y-%m-%d'),
                        'end_date': request.end.strftime('%Y-%m-%d'),
                        'available_date_range': None
                    })
                
                requested_period = f"{request.start.strftime('%Y-%m-%d')} ~ {request.end.strftime('%Y-%m-%d')}"
                raise MissingStockPriceDataException(
                    missing_stocks=missing_stocks,
                    requested_period=requested_period,
                    total_stocks=len(request.holdings)
                )
            
            portfolio_data, result_summary, execution_logs, final_status, benchmark_info, actual_start, actual_end = await self._execute_backtest(
                request, stock_prices, session
            )
            
            metrics = await self._calculate_metrics(result_summary)
            
            benchmark_metrics = await self._calculate_benchmark_metrics(
                request, result_summary, session
            )
            
            risk_free_rate_info = {
                'rate_type': 'AUTO',
                'source': 'backtest_execution'
            }
            
            portfolio_snapshot = self._create_portfolio_snapshot_response(
                request, 
                portfolio_data, 
                portfolio_id,
                actual_start,
                actual_end
            )
            
            execution_time = time.time() - start_time
            
            response = BacktestResponse(
                portfolio_snapshot=portfolio_snapshot,
                metrics=metrics,
                benchmark_metrics=benchmark_metrics,
                result_summary=result_summary,
                execution_time=execution_time,
                execution_logs=execution_logs,
                result_status=final_status,
                benchmark_info=benchmark_info,
                risk_free_rate_info=risk_free_rate_info,
                backtest_id=request.backtest_id
            )
            
            logger.info(
                "Backtest completed successfully",
                portfolio_id=portfolio_id,
                execution_time=f"{execution_time:.3f}s",
                total_return=str(metrics.total_return),
                final_status=final_status
            )
            
            return response
            
        except MissingStockPriceDataException as e:
            logger.error(f"Missing stock price data: {str(e)}")
            raise
        except InsufficientDataException as e:
            logger.error(f"Insufficient data: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}", exc_info=True)
            raise
