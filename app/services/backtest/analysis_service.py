"""백테스트 분석 관련 서비스"""

import time
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import BacktestRequest, PortfolioSnapshotResponse, HoldingSnapshotResponse
from app.services.analysis_service import AnalysisService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestAnalysisService:
    """백테스트 분석 관련 서비스"""
    
    def _create_portfolio_snapshot_response(
        self, 
        request: BacktestRequest, 
        portfolio_data: List[Dict[str, Any]], 
        portfolio_id: Optional[int] = None,
        actual_start: Optional[datetime] = None,
        actual_end: Optional[datetime] = None
    ) -> PortfolioSnapshotResponse:
        """포트폴리오 스냅샷 응답 생성"""
        if not portfolio_data:
            raise ValueError("포트폴리오 데이터가 없습니다.")
        
        if portfolio_id is None:
            portfolio_id = int(time.time())
        
        final_data = portfolio_data[-1]
        initial_data = portfolio_data[0]
        base_value = Decimal(str(initial_data['portfolio_value']))
        current_value = Decimal(str(final_data['portfolio_value']))
        
        holdings = []
        for i, holding in enumerate(request.holdings):
            holdings.append(HoldingSnapshotResponse(
                id=i + 1,
                stock_id=holding.code,
                quantity=holding.quantity
            ))
        
        effective_start = actual_start if actual_start else request.start
        effective_end = actual_end if actual_end else request.end
        
        return PortfolioSnapshotResponse(
            id=12345,
            portfolio_id=portfolio_id,
            base_value=base_value,
            current_value=current_value,
            start_at=effective_start,
            end_at=effective_end,
            created_at=datetime.utcnow(),
            execution_time=0.0,
            holdings=holdings
        )

    async def analyze_backtest_performance(
        self,
        portfolio_returns: pd.Series,
        benchmark_code: str,
        session: AsyncSession,
        risk_free_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """백테스트 결과에 대한 고도화된 성과 분석"""
        
        from app.repositories.benchmark_repository import BenchmarkRepository
        from app.repositories.risk_free_rate_repository import RiskFreeRateRepository
        
        benchmark_repo = BenchmarkRepository(session)
        risk_free_repo = RiskFreeRateRepository(session)
        
        start_date = portfolio_returns.index[0]
        end_date = portfolio_returns.index[-1]
        
        benchmark_returns = await benchmark_repo.get_benchmark_returns_series(
            benchmark_code, start_date, end_date
        )
        
        if risk_free_rate is None:
            risk_free_rate = await risk_free_repo.get_risk_free_rate("CD91", end_date) or 0.0
        
        if not benchmark_returns.empty:
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_synced = portfolio_returns.loc[common_dates]
            benchmark_synced = benchmark_returns.loc[common_dates]
        else:
            portfolio_synced = portfolio_returns
            benchmark_synced = pd.Series(dtype=float)
        
        analysis_service = AnalysisService()
        
        annual_return = portfolio_synced.mean() * 252
        annual_volatility = portfolio_synced.std() * np.sqrt(252)
        
        sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0
        
        max_dd = analysis_service._calculate_max_drawdown(portfolio_synced)
        downside_dev = analysis_service._calculate_downside_deviation(portfolio_synced)
        sortino = (annual_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0
        
        metrics = {
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_volatility),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'calmar_ratio': float(calmar),
            'max_drawdown': float(max_dd),
            'downside_deviation': float(downside_dev),
            'risk_free_rate_used': float(risk_free_rate)
        }
        
        if not benchmark_synced.empty:
            beta, alpha, correlation = analysis_service._calculate_beta_alpha(
                portfolio_synced, benchmark_synced, risk_free_rate
            )
            tracking_error = analysis_service._calculate_tracking_error(portfolio_synced, benchmark_synced)
            
            benchmark_annual_return = benchmark_synced.mean() * 252
            excess_return = annual_return - benchmark_annual_return
            
            metrics.update({
                'beta': float(beta),
                'alpha': float(alpha),
                'jensen_alpha': float(alpha),
                'tracking_error': float(tracking_error),
                'correlation_with_benchmark': float(correlation),
                'benchmark_annual_return': float(benchmark_annual_return),
                'excess_return': float(excess_return)
            })
        
        return metrics

