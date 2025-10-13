"""포트폴리오 분석 서비스 (MPT/CAPM) - 이동 윈도우 버전

개요:
- 3년 윈도우 크기로 1개월 간격으로 이동하며 포트폴리오 최적화 수행
- 최소 변동성 포트폴리오와 최대 샤프 포트폴리오의 비중을 각 시점별로 계산
- 백테스팅을 통해 전체 기간에 대한 성능 지표를 계산
- 최종 응답은 최근 시점의 비중과 백테스팅 기반 평균 성능 지표를 포함
- 벤치마크 비교와 고급 리스크 지표를 포함한 포트폴리오 분석을 제공합니다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
import time

from app.models.schemas import (
    AnalysisRequest,
    PortfolioAnalysisResponse,
    PortfolioData,
    AnalysisMetadata,
    BenchmarkInfo,
    AnalysisMetrics,
    BenchmarkComparison,
    StockDetails,
    BetaAnalysis,
)
from app.repositories.benchmark_repository import BenchmarkRepository
from app.repositories.risk_free_rate_repository import RiskFreeRateRepository
from app.services.risk_free_rate_calculator import RiskFreeRateCalculator
from app.utils.logger import get_logger
import numpy as np
from app.services.analysis.optimization_service import OptimizationService
from app.services.analysis.metrics_service import MetricsService
from app.services.analysis.data_service import DataService
from app.services.analysis.benchmark_service import BenchmarkService
from app.services.analysis.beta_service import BetaService
from app.services.analysis.compose_service import ComposeService


logger = get_logger(__name__)


class AnalysisService(OptimizationService, MetricsService, DataService, BenchmarkService, BetaService, ComposeService):
    """MPT/CAPM 기반 포트폴리오 분석 서비스 - 이동 윈도우 방식"""

    def __init__(self):
        self.window_years = 1      # 학습 윈도우 크기: 1년 (252 거래일)
        self.step_months = 0.5     # 윈도우 이동 간격: 2주 (10.5 거래일)
        self.backtest_months = 3   # 백테스팅 기간: 3개월 (63 거래일) - VaR/CVaR 계산용

    async def run_analysis(
        self,
        request: AnalysisRequest,
        session: AsyncSession,
    ) -> PortfolioAnalysisResponse:
        """이동 윈도우 기반 포트폴리오 분석 실행"""
        
        # 실행 시간 측정 시작
        start_time = time.time()

        # 전체 분석 기간 설정 (최소 5년 권장)
        analysis_end = datetime.utcnow()
        analysis_start = analysis_end - timedelta(days=252 * max(request.lookback_years, 5))

        # Repository 초기화
        benchmark_repo = BenchmarkRepository(session)
        risk_free_repo = RiskFreeRateRepository(session)

        # 전체 기간의 포트폴리오 가격 데이터 로드
        prices_df = await self._load_daily_prices(session, request, analysis_start, analysis_end)
        if prices_df.empty:
            execution_time = time.time() - start_time
            metadata = AnalysisMetadata(
                risk_free_rate_used=0.0,
                period={"start": analysis_start, "end": analysis_end},
                notes="No price data available for requested holdings.",
                execution_time=execution_time,
                portfolio_id=request.portfolio_id,
                timestamp=None,
            )
            return PortfolioAnalysisResponse(
                success=False,
                metadata=metadata,
                benchmark=None,
                portfolios=[],
                stock_details=None,
            )

        # 벤치마크 및 무위험수익률 조회
        benchmark_code = await self._determine_benchmark(request, benchmark_repo)
        benchmark_returns = await benchmark_repo.get_benchmark_returns_series(
            benchmark_code, analysis_start, analysis_end
        )

        # 무위험수익률 계산
        risk_free_calculator = RiskFreeRateCalculator(session)
        risk_free_rate = await risk_free_calculator.calculate_risk_free_rate(
            analysis_start, analysis_end, request.risk_free_rate
        )

        # 벤치마크와 시계열 동기화
        if not benchmark_returns.empty:
            prices_df, benchmark_returns = self._synchronize_time_series(prices_df, benchmark_returns)

        # 이동 윈도우 최적화 수행
        optimization_results = await self._perform_rolling_optimization(
            prices_df, benchmark_returns, risk_free_rate
        )

        # 백테스팅 기반 성능 지표 계산
        backtest_metrics = await self._calculate_backtest_metrics(
            optimization_results, benchmark_returns, risk_free_rate
        )

        # 벤치마크 비교 분석
        benchmark_comparison = await self._calculate_benchmark_comparison(
            optimization_results, benchmark_returns, benchmark_code
        ) if benchmark_code and not benchmark_returns.empty else None

        # 개별 종목 베타 계산
        stock_details = await self._calculate_stock_details(
            optimization_results, benchmark_code, benchmark_returns, session, prices_df
        )

        # 최종 응답 구성
        latest_weights = optimization_results['latest_weights']

        # 포트폴리오 데이터 구성
        portfolios = await self._build_portfolio_data(
            request,
            optimization_results,
            benchmark_returns,
            analysis_start,
            analysis_end,
            backtest_metrics,
            prices_df,
            risk_free_rate,
            benchmark_code,
        )

        # 벤치마크 정보 구성
        benchmark_info = None
        if not benchmark_returns.empty:
            benchmark_annual_return = benchmark_returns.mean() * 252.0
            benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
            benchmark_info = BenchmarkInfo(
                code=benchmark_code,
                benchmark_return=float(benchmark_annual_return),
                volatility=float(benchmark_volatility)
            )

        # 제약조건 위반 감지 (최신 윈도우 기준)
        capped_assets = []  # 상한에 걸린 종목
        floored_assets = []  # 하한에 걸린 종목
        cap_threshold = 0.9
        floor_threshold = 0.05  # 최소 비중 5%
        
        if 'latest_weights' in optimization_results:
            latest_ms = optimization_results['latest_weights'].get('max_sortino', {})
            latest_mv = optimization_results['latest_weights'].get('min_downside_risk', {})
            
            # 모든 종목 체크 (중복 제거)
            all_weights = {}
            for name, w in latest_ms.items():
                all_weights[('max_sortino', name)] = w
            for name, w in latest_mv.items():
                all_weights[('min_downside_risk', name)] = w
            
            for (portfolio_type, name), w in all_weights.items():
                try:
                    # 상한 체크 (90% 이상)
                    if w >= cap_threshold - 1e-9:
                        if name not in capped_assets:
                            capped_assets.append(name)
                    
                    # 하한 체크 (5% ± 0.1%)
                    if abs(w - floor_threshold) < 1e-3:
                        if name not in floored_assets:
                            floored_assets.append(name)
                except Exception:
                    continue

        # 메타데이터 구성
        prices_start = prices_df.index.min().isoformat() if not prices_df.empty else None
        prices_end = prices_df.index.max().isoformat() if not prices_df.empty else None
        bench_start = benchmark_returns.index.min().isoformat() if not benchmark_returns.empty else None
        bench_end = benchmark_returns.index.max().isoformat() if not benchmark_returns.empty else None
        total_windows = len(optimization_results.get('dates', []))

        base_notes = (
            f"benchmark={benchmark_code or 'N/A'}, "
            f"window_years={self.window_years}, step_months={self.step_months}, backtest_months={self.backtest_months}, windows={total_windows}, "
            f"prices_range=[{prices_start}..{prices_end}], benchmark_range=[{bench_start}..{bench_end}]"
        )
        
        # 제약조건 정보 추가
        constraint_notes = f", weight_constraints=[min=5%, max=90%]"
        
        # 상한/하한 위반 경고 추가
        cap_notes = f", weight_cap_applied=0.9, capped_assets={capped_assets}" if capped_assets else ""
        floor_notes = ""
        if floored_assets:
            floor_notes = f", weight_floor_applied=0.05, floored_assets={floored_assets}, WARNING: 일부 종목이 최소 비중 5%에 도달했습니다. 종목 조합을 다시 고려하세요."
        
        notes = base_notes + constraint_notes + cap_notes + floor_notes
        
        # 실행 시간 계산
        execution_time = time.time() - start_time
        logger.info(f"Portfolio analysis completed in {execution_time:.3f} seconds")

        metadata = AnalysisMetadata(
            risk_free_rate_used=risk_free_rate,
            period={"start": analysis_start, "end": analysis_end},
            notes=notes,
            execution_time=execution_time,
            portfolio_id=request.portfolio_id,
            timestamp=None
        )

        return PortfolioAnalysisResponse(
            success=True,
            metadata=metadata,
            benchmark=benchmark_info,
            portfolios=portfolios,
            stock_details=stock_details
        )


