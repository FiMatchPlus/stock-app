"""포트폴리오 분석 서비스 (MPT/CAPM) - 통합 버전

개요:
- 일별 가격에서 일별 수익률을 계산하고, 이를 바탕으로 연간 기준의 평균 수익과 동시 변동성을 추정합니다.
- 최소 변동성 포트폴리오와 최대 샤프 포트폴리오의 비중을 계산합니다.
- 벤치마크 비교와 고급 리스크 지표를 포함한 포트폴리오 분석을 제공합니다.
- 기본 지표부터 고급 리스크 지표까지 모든 분석 기능을 하나의 응답으로 제공합니다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import (
    AnalysisRequest,
    EnhancedAnalysisResponse,
    PortfolioWeights,
    EnhancedAnalysisMetrics,
    BenchmarkComparison,
)
from app.models.stock import StockPrice
from app.repositories.benchmark_repository import BenchmarkRepository
from app.repositories.risk_free_rate_repository import RiskFreeRateRepository
from app.utils.logger import get_logger


logger = get_logger(__name__)


class AnalysisService:
    """MPT/CAPM 기반 포트폴리오 분석 서비스"""

    def __init__(self):
        pass

    async def run_analysis(
        self,
        request: AnalysisRequest,
        session: AsyncSession,
    ) -> EnhancedAnalysisResponse:
        """분석 실행 (벤치마크 및 고급 지표 포함)

        모든 분석 기능을 포함하여 종합적인 포트폴리오 분석을 수행합니다.
        벤치마크가 없더라도 기본 지표는 모두 계산되어 반환됩니다.
        """

        lookback_end = datetime.utcnow()
        lookback_start = lookback_end - timedelta(days=252 * request.lookback_years)

        # Repository 초기화
        benchmark_repo = BenchmarkRepository(session)
        risk_free_repo = RiskFreeRateRepository(session)

        # 포트폴리오 가격 데이터 로드
        prices_df = await self._load_daily_prices(session, request, lookback_start, lookback_end)
        if prices_df.empty:
            return EnhancedAnalysisResponse(
                success=False,
                min_variance=PortfolioWeights(weights={}),
                max_sharpe=PortfolioWeights(weights={}),
                metrics={},
                risk_free_rate_used=0.0,
                analysis_period={"start": lookback_start, "end": lookback_end},
                notes="No price data available for requested holdings."
            )

        # 벤치마크 및 무위험수익률 조회
        benchmark_code = await self._determine_benchmark(request, benchmark_repo)
        benchmark_returns = await benchmark_repo.get_benchmark_returns_series(
            benchmark_code, lookback_start, lookback_end
        )
        risk_free_rate = await self._get_risk_free_rate(request, risk_free_repo, lookback_end)

        # 포트폴리오 수익률 계산
        returns_df = prices_df.pct_change().fillna(0.0)
        
        # 벤치마크와 시계열 동기화
        if not benchmark_returns.empty:
            returns_df, benchmark_returns = self._synchronize_time_series(returns_df, benchmark_returns)

        # 기본 통계 계산
        expected_returns = returns_df.mean() * 252.0
        cov_matrix = returns_df.cov() * 252.0

        # 포트폴리오 최적화
        min_var_w = self._min_variance_weights(cov_matrix)
        max_sharpe_w = self._max_sharpe_weights(expected_returns, cov_matrix, risk_free_rate)

        # 성과 지표 계산 (모든 지표 포함)
        mv_metrics = await self._calculate_enhanced_metrics(
            returns_df, expected_returns, cov_matrix, min_var_w, 
            benchmark_returns, risk_free_rate
        )
        ms_metrics = await self._calculate_enhanced_metrics(
            returns_df, expected_returns, cov_matrix, max_sharpe_w, 
            benchmark_returns, risk_free_rate
        )
        
        # 벤치마크 비교 분석
        benchmark_comparison = await self._calculate_benchmark_comparison(
            returns_df, max_sharpe_w, benchmark_returns, benchmark_code
        ) if benchmark_code and not benchmark_returns.empty else None
        
        metrics = {
            "min_variance": mv_metrics,
            "max_sharpe": ms_metrics,
        }

        return EnhancedAnalysisResponse(
            success=True,
            min_variance=PortfolioWeights(weights=min_var_w.to_dict()),
            max_sharpe=PortfolioWeights(weights=max_sharpe_w.to_dict()),
            metrics=metrics,
            benchmark_comparison=benchmark_comparison,
            risk_free_rate_used=risk_free_rate,
            analysis_period={"start": lookback_start, "end": lookback_end},
            notes=f"Analysis based on {benchmark_code} benchmark and {request.lookback_years}-year lookback period." if benchmark_code else f"Analysis based on {request.lookback_years}-year lookback period."
        )

    async def _load_daily_prices(
        self,
        session: AsyncSession,
        request: AnalysisRequest,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """선택 종목들의 일별 종가를 날짜 x 종목 형태의 표로 만듭니다.

        이 표를 기반으로 이후 단계에서 일별 수익률을 계산합니다.
        """
        from sqlalchemy import select, and_, asc

        symbols = [h.stock_code for h in request.holdings]
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
        return pivot

    def _min_variance_weights(self, cov: pd.DataFrame) -> pd.Series:
        """단순 최소 변동성 비중 근사

        이론적으로는 공분산 행렬을 이용한 정밀 해가 있지만, 여기서는 쉬운 근사를 사용합니다.
        변동성이 낮은 자산에 더 큰 비중을 주는 방식으로, 각 자산의 변동성 정보를 바탕으로 비중을 정규화합니다.
        """
        diag_var = np.diag(cov.values)
        inv_var = np.where(diag_var > 0, 1.0 / diag_var, 0.0)
        w = inv_var / inv_var.sum() if inv_var.sum() > 0 else np.ones_like(inv_var) / len(inv_var)
        return pd.Series(w, index=cov.columns)

    def _max_sharpe_weights(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        rf: float,
    ) -> pd.Series:
        """단순 최대 샤프 비중 근사

        이론적으로는 공분산 행렬과 초과수익을 이용한 해가 있지만, 여기서는 직관적인 점수 기반 근사를 사용합니다.
        위험 대비 기대수익이 높은 자산일수록 더 높은 점수를 부여하고, 그 점수를 정규화해 비중으로 사용합니다.
        """
        diag_var = np.diag(cov.values)
        excess = (mu.values - rf)
        score = np.where(diag_var > 0, excess / diag_var, 0.0)
        score = np.maximum(score, 0.0)
        if score.sum() == 0:
            score = np.ones_like(score)
        w = score / score.sum()
        return pd.Series(w, index=cov.columns)

    # 향상된 분석 메서드들
    async def _determine_benchmark(
        self, 
        request: AnalysisRequest, 
        benchmark_repo: BenchmarkRepository
    ) -> str:
        """적절한 벤치마크 결정"""
        if request.benchmark:
            # 사용자가 지정한 벤치마크가 있는지 확인
            available_benchmarks = await benchmark_repo.get_available_benchmarks()
            if request.benchmark in available_benchmarks:
                return request.benchmark
            else:
                logger.warning(f"Requested benchmark {request.benchmark} not available, falling back to KOSPI")
        
        # 기본적으로 KOSPI 사용
        return "KOSPI"

    async def _get_risk_free_rate(
        self,
        request: AnalysisRequest,
        risk_free_repo: RiskFreeRateRepository,
        target_date: datetime
    ) -> float:
        """무위험수익률 조회"""
        if request.risk_free_rate is not None:
            return request.risk_free_rate
        
        # 자동으로 CD91 금리 조회
        rate = await risk_free_repo.get_risk_free_rate("CD91", target_date)
        if rate is not None:
            return rate
        
        # CD91이 없으면 BOK 기준금리 시도
        rate = await risk_free_repo.get_risk_free_rate("BOK_BASE", target_date)
        if rate is not None:
            return rate
        
        logger.warning("No risk-free rate data available, using 0.0")
        return 0.0

    def _synchronize_time_series(
        self, 
        returns_df: pd.DataFrame, 
        benchmark_returns: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """포트폴리오와 벤치마크 수익률 시계열 동기화"""
        if benchmark_returns.empty:
            logger.warning("Empty benchmark returns, returning original portfolio returns")
            return returns_df, pd.Series(dtype=float)
        
        # 공통 날짜 찾기
        common_dates = returns_df.index.intersection(benchmark_returns.index)
        
        if len(common_dates) == 0:
            logger.warning("No common dates between portfolio and benchmark")
            return returns_df, pd.Series(dtype=float)
        
        # 공통 날짜로 필터링
        synced_returns = returns_df.loc[common_dates]
        synced_benchmark = benchmark_returns.loc[common_dates]
        
        logger.info(f"Synchronized {len(common_dates)} trading days between portfolio and benchmark")
        return synced_returns, synced_benchmark

    async def _calculate_enhanced_metrics(
        self,
        returns_df: pd.DataFrame,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        weights: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float,
    ) -> EnhancedAnalysisMetrics:
        """향상된 성과 지표 계산"""
        
        # 포트폴리오 수익률 계산
        portfolio_returns = returns_df.dot(weights)
        
        # 기본 지표
        expected_return = float(np.dot(weights.values, expected_returns.values))
        variance = float(np.dot(weights.values, np.dot(cov_matrix.values, weights.values)))
        std_deviation = float(np.sqrt(max(variance, 0.0)))
        
        # 벤치마크가 있는 경우 고급 지표 계산
        if not benchmark_returns.empty and len(benchmark_returns) > 0:
            beta, alpha, correlation = self._calculate_beta_alpha(
                portfolio_returns, benchmark_returns, risk_free_rate
            )
            tracking_error = self._calculate_tracking_error(portfolio_returns, benchmark_returns)
            upside_beta, downside_beta = self._calculate_upside_downside_beta(
                portfolio_returns, benchmark_returns, benchmark_returns.mean()
            )
        else:
            # 벤치마크 데이터가 없는 경우 기본값
            beta = 1.0
            alpha = 0.0
            correlation = 0.0
            tracking_error = 0.0
            upside_beta = 1.0
            downside_beta = 1.0
        
        # 위험조정 수익률 지표
        sharpe_ratio = (expected_return - risk_free_rate) / std_deviation if std_deviation > 0 else 0.0
        treynor_ratio = (expected_return - risk_free_rate) / beta if beta != 0 else 0.0
        
        # 하방편차 및 소르티노 비율
        downside_deviation = self._calculate_downside_deviation(portfolio_returns)
        sortino_ratio = (expected_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
        
        # 최대 낙폭 및 칼마 비율
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        calmar_ratio = expected_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # 정보비율 (벤치마크 대비 초과수익 / 트래킹에러)
        information_ratio = (expected_return - benchmark_returns.mean() * 252.0) / tracking_error if tracking_error > 0 else 0.0
        
        # 젠센 알파 (CAPM 기준)
        benchmark_annual_return = benchmark_returns.mean() * 252.0
        capm_expected_return = risk_free_rate + beta * (benchmark_annual_return - risk_free_rate)
        jensen_alpha = expected_return - capm_expected_return

        return EnhancedAnalysisMetrics(
            expected_return=expected_return,
            std_deviation=std_deviation,
            beta=beta,
            alpha=alpha,
            jensen_alpha=jensen_alpha,
            tracking_error=tracking_error,
            sharpe_ratio=sharpe_ratio,
            treynor_ratio=treynor_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            max_drawdown=max_drawdown,
            downside_deviation=downside_deviation,
            upside_beta=upside_beta,
            downside_beta=downside_beta,
            correlation_with_benchmark=correlation,
        )

    def _calculate_beta_alpha(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float
    ) -> Tuple[float, float, float]:
        """베타, 알파, 상관관계 계산"""
        try:
            # 초과수익률 계산
            portfolio_excess = portfolio_returns - risk_free_rate / 252.0
            benchmark_excess = benchmark_returns - risk_free_rate / 252.0
            
            # 베타 계산 (공분산 / 벤치마크 분산)
            covariance = np.cov(portfolio_excess, benchmark_excess)[0, 1]
            benchmark_variance = np.var(benchmark_excess)
            
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            # 알파 계산 (회귀 상수항)
            alpha = portfolio_excess.mean() - beta * benchmark_excess.mean()
            
            # 상관관계 계산
            correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            return float(beta), float(alpha) * 252.0, float(correlation)  # 알파는 연환산
            
        except Exception as e:
            logger.error(f"Error calculating beta/alpha: {str(e)}")
            return 1.0, 0.0, 0.0

    def _calculate_tracking_error(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """트래킹 에러 계산"""
        try:
            excess_returns = portfolio_returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)  # 연환산
            return float(tracking_error)
        except Exception as e:
            logger.error(f"Error calculating tracking error: {str(e)}")
            return 0.0

    def _calculate_upside_downside_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        benchmark_mean: float
    ) -> Tuple[float, float]:
        """상승/하락 베타 계산"""
        try:
            # 벤치마크가 평균보다 높은 날과 낮은 날 분리
            upside_mask = benchmark_returns > benchmark_mean
            downside_mask = benchmark_returns < benchmark_mean
            
            upside_beta = 1.0
            downside_beta = 1.0
            
            if upside_mask.sum() > 1:
                port_up = portfolio_returns[upside_mask]
                bench_up = benchmark_returns[upside_mask]
                upside_beta = np.cov(port_up, bench_up)[0, 1] / np.var(bench_up) if np.var(bench_up) > 0 else 1.0
            
            if downside_mask.sum() > 1:
                port_down = portfolio_returns[downside_mask]
                bench_down = benchmark_returns[downside_mask]
                downside_beta = np.cov(port_down, bench_down)[0, 1] / np.var(bench_down) if np.var(bench_down) > 0 else 1.0
            
            return float(upside_beta), float(downside_beta)
            
        except Exception as e:
            logger.error(f"Error calculating upside/downside beta: {str(e)}")
            return 1.0, 1.0

    def _calculate_downside_deviation(
        self,
        returns: pd.Series,
        target_return: float = 0.0
    ) -> float:
        """하방편차 계산"""
        try:
            downside_returns = returns[returns < target_return / 252.0]
            if len(downside_returns) == 0:
                return 0.0
            downside_deviation = downside_returns.std() * np.sqrt(252)
            return float(downside_deviation)
        except Exception as e:
            logger.error(f"Error calculating downside deviation: {str(e)}")
            return 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """최대 낙폭 계산"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return float(drawdown.min())
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0

    async def _calculate_benchmark_comparison(
        self,
        returns_df: pd.DataFrame,
        portfolio_weights: pd.Series,
        benchmark_returns: pd.Series,
        benchmark_code: str
    ) -> Optional[BenchmarkComparison]:
        """벤치마크 비교 분석"""
        try:
            if benchmark_returns.empty:
                return None
            
            # 포트폴리오 수익률
            portfolio_returns = returns_df.dot(portfolio_weights)
            
            # 벤치마크 통계
            benchmark_annual_return = benchmark_returns.mean() * 252.0
            benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
            
            # 포트폴리오 통계
            portfolio_annual_return = portfolio_returns.mean() * 252.0
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            
            # 초과 성과
            excess_return = portfolio_annual_return - benchmark_annual_return
            relative_volatility = portfolio_volatility / benchmark_volatility if benchmark_volatility > 0 else 1.0
            
            # 성과 기여도 분석 (단순화된 버전)
            security_selection = excess_return * 0.7  # 대략적인 추정
            timing_effect = excess_return * 0.3
            
            return BenchmarkComparison(
                benchmark_code=benchmark_code,
                benchmark_return=float(benchmark_annual_return),
                benchmark_volatility=float(benchmark_volatility),
                excess_return=float(excess_return),
                relative_volatility=float(relative_volatility),
                security_selection=float(security_selection),
                timing_effect=float(timing_effect)
            )
            
        except Exception as e:
            logger.error(f"Error calculating benchmark comparison: {str(e)}")
            return None


