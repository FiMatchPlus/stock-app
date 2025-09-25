"""포트폴리오 분석 서비스 (MPT/CAPM)

개요:
- 일별 가격에서 일별 수익률을 계산하고, 이를 바탕으로 연간 기준의 평균 수익과 동시 변동성을 추정합니다.
- 단순하고 직관적인 휴리스틱으로 최소 변동성 포트폴리오와 최대 샤프 포트폴리오의 비중을 근사합니다.
- 산출된 비중으로 기대수익, 위험도(표준편차), 샤프 비율, 시장 민감도(베타), 초과성과(젠센 알파)를 제공합니다.

참고:
- 본 구현은 설명과 초기 분석을 위해 안정적인 근사 방식을 사용하며, 복잡한 제약조건이나 정밀 최적화 대신 이해 용이성과 견고성을 우선합니다.
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
    AnalysisResponse,
    PortfolioWeights,
    AnalysisMetrics,
    EnhancedAnalysisMetrics,
    EnhancedAnalysisResponse,
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
        use_enhanced: bool = False,
    ) -> AnalysisResponse | EnhancedAnalysisResponse:
        """분석 실행 (벤치마크 연동 가능)

        Args:
            use_enhanced: True이면 실제 벤치마크와 무위험수익률을 사용한 고급 분석
        """

        lookback_end = datetime.utcnow()
        lookback_start = lookback_end - timedelta(days=252 * request.lookback_years)

        # Repository 초기화
        benchmark_repo = BenchmarkRepository(session)
        risk_free_repo = RiskFreeRateRepository(session)

        # 포트폴리오 가격 데이터 로드
        prices_df = await self._load_daily_prices(session, request, lookback_start, lookback_end)
        if prices_df.empty:
            base_response = AnalysisResponse(
                success=False,
                min_variance=PortfolioWeights(weights={}),
                max_sharpe=PortfolioWeights(weights={}),
                metrics={},
                notes="No price data available for requested holdings."
            )
            
            if use_enhanced:
                return EnhancedAnalysisResponse(
                    success=False,
                    min_variance=PortfolioWeights(weights={}),
                    max_sharpe=PortfolioWeights(weights={}),
                    metrics={},
                    risk_free_rate_used=0.0,
                    analysis_period={"start": lookback_start, "end": lookback_end},
                    notes="No price data available for requested holdings."
                )
            return base_response

        # 벤치마크 및 무위험수익률 조회 (고급 분석 모드인 경우)
        if use_enhanced:
            benchmark_code = await self._determine_benchmark(request, benchmark_repo)
            benchmark_returns = await benchmark_repo.get_benchmark_returns_series(
                benchmark_code, lookback_start, lookback_end
            )
            risk_free_rate = await self._get_risk_free_rate(request, risk_free_repo, lookback_end)
        else:
            benchmark_returns = pd.Series(dtype=float)
            risk_free_rate = request.risk_free_rate if request.risk_free_rate is not None else 0.0
            benchmark_code = None

        # 포트폴리오 수익률 계산
        returns_df = prices_df.pct_change().fillna(0.0)
        
        # 벤치마크와 시계열 동기화 (고급 분석 모드)
        if use_enhanced and not benchmark_returns.empty:
            returns_df, benchmark_returns = self._synchronize_time_series(returns_df, benchmark_returns)

        # 기본 통계 계산
        expected_returns = returns_df.mean() * 252.0
        cov_matrix = returns_df.cov() * 252.0

        # 포트폴리오 최적화
        min_var_w = self._min_variance_weights(cov_matrix)
        max_sharpe_w = self._max_sharpe_weights(expected_returns, cov_matrix, risk_free_rate)

        # 성과 지표 계산
        if use_enhanced:
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
            ) if benchmark_code else None
            
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
                notes=f"Analysis based on {benchmark_code} benchmark and {request.lookback_years}-year lookback period." if benchmark_code else None
            )
        else:
            # 기존 방식 (하위 호환성)
            mv_metrics = self._metrics_for_weights(returns_df, expected_returns, cov_matrix, min_var_w, risk_free_rate)
            ms_metrics = self._metrics_for_weights(returns_df, expected_returns, cov_matrix, max_sharpe_w, risk_free_rate)

            metrics = {
                "min_variance": mv_metrics,
                "max_sharpe": ms_metrics,
            }

            return AnalysisResponse(
                success=True,
                min_variance=PortfolioWeights(weights=min_var_w.to_dict()),
                max_sharpe=PortfolioWeights(weights=max_sharpe_w.to_dict()),
                metrics=metrics,
                notes="Using basic analysis mode. Use enhanced analysis for benchmark comparison."
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

    def _metrics_for_weights(
        self,
        returns_df: pd.DataFrame,
        mu: pd.Series,
        cov: pd.DataFrame,
        w: pd.Series,
        rf: float,
    ) -> AnalysisMetrics:
        # 포트폴리오 기대수익(연간 기준)을 계산합니다.
        exp_ret = float(np.dot(w.values, mu.values))
        # 포트폴리오의 위험도를 나타내는 분산과 표준편차를 계산합니다.
        variance = float(np.dot(w.values, np.dot(cov.values, w.values)))
        std_dev = float(np.sqrt(max(variance, 0.0)))
        # 샤프 비율은 위험 대비 초과수익을 의미합니다.
        sharpe = (exp_ret - rf) / std_dev if std_dev > 0 else 0.0

        # 간단한 베타 근사: 시장 대리변수(종목 평균 일간 수익률) 대비 민감도를 계산합니다.
        port_daily = returns_df.dot(w)
        market_proxy = returns_df.mean(axis=1)
        cov_pm = float(np.cov(port_daily, market_proxy)[0, 1])
        var_m = float(np.var(market_proxy))
        beta = cov_pm / var_m if var_m > 0 else 0.0

        # CAPM 요구수익률과 젠센 알파(요구수익을 초과하는 성과)를 계산합니다.
        req_return_capm = rf + beta * max(market_proxy.mean() * 252.0 - rf, 0.0)
        jensen_alpha = exp_ret - req_return_capm

        return AnalysisMetrics(
            expected_return=exp_ret,
            std_deviation=std_dev,
            beta=beta,
            sharpe_ratio=sharpe,
            jensen_alpha=jensen_alpha,
        )

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


