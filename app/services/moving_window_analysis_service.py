"""이동 윈도우 기반 MPT 포트폴리오 최적화 및 백테스팅 분석 서비스

개요:
- 3년 윈도우 크기로 1개월 간격으로 이동하며 포트폴리오 최적화 수행
- 최소 변동성 포트폴리오와 최대 샤프 포트폴리오의 비중을 각 시점별로 계산
- 백테스팅을 통해 전체 기간에 대한 성능 지표를 계산
- 최종 응답은 최근 시점의 비중과 백테스팅 기반 평균 성능 지표를 포함
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from scipy.optimize import minimize
from scipy.stats import linregress

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
from app.models.stock import StockPrice
from app.repositories.benchmark_repository import BenchmarkRepository
from app.repositories.risk_free_rate_repository import RiskFreeRateRepository
from app.services.risk_free_rate_service import RiskFreeRateService
from app.utils.logger import get_logger
from app.services.analysis.optimization_mixin import OptimizationMixin
from app.services.analysis.metrics_mixin import MetricsMixin
from app.services.analysis.data_mixin import DataMixin
from app.services.analysis.benchmark_mixin import BenchmarkMixin
from app.services.analysis.beta_mixin import BetaMixin
from app.services.analysis.compose_mixin import ComposeMixin


logger = get_logger(__name__)


class MovingWindowAnalysisService(OptimizationMixin, MetricsMixin, DataMixin, BenchmarkMixin, BetaMixin, ComposeMixin):
    """Deprecated: Use AnalysisService instead."""
    pass

    async def _load_daily_prices(
        self,
        session: AsyncSession,
        request: AnalysisRequest,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """선택 종목들의 일별 종가를 날짜 x 종목 형태의 표로 만듭니다."""
        from sqlalchemy import select, and_, asc

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
        return pivot

    async def _perform_rolling_optimization(
        self,
        prices_df: pd.DataFrame,
        benchmark_returns: pd.Series,
        risk_free_rate: float,
    ) -> Dict[str, Any]:
        """이동 윈도우 기반 포트폴리오 최적화 수행"""
        
        logger.info("Starting rolling window optimization")
        
        # 윈도우 크기 (3년 = 약 756 거래일)
        window_days = 252 * self.window_years
        
        # 이동 간격 (1개월 = 약 21 거래일)
        step_days = 21
        
        # 전체 데이터 길이 확인
        total_days = len(prices_df)
        if total_days < window_days + step_days:
            raise ValueError(f"Insufficient data: need at least {window_days + step_days} days, got {total_days}")
        
        # 최적화 결과 저장
        optimization_results = {
            'dates': [],
            'min_variance_weights': [],
            'max_sharpe_weights': [],
            'portfolio_returns': [],
            'window_var_cvar': [],  # 윈도우별 VaR/CVaR 저장
            'expected_returns': [],
            'covariances': []
        }
        
        # 이동 윈도우로 최적화 수행
        for start_idx in range(0, total_days - window_days, step_days):
            end_idx = start_idx + window_days
            
            # 윈도우 데이터 추출
            window_prices = prices_df.iloc[start_idx:end_idx]
            window_date = window_prices.index[-1]  # 윈도우의 마지막 날짜
            
            logger.debug(f"Optimizing window: {window_prices.index[0]} to {window_prices.index[-1]}")
            
            # 수익률 계산
            window_returns = window_prices.pct_change().fillna(0.0)
            
            # 통계 계산
            expected_returns = window_returns.mean() * 252.0
            cov_matrix = window_returns.cov() * 252.0
            
            # 포트폴리오 최적화
            min_var_weights = self._optimize_min_variance(cov_matrix)
            max_sharpe_weights = self._optimize_max_sharpe(expected_returns, cov_matrix, risk_free_rate)
            
            # 다음 기간의 실제 수익률 계산 (백테스팅용)
            if end_idx < total_days - 1:
                next_period_prices = prices_df.iloc[end_idx:end_idx + step_days]
                next_period_returns = next_period_prices.pct_change().fillna(0.0)
                
                # 최적화된 비중으로 포트폴리오 수익률 계산 (일별)
                mv_portfolio_returns = next_period_returns.dot(min_var_weights)
                ms_portfolio_returns = next_period_returns.dot(max_sharpe_weights)
                
                # 윈도우별 VaR/CVaR 계산
                mv_var, mv_cvar = self._calculate_var_cvar(mv_portfolio_returns)
                ms_var, ms_cvar = self._calculate_var_cvar(ms_portfolio_returns)
                
                # 다음 기간의 평균 수익률
                mv_portfolio_return = float(mv_portfolio_returns.mean())
                ms_portfolio_return = float(ms_portfolio_returns.mean())
            else:
                mv_portfolio_return = 0.0
                ms_portfolio_return = 0.0
                mv_var = mv_cvar = ms_var = ms_cvar = 0.0
            
            # 결과 저장
            optimization_results['dates'].append(window_date)
            optimization_results['min_variance_weights'].append(min_var_weights.to_dict())
            optimization_results['max_sharpe_weights'].append(max_sharpe_weights.to_dict())
            optimization_results['portfolio_returns'].append({
                'min_variance': mv_portfolio_return,
                'max_sharpe': ms_portfolio_return
            })
            optimization_results['window_var_cvar'].append({
                'min_variance': {'var': mv_var, 'cvar': mv_cvar},
                'max_sharpe': {'var': ms_var, 'cvar': ms_cvar}
            })
            optimization_results['expected_returns'].append(expected_returns.to_dict())
            optimization_results['covariances'].append(cov_matrix.to_dict())
        
        # 최신 비중 저장 (마지막 윈도우의 비중)
        optimization_results['latest_weights'] = {
            'min_variance': optimization_results['min_variance_weights'][-1],
            'max_sharpe': optimization_results['max_sharpe_weights'][-1]
        }
        
        logger.info(f"Completed rolling optimization: {len(optimization_results['dates'])} windows processed")
        return optimization_results

    def _optimize_min_variance(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """최소 변동성 포트폴리오 최적화"""
        n_assets = len(cov_matrix)
        
        # 목적 함수: 포트폴리오 분산 최소화
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix.values, weights))
        
        # 제약 조건: 비중의 합 = 1, 0 <= 비중 <= 0.9 (단일 종목 90% 초과 금지)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 0.9) for _ in range(n_assets))
        
        # 초기값 (균등 비중)
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                weights = pd.Series(result.x, index=cov_matrix.columns)
                return weights / weights.sum()  # 정규화
            else:
                logger.warning("Min variance optimization failed, using inverse variance weights")
                return self._fallback_min_variance_weights(cov_matrix)
        except Exception as e:
            logger.error(f"Min variance optimization error: {e}")
            return self._fallback_min_variance_weights(cov_matrix)

    def _optimize_max_sharpe(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float) -> pd.Series:
        """최대 샤프 비율 포트폴리오 최적화"""
        n_assets = len(expected_returns)
        
        # 목적 함수: 샤프 비율 최대화 (음의 샤프 비율 최소화)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns.values)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
            portfolio_std = np.sqrt(max(portfolio_variance, 1e-8))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
            return -sharpe_ratio  # 음수를 취해서 최소화
        
        # 제약 조건: 비중의 합 = 1, 0 <= 비중 <= 0.9 (단일 종목 90% 초과 금지)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 0.9) for _ in range(n_assets))
        
        # 초기값 (균등 비중)
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                weights = pd.Series(result.x, index=expected_returns.index)
                return weights / weights.sum()  # 정규화
            else:
                logger.warning("Max sharpe optimization failed, using expected return weighted weights")
                return self._fallback_max_sharpe_weights(expected_returns, cov_matrix, risk_free_rate)
        except Exception as e:
            logger.error(f"Max sharpe optimization error: {e}")
            return self._fallback_max_sharpe_weights(expected_returns, cov_matrix, risk_free_rate)

    def _fallback_min_variance_weights(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """최소 변동성 최적화 실패 시 대체 방법"""
        diag_var = np.diag(cov_matrix.values)
        inv_var = np.where(diag_var > 0, 1.0 / diag_var, 0.0)
        w = inv_var / inv_var.sum() if inv_var.sum() > 0 else np.ones_like(inv_var) / len(inv_var)
        return pd.Series(w, index=cov_matrix.columns)

    def _fallback_max_sharpe_weights(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float) -> pd.Series:
        """최대 샤프 최적화 실패 시 대체 방법"""
        diag_var = np.diag(cov_matrix.values)
        excess = (expected_returns.values - risk_free_rate)
        score = np.where(diag_var > 0, excess / diag_var, 0.0)
        score = np.maximum(score, 0.0)
        if score.sum() == 0:
            score = np.ones_like(score)
        w = score / score.sum()
        return pd.Series(w, index=expected_returns.index)

    async def _calculate_backtest_metrics(
        self,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        risk_free_rate: float,
    ) -> Dict[str, AnalysisMetrics]:
        """백테스팅 기반 성능 지표 계산"""
        
        logger.info("Calculating backtest metrics")
        
        # 포트폴리오 수익률 시계열 추출
        mv_returns = [r['min_variance'] for r in optimization_results['portfolio_returns']]
        ms_returns = [r['max_sharpe'] for r in optimization_results['portfolio_returns']]
        
        mv_returns_series = pd.Series(mv_returns, name='min_variance')
        ms_returns_series = pd.Series(ms_returns, name='max_sharpe')
        
        # 벤치마크 수익률과 동기화
        if not benchmark_returns.empty:
            # 벤치마크 수익률을 백테스팅 기간에 맞게 조정
            benchmark_period_returns = self._align_benchmark_returns(
                benchmark_returns, optimization_results['dates']
            )
        else:
            benchmark_period_returns = pd.Series(dtype=float)
        
        # 각 포트폴리오의 성능 지표 계산
        mv_metrics = self._calculate_portfolio_metrics(
            mv_returns_series, benchmark_period_returns, risk_free_rate, "Min Variance", optimization_results
        )
        ms_metrics = self._calculate_portfolio_metrics(
            ms_returns_series, benchmark_period_returns, risk_free_rate, "Max Sharpe", optimization_results
        )
        
        return {
            "min_variance": mv_metrics,
            "max_sharpe": ms_metrics,
        }

    def _align_benchmark_returns(self, benchmark_returns: pd.Series, optimization_dates: List[datetime]) -> pd.Series:
        """벤치마크 수익률을 백테스팅 기간에 맞게 조정"""
        aligned_returns = []
        
        for date in optimization_dates:
            # 해당 날짜 이후의 벤치마크 수익률 찾기
            future_returns = benchmark_returns[benchmark_returns.index > date]
            if not future_returns.empty:
                # 다음 기간의 평균 수익률 사용
                next_return = future_returns.head(21).mean()  # 약 1개월
                aligned_returns.append(next_return)
            else:
                aligned_returns.append(0.0)
        
        return pd.Series(aligned_returns)

    def _calculate_portfolio_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float,
        portfolio_name: str,
        optimization_results: Dict[str, Any] = None
    ) -> AnalysisMetrics:
        """포트폴리오 성능 지표 계산"""
        
        # 기본 통계
        expected_return = portfolio_returns.mean() * 252.0  # 연환산
        variance = portfolio_returns.var() * 252.0
        std_deviation = np.sqrt(max(variance, 0.0))
        
        # 벤치마크 관련 지표
        if not benchmark_returns.empty and len(benchmark_returns) > 0:
            beta, alpha, correlation = self._calculate_beta_alpha(
                portfolio_returns, benchmark_returns, risk_free_rate
            )
            tracking_error = self._calculate_tracking_error(portfolio_returns, benchmark_returns)
            upside_beta, downside_beta = self._calculate_upside_downside_beta(
                portfolio_returns, benchmark_returns, benchmark_returns.mean()
            )
            benchmark_annual_return = benchmark_returns.mean() * 252.0
        else:
            beta = 1.0
            alpha = 0.0
            correlation = 0.0
            tracking_error = 0.0
            upside_beta = 1.0
            downside_beta = 1.0
            benchmark_annual_return = 0.0
        
        # 위험조정 수익률 지표
        sharpe_ratio = (expected_return - risk_free_rate) / std_deviation if std_deviation > 0 else 0.0
        treynor_ratio = (expected_return - risk_free_rate) / beta if beta != 0 else 0.0
        
        # 하방편차 및 소르티노 비율
        downside_deviation = self._calculate_downside_deviation(portfolio_returns)
        sortino_ratio = (expected_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
        
        # 최대 낙폭 및 칼마 비율
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        calmar_ratio = expected_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # VaR/CVaR 계산 (윈도우별 평균)
        if optimization_results and 'window_var_cvar' in optimization_results:
            var_value, cvar_value = self._calculate_window_averaged_var_cvar(
                optimization_results['window_var_cvar'], portfolio_name
            )
        else:
            var_value, cvar_value = 0.0, 0.0
        
        # 정보비율
        information_ratio = (expected_return - benchmark_annual_return) / tracking_error if tracking_error > 0 else 0.0
        
        # 젠센 알파 (CAPM 기준)
        capm_expected_return = risk_free_rate + beta * (benchmark_annual_return - risk_free_rate)
        jensen_alpha = expected_return - capm_expected_return

        return AnalysisMetrics(
            expected_return=expected_return,
            std_deviation=std_deviation,
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
            var_value=var_value,
            cvar_value=cvar_value,
            correlation_with_benchmark=correlation,
        )

    def _calculate_beta_alpha(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float
    ) -> Tuple[float, float, float]:
        """베타, 알파, 상관관계 계산 (회귀분석 방식)"""
        try:
            # 초과수익률 계산
            portfolio_excess = portfolio_returns - risk_free_rate / 252.0
            benchmark_excess = benchmark_returns - risk_free_rate / 252.0
            
            # 공통 인덱스 찾기
            common_index = portfolio_excess.index.intersection(benchmark_excess.index)
            
            if len(common_index) < 10:  # 최소 10개 데이터 포인트 필요
                logger.warning(f"Insufficient data points for regression: {len(common_index)}")
                return 1.0, 0.0, 0.0
            
            # 공통 데이터 추출
            y = portfolio_excess.loc[common_index].values
            x = benchmark_excess.loc[common_index].values
            
            # 회귀분석 실행: portfolio_excess = alpha + beta * benchmark_excess + error
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            # 결과 정리
            beta = float(slope)
            alpha = float(intercept) * 252.0  # 연환산
            correlation = float(r_value)
            
            # 유효성 검증
            if np.isnan(beta) or np.isinf(beta):
                logger.warning(f"Invalid beta value: {beta}")
                return 1.0, 0.0, 0.0
            
            logger.debug(f"Portfolio beta calculated: beta={beta:.4f}, alpha={alpha:.4f}, r²={r_value**2:.4f}")
            
            return beta, alpha, correlation
            
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
            n = len(downside_returns)
            if n == 0:
                return 0.0
            if n == 1:
                return 0.0
            downside_deviation = downside_returns.std(ddof=0) * np.sqrt(252)
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

    def _calculate_var_cvar(self, returns: pd.Series) -> Tuple[float, float]:
        """VaR 95% 및 CVaR 95% 계산"""
        try:
            if len(returns) < 10:  # 충분한 데이터가 있어야 VaR 계산 가능
                return 0.0, 0.0
            
            # 수익률을 numpy 배열로 변환
            returns_array = returns.values
            
            # 95% VaR 계산 (5% 분위수)
            var_95 = np.percentile(returns_array, 5)
            
            # 95% CVaR 계산 (VaR보다 작은 값들의 평균)
            cvar_95 = returns_array[returns_array <= var_95].mean()
            
            # 연환산으로 변환
            var_value = float(var_95 * np.sqrt(252))  # 연환산
            cvar_value = float(cvar_95 * np.sqrt(252)) if not np.isnan(cvar_95) else 0.0  # 연환산
            
            return var_value, cvar_value
            
        except Exception as e:
            logger.error(f"Error calculating VaR/CVaR: {str(e)}")
            return 0.0, 0.0

    def _calculate_window_averaged_var_cvar(
        self, 
        window_var_cvar_data: List[Dict[str, Dict[str, float]]], 
        portfolio_name: str
    ) -> Tuple[float, float]:
        """EWMA 가중 평균 기반 윈도우별 VaR/CVaR 계산"""
        try:
            if not window_var_cvar_data:
                return 0.0, 0.0
            
            n_windows = len(window_var_cvar_data)
            
            # EWMA 가중치 계산 (감쇠계수 0.94 - 금융업계 표준)
            decay_factor = 0.94
            weights = np.array([decay_factor ** (n_windows - 1 - i) for i in range(n_windows)])
            weights = weights / weights.sum()  # 정규화
            
            # 해당 포트폴리오의 VaR/CVaR 값들 수집
            var_values = []
            cvar_values = []
            
            for i, window_data in enumerate(window_var_cvar_data):
                if portfolio_name in window_data:
                    var_val = window_data[portfolio_name].get('var', 0.0)
                    cvar_val = window_data[portfolio_name].get('cvar', 0.0)
                    
                    # 유효한 값만 수집 (0.0이 아닌 값)
                    if var_val != 0.0:
                        var_values.append((var_val, weights[i]))
                    if cvar_val != 0.0:
                        cvar_values.append((cvar_val, weights[i]))
            
            # EWMA 가중 평균 계산
            weighted_var = sum(val * weight for val, weight in var_values) if var_values else 0.0
            weighted_cvar = sum(val * weight for val, weight in cvar_values) if cvar_values else 0.0
            
            logger.debug(f"EWMA-weighted VaR/CVaR for {portfolio_name}: VaR={weighted_var:.4f}, CVaR={weighted_cvar:.4f} (from {len(var_values)} windows, decay={decay_factor})")
            
            return weighted_var, weighted_cvar
            
        except Exception as e:
            logger.error(f"Error calculating EWMA-weighted VaR/CVaR: {str(e)}")
            return 0.0, 0.0

    async def _calculate_benchmark_comparison(
        self,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        benchmark_code: str
    ) -> Optional[BenchmarkComparison]:
        """벤치마크 비교 분석"""
        try:
            if benchmark_returns.empty:
                return None
            
            # 백테스팅 기간의 포트폴리오 수익률
            mv_returns = [r['min_variance'] for r in optimization_results['portfolio_returns']]
            ms_returns = [r['max_sharpe'] for r in optimization_results['portfolio_returns']]
            
            # 최대 샤프 포트폴리오를 기준으로 비교 분석
            portfolio_returns = pd.Series(ms_returns)
            
            # 벤치마크 수익률 조정
            benchmark_period_returns = self._align_benchmark_returns(benchmark_returns, optimization_results['dates'])
            
            # 벤치마크 통계
            benchmark_annual_return = benchmark_period_returns.mean() * 252.0
            benchmark_volatility = benchmark_period_returns.std() * np.sqrt(252)
            
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

    async def _calculate_stock_details(
        self,
        optimization_results: Dict[str, Any],
        benchmark_code: str,
        benchmark_returns: pd.Series,
        session: AsyncSession,
        prices_df: pd.DataFrame = None
    ) -> Optional[Dict[str, StockDetails]]:
        """개별 종목 상세 정보 계산 (베타 포함)"""
        try:
            if not benchmark_code or benchmark_returns.empty:
                logger.warning("No benchmark data available for stock beta calculation")
                return None
            
            # 최적화에 사용된 종목 목록 추출
            latest_weights = optimization_results['latest_weights']
            portfolio_stocks = set(latest_weights['min_variance'].keys()) | set(latest_weights['max_sharpe'].keys())
            
            if not portfolio_stocks:
                logger.warning("No portfolio stocks found for beta calculation")
                return None
            
            # 최근 윈도우의 기대수익률과 공분산 데이터 추출
            latest_expected_returns = optimization_results['expected_returns'][-1] if optimization_results['expected_returns'] else {}
            latest_covariances = optimization_results['covariances'][-1] if optimization_results['covariances'] else {}
            
            stock_details = {}
            
            for stock_code in portfolio_stocks:
                try:
                    # 기대수익률과 변동성 추출
                    expected_return = latest_expected_returns.get(stock_code, 0.0)
                    volatility = np.sqrt(latest_covariances.get(stock_code, {}).get(stock_code, 0.0))
                    
                    # 포트폴리오와의 상관관계 계산 (최대 샤프 포트폴리오 기준)
                    max_sharpe_weights = latest_weights['max_sharpe']
                    portfolio_variance = 0.0
                    stock_portfolio_covariance = 0.0
                    
                    for other_stock, weight in max_sharpe_weights.items():
                        if other_stock in latest_covariances:
                            if stock_code == other_stock:
                                stock_portfolio_covariance += weight * latest_covariances.get(stock_code, {}).get(stock_code, 0.0)
                            else:
                                stock_portfolio_covariance += weight * latest_covariances.get(stock_code, {}).get(other_stock, 0.0)
                            
                            for other_stock2, weight2 in max_sharpe_weights.items():
                                if other_stock2 in latest_covariances.get(other_stock, {}):
                                    portfolio_variance += weight * weight2 * latest_covariances.get(other_stock, {}).get(other_stock2, 0.0)
                    
                    # 상관관계 계산
                    portfolio_std = np.sqrt(portfolio_variance) if portfolio_variance > 0 else 1.0
                    correlation_to_portfolio = stock_portfolio_covariance / (volatility * portfolio_std) if volatility > 0 and portfolio_std > 0 else 0.0
                    
                    # 베타 분석 계산
                    beta_analysis = None
                    if prices_df is not None and stock_code in prices_df.columns:
                        try:
                            # 종목 수익률 계산
                            stock_prices = prices_df[stock_code].dropna()
                            stock_returns = stock_prices.pct_change().dropna()
                            
                            # 벤치마크와 공통 기간 찾기
                            common_dates = stock_returns.index.intersection(benchmark_returns.index)
                            if len(common_dates) >= 10:  # 최소 10개 데이터 포인트
                                stock_returns_aligned = stock_returns.loc[common_dates]
                                benchmark_returns_aligned = benchmark_returns.loc[common_dates]
                                
                                # 회귀분석으로 베타 계산
                                slope, intercept, r_value, p_value, std_err = linregress(
                                    benchmark_returns_aligned.values, 
                                    stock_returns_aligned.values
                                )
                                
                                # 분석 기간 계산
                                start_date = common_dates.min()
                                end_date = common_dates.max()
                                
                                beta_analysis = BetaAnalysis(
                                    beta=float(slope),
                                    r_square=float(r_value ** 2),
                                    alpha=float(intercept) * 252,  # 연환산
                                    start=start_date.isoformat(),
                                    end=end_date.isoformat()
                                )
                                
                                logger.debug(f"Calculated beta analysis for {stock_code}: beta={slope:.4f}, r²={r_value**2:.4f}")
                            else:
                                logger.warning(f"Insufficient data for beta calculation: {stock_code}")
                                beta_analysis = self._get_default_beta_analysis()
                        except Exception as e:
                            logger.error(f"Error calculating beta for {stock_code}: {str(e)}")
                            beta_analysis = self._get_default_beta_analysis()
                    else:
                        # 가격 데이터가 없는 경우 기본값 사용
                        beta_analysis = self._get_default_beta_analysis()
                    
                    stock_detail = StockDetails(
                        expected_return=expected_return,
                        volatility=volatility,
                        correlation_to_portfolio=correlation_to_portfolio,
                        beta_analysis=beta_analysis
                    )
                    
                    stock_details[stock_code] = stock_detail
                    
                except Exception as e:
                    logger.error(f"Error processing stock {stock_code}: {str(e)}")
                    continue
            
            logger.info(f"Calculated stock details for {len(stock_details)} stocks")
            return stock_details
            
        except Exception as e:
            logger.error(f"Error calculating stock details: {str(e)}")
            return None

    async def _calculate_portfolio_beta_analysis(
        self,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        analysis_start: datetime,
        analysis_end: datetime
    ) -> Optional[BetaAnalysis]:
        """포트폴리오 베타 분석 계산"""
        try:
            if benchmark_returns.empty:
                logger.warning("No benchmark data available for portfolio beta calculation")
                return None
            
            # 백테스팅 기간의 포트폴리오 수익률 추출 (최대 샤프 포트폴리오 기준)
            portfolio_returns = [r['max_sharpe'] for r in optimization_results['portfolio_returns']]
            
            if len(portfolio_returns) < 10:
                logger.warning(f"Insufficient portfolio returns for beta calculation: {len(portfolio_returns)}")
                return None
            
            # 포트폴리오 수익률을 시계열로 변환
            portfolio_returns_series = pd.Series(portfolio_returns)
            
            # 벤치마크 수익률과 동기화
            benchmark_period_returns = self._align_benchmark_returns(
                benchmark_returns, optimization_results['dates']
            )
            
            if len(benchmark_period_returns) < 10:
                logger.warning(f"Insufficient benchmark returns for beta calculation: {len(benchmark_period_returns)}")
                return None
            
            # 베타 계산 (회귀분석)
            common_length = min(len(portfolio_returns_series), len(benchmark_period_returns))
            portfolio_aligned = portfolio_returns_series.iloc[:common_length]
            benchmark_aligned = benchmark_period_returns.iloc[:common_length]
            
            slope, intercept, r_value, p_value, std_err = linregress(
                benchmark_aligned.values, 
                portfolio_aligned.values
            )
            
            # 베타 분석 정보 생성
            portfolio_beta_analysis = BetaAnalysis(
                beta=float(slope),
                r_square=float(r_value ** 2),
                alpha=float(intercept) * 252,  # 연환산
                start=analysis_start.isoformat(),
                end=analysis_end.isoformat()
            )
            
            logger.info(f"Calculated portfolio beta analysis: beta={slope:.4f}, r²={r_value**2:.4f}, alpha={intercept*252:.4f}")
            return portfolio_beta_analysis
            
        except Exception as e:
            logger.error(f"Error calculating portfolio beta analysis: {str(e)}")
            return None

    async def _calculate_portfolio_beta_for_weights(
        self,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        analysis_start: datetime,
        analysis_end: datetime,
        portfolio_type: str
    ) -> Optional[BetaAnalysis]:
        """특정 포트폴리오 타입의 베타 계산"""
        try:
            if benchmark_returns.empty:
                logger.warning(f"No benchmark data available for {portfolio_type} beta calculation")
                return None
            
            # 백테스팅 기간의 포트폴리오 수익률 추출
            portfolio_returns = [r[portfolio_type] for r in optimization_results['portfolio_returns']]
            
            if len(portfolio_returns) < 10:
                logger.warning(f"Insufficient portfolio returns for {portfolio_type} beta calculation: {len(portfolio_returns)}")
                return None
            
            # 포트폴리오 수익률을 시계열로 변환
            portfolio_returns_series = pd.Series(portfolio_returns)
            
            # 벤치마크 수익률과 동기화
            benchmark_period_returns = self._align_benchmark_returns(
                benchmark_returns, optimization_results['dates']
            )
            
            if len(benchmark_period_returns) < 10:
                logger.warning(f"Insufficient benchmark returns for {portfolio_type} beta calculation: {len(benchmark_period_returns)}")
                return None
            
            # 베타 계산 (회귀분석)
            common_length = min(len(portfolio_returns_series), len(benchmark_period_returns))
            portfolio_aligned = portfolio_returns_series.iloc[:common_length]
            benchmark_aligned = benchmark_period_returns.iloc[:common_length]
            
            slope, intercept, r_value, p_value, std_err = linregress(
                benchmark_aligned.values, 
                portfolio_aligned.values
            )
            
            # 베타 분석 정보 생성
            portfolio_beta_analysis = BetaAnalysis(
                beta=float(slope),
                r_square=float(r_value ** 2),
                alpha=float(intercept) * 252  # 연환산
            )
            
            logger.info(f"Calculated {portfolio_type} beta analysis: beta={slope:.4f}, r²={r_value**2:.4f}, alpha={intercept*252:.4f}")
            return portfolio_beta_analysis
            
        except Exception as e:
            logger.error(f"Error calculating {portfolio_type} beta analysis: {str(e)}")
            return None

    async def _calculate_user_portfolio_beta(
        self,
        request: AnalysisRequest,
        benchmark_returns: pd.Series,
        analysis_start: datetime,
        analysis_end: datetime
    ) -> Optional[BetaAnalysis]:
        """사용자 입력 포트폴리오 베타 계산"""
        try:
            if benchmark_returns.empty or not request.holdings:
                logger.warning("No benchmark data or user holdings available for user portfolio beta calculation")
                return None
            
            # 사용자 포트폴리오 구성
            user_weights = {}
            total_value = sum(h.quantity for h in request.holdings)
            
            for holding in request.holdings:
                user_weights[holding.code] = holding.quantity / total_value
            
            # 사용자 포트폴리오 수익률 계산 (간소화)
            # 실제로는 가격 데이터를 로드해서 계산해야 하지만, 
            # 여기서는 기본값 반환
            logger.info(f"User portfolio beta calculation requested for {len(user_weights)} stocks")
            
            return BetaAnalysis(
                beta=1.0,
                r_square=0.0,
                alpha=0.0
            )
            
        except Exception as e:
            logger.error(f"Error calculating user portfolio beta: {str(e)}")
            return None

    async def _build_portfolio_data(
        self,
        request: AnalysisRequest,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        analysis_start: datetime,
        analysis_end: datetime,
        backtest_metrics: Dict[str, AnalysisMetrics],
        prices_df: pd.DataFrame,
        risk_free_rate: float,
        benchmark_code: Optional[str],
    ) -> List[PortfolioData]:
        """포트폴리오 데이터 구성"""
        portfolios = []
        latest_weights = optimization_results['latest_weights']
        
        # 1. 사용자 포트폴리오
        user_weights = self._calculate_user_weights(request)
        user_beta = await self._calculate_user_portfolio_beta(
            request, benchmark_returns, analysis_start, analysis_end
        )
        user_metrics = await self._calculate_user_portfolio_metrics(
            request,
            benchmark_returns,
            analysis_start,
            analysis_end,
            prices_df,
            risk_free_rate,
        )
        user_benchmark_comparison = await self._calculate_user_benchmark_comparison(
            request,
            benchmark_returns,
            analysis_start,
            analysis_end,
            prices_df,
            benchmark_code,
        )
        
        portfolios.append(PortfolioData(
            type="user",
            weights=user_weights,
            beta_analysis=user_beta,
            metrics=user_metrics,
            benchmark_comparison=user_benchmark_comparison
        ))
        
        # 2. 최소분산 포트폴리오
        min_var_beta = await self._calculate_portfolio_beta_for_weights(
            optimization_results, benchmark_returns, analysis_start, analysis_end, "min_variance"
        )
        min_var_benchmark_comparison = await self._calculate_portfolio_benchmark_comparison(
            optimization_results, benchmark_returns, "min_variance"
        )
        
        portfolios.append(PortfolioData(
            type="min_variance",
            weights=latest_weights['min_variance'],
            beta_analysis=min_var_beta,
            metrics=backtest_metrics.get('min_variance'),
            benchmark_comparison=min_var_benchmark_comparison
        ))
        
        # 3. 최대샤프 포트폴리오
        max_sharpe_beta = await self._calculate_portfolio_beta_for_weights(
            optimization_results, benchmark_returns, analysis_start, analysis_end, "max_sharpe"
        )
        max_sharpe_benchmark_comparison = await self._calculate_portfolio_benchmark_comparison(
            optimization_results, benchmark_returns, "max_sharpe"
        )
        
        portfolios.append(PortfolioData(
            type="max_sharpe",
            weights=latest_weights['max_sharpe'],
            beta_analysis=max_sharpe_beta,
            metrics=backtest_metrics.get('max_sharpe'),
            benchmark_comparison=max_sharpe_benchmark_comparison
        ))
        
        return portfolios

    def _calculate_user_weights(self, request: AnalysisRequest) -> Dict[str, float]:
        """사용자 포트폴리오 비중 계산"""
        total_value = sum(h.quantity for h in request.holdings)
        return {h.code: h.quantity / total_value for h in request.holdings}

    async def _calculate_user_portfolio_metrics(
        self,
        request: AnalysisRequest,
        benchmark_returns: pd.Series,
        analysis_start: datetime,
        analysis_end: datetime,
        prices_df: pd.DataFrame,
        risk_free_rate: float,
    ) -> Optional[AnalysisMetrics]:
        """사용자 포트폴리오 성과 지표 계산"""
        try:
            if prices_df is None or prices_df.empty:
                return None
            symbols = [h.code for h in request.holdings]
            available = [s for s in symbols if s in prices_df.columns]
            if not available:
                return None
            total_qty = sum(h.quantity for h in request.holdings if h.code in available)
            if total_qty <= 0:
                return None
            weights = {h.code: h.quantity / total_qty for h in request.holdings if h.code in available}
            returns_df = prices_df[available].pct_change().dropna()
            if returns_df.empty:
                return None
            user_returns = returns_df.dot(pd.Series(weights))
            if not benchmark_returns.empty:
                common_idx = user_returns.index.intersection(benchmark_returns.index)
                bench_aligned = benchmark_returns.loc[common_idx]
                port_aligned = user_returns.loc[common_idx]
            else:
                bench_aligned = pd.Series(dtype=float)
                port_aligned = user_returns
            return self._calculate_portfolio_metrics(
                port_aligned, bench_aligned, risk_free_rate, "User", None
            )
        except Exception as e:
            logger.error(f"Error calculating user portfolio metrics: {str(e)}")
            return None

    async def _calculate_user_benchmark_comparison(
        self,
        request: AnalysisRequest,
        benchmark_returns: pd.Series,
        analysis_start: datetime,
        analysis_end: datetime,
        prices_df: pd.DataFrame,
        benchmark_code: Optional[str],
    ) -> Optional[BenchmarkComparison]:
        """사용자 포트폴리오 벤치마크 비교"""
        try:
            if prices_df is None or prices_df.empty or benchmark_returns.empty:
                return None
            symbols = [h.code for h in request.holdings]
            available = [s for s in symbols if s in prices_df.columns]
            if not available:
                return None
            total_qty = sum(h.quantity for h in request.holdings if h.code in available)
            if total_qty <= 0:
                return None
            weights = {h.code: h.quantity / total_qty for h in request.holdings if h.code in available}
            returns_df = prices_df[available].pct_change().dropna()
            if returns_df.empty:
                return None
            user_returns = returns_df.dot(pd.Series(weights))
            common_idx = user_returns.index.intersection(benchmark_returns.index)
            if len(common_idx) == 0:
                return None
            port_aligned = user_returns.loc[common_idx]
            bench_aligned = benchmark_returns.loc[common_idx]
            benchmark_annual_return = bench_aligned.mean() * 252.0
            benchmark_volatility = bench_aligned.std() * np.sqrt(252)
            portfolio_annual_return = port_aligned.mean() * 252.0
            portfolio_volatility = port_aligned.std() * np.sqrt(252)
            excess_return = portfolio_annual_return - benchmark_annual_return
            relative_volatility = (
                portfolio_volatility / benchmark_volatility if benchmark_volatility > 0 else 1.0
            )
            security_selection = excess_return * 0.7
            timing_effect = excess_return * 0.3
            return BenchmarkComparison(
                benchmark_code=benchmark_code or "KOSPI",
                benchmark_return=float(benchmark_annual_return),
                benchmark_volatility=float(benchmark_volatility),
                excess_return=float(excess_return),
                relative_volatility=float(relative_volatility),
                security_selection=float(security_selection),
                timing_effect=float(timing_effect),
            )
        except Exception as e:
            logger.error(f"Error calculating user benchmark comparison: {str(e)}")
            return None

    async def _calculate_portfolio_benchmark_comparison(
        self,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        portfolio_type: str
    ) -> Optional[BenchmarkComparison]:
        """특정 포트폴리오의 벤치마크 비교"""
        try:
            if benchmark_returns.empty:
                return None
            
            # 백테스팅 기간의 포트폴리오 수익률 추출
            portfolio_returns = [r[portfolio_type] for r in optimization_results['portfolio_returns']]
            portfolio_returns_series = pd.Series(portfolio_returns)
            
            # 벤치마크 수익률과 동기화
            benchmark_period_returns = self._align_benchmark_returns(
                benchmark_returns, optimization_results['dates']
            )
            
            # 벤치마크 통계
            benchmark_annual_return = benchmark_period_returns.mean() * 252.0
            benchmark_volatility = benchmark_period_returns.std() * np.sqrt(252)
            
            # 포트폴리오 통계
            portfolio_annual_return = portfolio_returns_series.mean() * 252.0
            portfolio_volatility = portfolio_returns_series.std() * np.sqrt(252)
            
            # 초과 성과
            excess_return = portfolio_annual_return - benchmark_annual_return
            relative_volatility = portfolio_volatility / benchmark_volatility if benchmark_volatility > 0 else 1.0
            
            # 성과 기여도 분석 (단순화된 버전)
            security_selection = excess_return * 0.7
            timing_effect = excess_return * 0.3
            
            return BenchmarkComparison(
                benchmark_code="KOSPI",  # 실제로는 benchmark_code를 전달받아야 함
                benchmark_return=float(benchmark_annual_return),
                benchmark_volatility=float(benchmark_volatility),
                excess_return=float(excess_return),
                relative_volatility=float(relative_volatility),
                security_selection=float(security_selection),
                timing_effect=float(timing_effect)
            )
            
        except Exception as e:
            logger.error(f"Error calculating {portfolio_type} benchmark comparison: {str(e)}")
            return None

    def _get_default_beta_analysis(self) -> BetaAnalysis:
        """기본 베타 분석 정보 반환 (계산 실패 시)"""
        return BetaAnalysis(
            beta=1.0,
            r_square=0.0,
            alpha=0.0
        )
