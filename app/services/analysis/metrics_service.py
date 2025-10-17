from __future__ import annotations

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from scipy.stats import linregress

from app.models.schemas import AnalysisMetrics
from app.utils.logger import get_logger


logger = get_logger(__name__)


class MetricsService:
    async def _calculate_backtest_metrics(
        self,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        risk_free_rate: float,
    ) -> Dict[str, Any]:
        """백테스팅 기반 성능 지표 계산"""
        
        logger.info("백테스트 메트릭 계산 중")
        
        mv_returns = [r['min_downside_risk'] for r in optimization_results['portfolio_returns']]
        ms_returns = [r['max_sortino'] for r in optimization_results['portfolio_returns']]
        dates = optimization_results['dates']
        
        mv_returns_series = pd.Series(mv_returns, index=dates, name='min_downside_risk')
        ms_returns_series = pd.Series(ms_returns, index=dates, name='max_sortino')
        
        if not benchmark_returns.empty:
            benchmark_period_returns = self._align_benchmark_returns(
                benchmark_returns, optimization_results['dates']
            )
            benchmark_period_returns.index = dates
        else:
            benchmark_period_returns = pd.Series(dtype=float)
        
        mv_metrics = self._calculate_portfolio_metrics(
            mv_returns_series, benchmark_period_returns, risk_free_rate, "Min Variance", optimization_results
        )
        ms_metrics = self._calculate_portfolio_metrics(
            ms_returns_series, benchmark_period_returns, risk_free_rate, "Max Sortino", optimization_results
        )
        
        return {
            "min_downside_risk": mv_metrics,
            "max_sortino": ms_metrics,
        }

    def _align_benchmark_returns(self, benchmark_returns: pd.Series, optimization_dates: List) -> pd.Series:
        """벤치마크 수익률을 백테스팅 기간에 맞게 조정"""
        aligned_returns = []
        
        for date in optimization_dates:
            future_returns = benchmark_returns[benchmark_returns.index > date]
            if not future_returns.empty:
                next_return = future_returns.head(21).mean()
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
    ) -> Any:
        """포트폴리오 성능 지표 계산"""
        
        expected_return = portfolio_returns.mean() * 252.0
        variance = portfolio_returns.var() * 252.0
        std_deviation = np.sqrt(max(variance, 0.0))
        
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
        
        sharpe_ratio = (expected_return - risk_free_rate) / std_deviation if std_deviation > 0 else 0.0
        treynor_ratio = (expected_return - risk_free_rate) / beta if beta != 0 else 0.0
        
        downside_deviation = self._calculate_downside_deviation(portfolio_returns)
        sortino_ratio = (expected_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
        
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        calmar_ratio = expected_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        if optimization_results and 'window_var_cvar' in optimization_results:
            var_value, cvar_value = self._calculate_window_averaged_var_cvar(
                optimization_results['window_var_cvar'], portfolio_name
            )
        else:
            var_value, cvar_value = 0.0, 0.0
        
        information_ratio = (expected_return - benchmark_annual_return) / tracking_error if tracking_error > 0 else 0.0

        return AnalysisMetrics(
            expected_return=expected_return,
            std_deviation=std_deviation,
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
            portfolio_excess = portfolio_returns - risk_free_rate / 252.0
            benchmark_excess = benchmark_returns - risk_free_rate / 252.0
            
            common_index = portfolio_excess.index.intersection(benchmark_excess.index)
            
            if len(common_index) < 10:
                logger.warning(f"회귀분석을 위한 데이터 포인트 부족: {len(common_index)}")
                return 1.0, 0.0, 0.0
            
            y = portfolio_excess.loc[common_index].values
            x = benchmark_excess.loc[common_index].values
            
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            beta = float(slope)
            alpha = float(intercept) * 252.0
            correlation = float(r_value)
            
            if np.isnan(beta) or np.isinf(beta):
                logger.warning(f"유효하지 않은 베타 값: {beta}")
                return 1.0, 0.0, 0.0
            
            logger.debug(f"포트폴리오 베타 계산 완료: beta={beta:.4f}, alpha={alpha:.4f}, r²={r_value**2:.4f}")
            
            return beta, alpha, correlation
            
        except Exception as e:
            logger.error(f"베타/알파 계산 오류: {str(e)}")
            return 1.0, 0.0, 0.0

    def _calculate_tracking_error(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """트래킹 에러 계산"""
        try:
            excess_returns = portfolio_returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            return float(tracking_error)
        except Exception as e:
            logger.error(f"트래킹 에러 계산 오류: {str(e)}")
            return 0.0

    def _calculate_upside_downside_beta(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, benchmark_mean: float
    ) -> Tuple[float, float]:
        """상승/하락 베타 계산"""
        try:
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
            logger.error(f"상승/하락 베타 계산 오류: {str(e)}")
            return 1.0, 1.0

    def _calculate_downside_deviation(self, returns: pd.Series, target_return: float = 0.0) -> float:
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
            logger.error(f"하방편차 계산 오류: {str(e)}")
            return 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """최대 낙폭 계산"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return float(drawdown.min())
        except Exception as e:
            logger.error(f"최대 낙폭 계산 오류: {str(e)}")
            return 0.0

    def _calculate_var_cvar(self, returns: pd.Series) -> Tuple[float, float]:
        """VaR 95% 및 CVaR 95% 계산
        
        Args:
            returns: 일별 포트폴리오 수익률 시계열
            
        Returns:
            (var_95, cvar_95): 일별 VaR/CVaR 값 (음수)
        """
        try:
            n = len(returns)
            
            if n < 5:
                logger.warning(f"VaR/CVaR 계산을 위한 데이터 부족: {n}개 (최소 5개 권장)")
                return 0.0, 0.0
            
            returns_array = returns.values
            
            var_95 = np.percentile(returns_array, 5)
            
            tail_losses = returns_array[returns_array < var_95]
            
            if len(tail_losses) > 0:
                cvar_95 = tail_losses.mean()
            else:
                cvar_95 = var_95
            
            var_value = float(var_95)
            cvar_value = float(cvar_95) if not np.isnan(cvar_95) else var_95
            
            logger.debug(f"VaR/CVaR 계산 완료: 데이터 {n}개, VaR={var_value:.4f}, CVaR={cvar_value:.4f}")
            
            return var_value, cvar_value
            
        except Exception as e:
            logger.error(f"VaR/CVaR 계산 오류: {str(e)}")
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
            
            decay_factor = 0.94
            weights = np.array([decay_factor ** (n_windows - 1 - i) for i in range(n_windows)])
            weights = weights / weights.sum()
            
            var_values = []
            cvar_values = []
            
            for i, window_data in enumerate(window_var_cvar_data):
                if portfolio_name in window_data:
                    var_val = window_data[portfolio_name].get('var', 0.0)
                    cvar_val = window_data[portfolio_name].get('cvar', 0.0)
                    
                    var_values.append((var_val, weights[i]))
                    cvar_values.append((cvar_val, weights[i]))
            
            weighted_var = sum(val * weight for val, weight in var_values) if var_values else 0.0
            weighted_cvar = sum(val * weight for val, weight in cvar_values) if cvar_values else 0.0
            
            logger.debug(f"EWMA-weighted VaR/CVaR for {portfolio_name}: VaR={weighted_var:.4f}, CVaR={weighted_cvar:.4f} (from {len(var_values)} windows, decay={decay_factor})")
            
            return weighted_var, weighted_cvar
            
        except Exception as e:
            logger.error(f"EWMA 가중 VaR/CVaR 계산 오류: {str(e)}")
            return 0.0, 0.0

