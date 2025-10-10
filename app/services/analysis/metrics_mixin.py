from __future__ import annotations

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd

from app.utils.logger import get_logger


logger = get_logger(__name__)


class MetricsMixin:
    async def _calculate_backtest_metrics(
        self,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        risk_free_rate: float,
    ) -> Dict[str, Any]:
        return await super()._calculate_backtest_metrics(optimization_results, benchmark_returns, risk_free_rate)

    def _align_benchmark_returns(self, benchmark_returns: pd.Series, optimization_dates: List) -> pd.Series:
        return super()._align_benchmark_returns(benchmark_returns, optimization_dates)

    def _calculate_portfolio_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float,
        portfolio_name: str,
        optimization_results: Dict[str, Any] = None
    ) -> Any:
        return super()._calculate_portfolio_metrics(
            portfolio_returns, benchmark_returns, risk_free_rate, portfolio_name, optimization_results
        )

    def _calculate_beta_alpha(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float
    ) -> Tuple[float, float, float]:
        return super()._calculate_beta_alpha(portfolio_returns, benchmark_returns, risk_free_rate)

    def _calculate_tracking_error(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        return super()._calculate_tracking_error(portfolio_returns, benchmark_returns)

    def _calculate_upside_downside_beta(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, benchmark_mean: float
    ) -> Tuple[float, float]:
        return super()._calculate_upside_downside_beta(portfolio_returns, benchmark_returns, benchmark_mean)

    def _calculate_downside_deviation(self, returns: pd.Series, target_return: float = 0.0) -> float:
        return super()._calculate_downside_deviation(returns, target_return)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        return super()._calculate_max_drawdown(returns)

    def _calculate_var_cvar(self, returns: pd.Series) -> Tuple[float, float]:
        return super()._calculate_var_cvar(returns)

    def _calculate_window_averaged_var_cvar(
        self, window_var_cvar_data: List[Dict[str, Dict[str, float]]], portfolio_name: str
    ) -> Tuple[float, float]:
        return super()._calculate_window_averaged_var_cvar(window_var_cvar_data, portfolio_name)


