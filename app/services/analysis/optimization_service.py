from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app.utils.logger import get_logger


logger = get_logger(__name__)


class OptimizationService:
    async def _perform_rolling_optimization(
        self,
        prices_df: pd.DataFrame,
        benchmark_returns: pd.Series,
        risk_free_rate: float,
    ) -> Dict[str, Any]:
        return await super()._perform_rolling_optimization(prices_df, benchmark_returns, risk_free_rate)

    def _optimize_min_variance(self, cov_matrix: pd.DataFrame) -> pd.Series:
        return super()._optimize_min_variance(cov_matrix)

    def _optimize_max_sharpe(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float) -> pd.Series:
        return super()._optimize_max_sharpe(expected_returns, cov_matrix, risk_free_rate)

    def _fallback_min_variance_weights(self, cov_matrix: pd.DataFrame) -> pd.Series:
        return super()._fallback_min_variance_weights(cov_matrix)

    def _fallback_max_sharpe_weights(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float) -> pd.Series:
        return super()._fallback_max_sharpe_weights(expected_returns, cov_matrix, risk_free_rate)


