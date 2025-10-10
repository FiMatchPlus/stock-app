from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List
import pandas as pd


class BetaService:
    async def _calculate_stock_details(
        self,
        optimization_results: Dict[str, Any],
        benchmark_code: str,
        benchmark_returns: pd.Series,
        session,
        prices_df: pd.DataFrame = None,
    ) -> Optional[Dict[str, Any]]:
        return await super()._calculate_stock_details(
            optimization_results, benchmark_code, benchmark_returns, session, prices_df
        )

    async def _calculate_portfolio_beta_analysis(
        self,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        analysis_start,
        analysis_end,
    ) -> Optional[object]:
        return await super()._calculate_portfolio_beta_analysis(
            optimization_results, benchmark_returns, analysis_start, analysis_end
        )

    async def _calculate_portfolio_beta_for_weights(
        self,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        analysis_start,
        analysis_end,
        portfolio_type: str,
    ) -> Optional[object]:
        return await super()._calculate_portfolio_beta_for_weights(
            optimization_results, benchmark_returns, analysis_start, analysis_end, portfolio_type
        )

    async def _calculate_user_portfolio_beta(
        self,
        request,
        benchmark_returns: pd.Series,
        analysis_start,
        analysis_end,
    ) -> Optional[object]:
        return await super()._calculate_user_portfolio_beta(
            request, benchmark_returns, analysis_start, analysis_end
        )


