from __future__ import annotations

from typing import List, Optional, Dict, Any
import pandas as pd


class ComposeService:
    async def _build_portfolio_data(
        self,
        request,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        analysis_start,
        analysis_end,
        backtest_metrics,
        prices_df: pd.DataFrame,
        risk_free_rate: float,
        benchmark_code: Optional[str],
    ) -> List[object]:
        return await super()._build_portfolio_data(
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


