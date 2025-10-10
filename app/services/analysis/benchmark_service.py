from __future__ import annotations

from typing import Optional
import pandas as pd


class BenchmarkService:
    async def _determine_benchmark(self, request, benchmark_repo) -> str:
        return await super()._determine_benchmark(request, benchmark_repo)

    async def _calculate_benchmark_comparison(
        self,
        optimization_results,
        benchmark_returns: pd.Series,
        benchmark_code: str,
    ) -> Optional[object]:
        return await super()._calculate_benchmark_comparison(
            optimization_results, benchmark_returns, benchmark_code
        )

    async def _calculate_user_benchmark_comparison(
        self,
        request,
        benchmark_returns: pd.Series,
        analysis_start,
        analysis_end,
        prices_df: pd.DataFrame,
        benchmark_code: Optional[str],
    ) -> Optional[object]:
        return await super()._calculate_user_benchmark_comparison(
            request, benchmark_returns, analysis_start, analysis_end, prices_df, benchmark_code
        )


