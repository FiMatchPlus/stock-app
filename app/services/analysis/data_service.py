from __future__ import annotations

from datetime import datetime
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession


class DataService:
    async def _load_daily_prices(
        self,
        session: AsyncSession,
        request,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        return await super()._load_daily_prices(session, request, start_dt, end_dt)

    def _synchronize_time_series(self, prices_df: pd.DataFrame, benchmark_returns: pd.Series):
        return super()._synchronize_time_series(prices_df, benchmark_returns)


