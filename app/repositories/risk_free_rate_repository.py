"""무위험수익률 데이터 리포지토리"""

from typing import List, Optional, Dict
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, asc

from app.models.stock import RiskFreeRate
from app.repositories.base import BaseRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RiskFreeRateRepository(BaseRepository[RiskFreeRate]):
    """무위험수익률 데이터 리포지토리"""

    def __init__(self, session: AsyncSession):
        super().__init__(RiskFreeRate, session)

    async def get_risk_free_rate(
        self,
        rate_type: str = "CD91",
        target_date: Optional[datetime] = None
    ) -> Optional[float]:
        """특정 날짜의 무위험수익률 조회 (연율 기준)"""
        try:
            if target_date is None:
                target_date = datetime.utcnow()

            stmt = (
                select(RiskFreeRate)
                .where(
                    and_(
                        RiskFreeRate.rate_type == rate_type,
                        RiskFreeRate.datetime <= target_date
                    )
                )
                .order_by(desc(RiskFreeRate.datetime))
                .limit(1)
            )
            
            result = await self.session.execute(stmt)
            rate_data = result.scalar_one_or_none()
            
            if rate_data:
                return float(rate_data.rate) / 100.0
            
            logger.warning(f"{target_date} 또는 그 이전의 {rate_type} 무위험수익률을 찾을 수 없음")
            return None

        except Exception as e:
            logger.error(f"무위험수익률 조회 오류: {str(e)}")
            raise

    async def get_risk_free_rate_series(
        self,
        start_date: datetime,
        end_date: datetime,
        rate_type: str = "CD91"
    ) -> pd.Series:
        """지정된 기간의 무위험수익률 시계열 조회"""
        try:
            stmt = (
                select(RiskFreeRate)
                .where(
                    and_(
                        RiskFreeRate.rate_type == rate_type,
                        RiskFreeRate.datetime >= start_date,
                        RiskFreeRate.datetime <= end_date
                    )
                )
                .order_by(asc(RiskFreeRate.datetime))
            )
            
            result = await self.session.execute(stmt)
            rate_data = result.scalars().all()
            
            if not rate_data:
                logger.warning(f"{start_date}와 {end_date} 사이의 {rate_type} 무위험수익률 시리즈를 찾을 수 없음")
                return pd.Series(dtype=float)

            data = []
            for rate in rate_data:
                data.append({
                    'datetime': rate.datetime,
                    'rate': float(rate.rate) / 100.0,
                    'daily_rate': float(rate.daily_rate) if rate.daily_rate else 0.0
                })

            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
            
            rate_series = pd.Series(
                data=df['rate'].values,
                index=df['datetime'],
                name=f'{rate_type}_rate'
            )
            
            logger.info(f"{rate_type}에 대해 {len(rate_series)}개 무위험수익률 레코드 조회 완료")
            return rate_series

        except Exception as e:
            logger.error(f"무위험수익률 시리즈 조회 오류: {str(e)}")
            raise

    async def get_available_rate_types(self) -> List[str]:
        """사용 가능한 금리 유형 목록 조회"""
        try:
            stmt = select(RiskFreeRate.rate_type).distinct()
            result = await self.session.execute(stmt)
            return [rate_type for (rate_type,) in result.fetchall()]

        except Exception as e:
            logger.error(f"사용 가능한 금리 유형 조회 오류: {str(e)}")
            raise

    async def interpolate_missing_rates(
        self,
        rate_series: pd.Series,
        target_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """누락된 날짜의 금리를 보간하여 시계열 생성"""
        try:
            complete_series = rate_series.reindex(target_dates)
            
            complete_series = complete_series.fillna(method='ffill')
            
            complete_series = complete_series.fillna(method='bfill')
            
            complete_series = complete_series.fillna(0.0)
            
            logger.info(f"무위험수익률 시리즈를 {len(complete_series)}개 관측치로 보간 완료")
            return complete_series

        except Exception as e:
            logger.error(f"무위험수익률 보간 오류: {str(e)}")
            raise
