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

            # 해당 날짜 또는 그 이전 가장 최근 데이터 조회
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
                # 연율 기준 금리를 소수로 변환 (예: 3.25% -> 0.0325)
                return float(rate_data.rate) / 100.0
            
            logger.warning(f"No risk-free rate found for {rate_type} on or before {target_date}")
            return None

        except Exception as e:
            logger.error(f"Error retrieving risk-free rate: {str(e)}")
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
                logger.warning(f"No risk-free rate series found for {rate_type} between {start_date} and {end_date}")
                return pd.Series(dtype=float)

            # DataFrame으로 변환 후 시계열 생성
            data = []
            for rate in rate_data:
                data.append({
                    'datetime': rate.datetime,
                    'rate': float(rate.rate) / 100.0,  # 연율 기준을 소수로 변환
                    'daily_rate': float(rate.daily_rate) if rate.daily_rate else 0.0
                })

            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
            
            # 날짜를 인덱스로 하는 시리즈 반환
            rate_series = pd.Series(
                data=df['rate'].values,
                index=df['datetime'],
                name=f'{rate_type}_rate'
            )
            
            logger.info(f"Retrieved {len(rate_series)} risk-free rate records for {rate_type}")
            return rate_series

        except Exception as e:
            logger.error(f"Error retrieving risk-free rate series: {str(e)}")
            raise

    async def get_available_rate_types(self) -> List[str]:
        """사용 가능한 금리 유형 목록 조회"""
        try:
            stmt = select(RiskFreeRate.rate_type).distinct()
            result = await self.session.execute(stmt)
            return [rate_type for (rate_type,) in result.fetchall()]

        except Exception as e:
            logger.error(f"Error retrieving available rate types: {str(e)}")
            raise

    async def get_latest_risk_free_rate(self, rate_type: str = "CD91") -> Optional[RiskFreeRate]:
        """특정 금리 유형의 최신 무위험수익률 조회"""
        try:
            stmt = (
                select(RiskFreeRate)
                .where(RiskFreeRate.rate_type == rate_type)
                .order_by(desc(RiskFreeRate.datetime))
                .limit(1)
            )
            
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Error retrieving latest risk-free rate for {rate_type}: {str(e)}")
            raise

    async def interpolate_missing_rates(
        self,
        rate_series: pd.Series,
        target_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """누락된 날짜의 금리를 보간하여 시계열 생성"""
        try:
            # 기존 시리즈를 날짜 인덱스로 리인덱싱
            complete_series = rate_series.reindex(target_dates)
            
            # 전진 채우기로 누락 데이터 보간 (금리는 보통 변화가 느림)
            complete_series = complete_series.fillna(method='ffill')
            
            # 시작 부분에 여전히 NaN이 있으면 후진 채우기
            complete_series = complete_series.fillna(method='bfill')
            
            # 여전히 NaN이 있으면 0으로 채우기
            complete_series = complete_series.fillna(0.0)
            
            logger.info(f"Interpolated risk-free rate series to {len(complete_series)} observations")
            return complete_series

        except Exception as e:
            logger.error(f"Error interpolating risk-free rates: {str(e)}")
            raise

    async def get_risk_free_rate_stats(
        self,
        start_date: datetime,
        end_date: datetime,
        rate_type: str = "CD91"
    ) -> Dict[str, float]:
        """무위험수익률 통계 정보 조회"""
        try:
            rate_series = await self.get_risk_free_rate_series(start_date, end_date, rate_type)
            
            if rate_series.empty:
                return {}
            
            return {
                'mean_rate': float(rate_series.mean()),
                'median_rate': float(rate_series.median()),
                'std_rate': float(rate_series.std()),
                'min_rate': float(rate_series.min()),
                'max_rate': float(rate_series.max()),
                'current_rate': float(rate_series.iloc[-1]) if len(rate_series) > 0 else 0.0,
                'observation_days': len(rate_series)
            }

        except Exception as e:
            logger.error(f"Error calculating risk-free rate statistics: {str(e)}")
            raise


