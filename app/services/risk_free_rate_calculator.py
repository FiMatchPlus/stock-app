"""무위험수익률 계산기"""

from typing import Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.stock import RiskFreeRate
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RiskFreeRateCalculator:
    """무위험수익률 계산기"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def calculate_risk_free_rate(
        self,
        analysis_start: datetime,
        analysis_end: datetime,
        user_risk_free_rate: Optional[float] = None
    ) -> float:
        """분석 기간에 따른 무위험수익률 계산
        
        Args:
            analysis_start: 분석 시작일
            analysis_end: 분석 종료일
            user_risk_free_rate: 사용자가 직접 지정한 무위험수익률
            
        Returns:
            float: 무위험수익률 (연환산)
        """
        try:
            if user_risk_free_rate is not None:
                logger.info(f"Using user-specified risk-free rate: {user_risk_free_rate}")
                return user_risk_free_rate
            
            rate_type = self._select_treasury_bond_type(analysis_start, analysis_end)
            logger.info(f"Selected treasury bond type: {rate_type} for analysis period: {(analysis_end - analysis_start).days} days")
            
            annual_rates = await self._get_treasury_daily_returns(rate_type, analysis_start, analysis_end)
            
            if not annual_rates:
                logger.warning(f"No treasury data available for {rate_type}, using default 0.0")
                return 0.0
            
            average_rate_pct = sum(annual_rates) / len(annual_rates)
            annualized_rate = average_rate_pct / 100
            
            logger.info(f"Calculated risk-free rate: {annualized_rate:.4f} ({average_rate_pct:.2f}% from {len(annual_rates)} data points)")
            return annualized_rate
            
        except Exception as e:
            logger.error(f"Error calculating risk-free rate: {str(e)}")
            return 0.0
    
    def _select_treasury_bond_type(self, start: datetime, end: datetime) -> str:
        """분석 기간 길이에 따른 국고채 타입 선택
        
        Args:
            start: 분석 시작일
            end: 분석 종료일
            
        Returns:
            str: 국고채 타입 (TB1Y, TB3Y, TB5Y)
        """
        analysis_days = (end - start).days
        
        if analysis_days < 365:
            return "TB1Y"
        elif analysis_days < 1095:
            return "TB3Y"
        else:
            return "TB5Y"
    
    async def _get_treasury_daily_returns(
        self,
        rate_type: str,
        start: datetime,
        end: datetime
    ) -> list[float]:
        """특정 기간의 국고채 연율 데이터 조회
        
        Args:
            rate_type: 국고채 타입 (TB1Y, TB3Y, TB5Y)
            start: 시작일
            end: 종료일
            
        Returns:
            list[float]: 연율(%) 리스트 (예: [3.26, 3.28, 3.25, ...])
        """
        try:
            stmt = (
                select(RiskFreeRate.rate)
                .where(
                    and_(
                        RiskFreeRate.rate_type == rate_type,
                        RiskFreeRate.datetime >= start,
                        RiskFreeRate.datetime <= end
                    )
                )
                .order_by(RiskFreeRate.datetime)
            )
            
            result = await self.session.execute(stmt)
            rates = [float(row[0]) for row in result.fetchall()]
            
            logger.debug(f"Retrieved {len(rates)} daily rates for {rate_type}")
            return rates
            
        except Exception as e:
            logger.error(f"Error retrieving treasury rates: {str(e)}")
            return []