"""무위험수익률 계산 서비스"""

from typing import Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.stock import RiskFreeRate
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RiskFreeRateService:
    """무위험수익률 계산 서비스"""
    
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
            # 1. 사용자가 직접 지정한 경우
            if user_risk_free_rate is not None:
                logger.info(f"Using user-specified risk-free rate: {user_risk_free_rate}")
                return user_risk_free_rate
            
            # 2. 분석 기간 길이에 따른 국고채 선택
            rate_type = self._select_treasury_bond_type(analysis_start, analysis_end)
            logger.info(f"Selected treasury bond type: {rate_type} for analysis period: {(analysis_end - analysis_start).days} days")
            
            # 3. 해당 기간의 국고채 수익률 조회 및 평균 계산
            daily_returns = await self._get_treasury_daily_returns(rate_type, analysis_start, analysis_end)
            
            if not daily_returns:
                logger.warning(f"No treasury data available for {rate_type}, using default 0.0")
                return 0.0
            
            # 4. 일별 수익률의 평균을 연환산으로 변환
            average_daily_return = sum(daily_returns) / len(daily_returns)
            annualized_rate = average_daily_return * 252  # 252 거래일 기준
            
            logger.info(f"Calculated risk-free rate: {annualized_rate:.4f} (from {len(daily_returns)} daily returns)")
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
            str: 국고채 타입 (Treasury1Y, Treasury3Y, Treasury5Y)
        """
        analysis_days = (end - start).days
        
        if analysis_days < 365:  # 1년 미만
            return "Treasury1Y"
        elif analysis_days < 1095:  # 3년 미만
            return "Treasury3Y"
        else:  # 3년 이상
            return "Treasury5Y"
    
    async def _get_treasury_daily_returns(
        self,
        rate_type: str,
        start: datetime,
        end: datetime
    ) -> list[float]:
        """특정 기간의 국고채 일별 수익률 조회
        
        Args:
            rate_type: 국고채 타입
            start: 시작일
            end: 종료일
            
        Returns:
            list[float]: 일별 수익률 리스트
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