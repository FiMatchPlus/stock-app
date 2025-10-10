"""백테스트 무위험수익률 관련 서비스"""

from typing import Optional, Dict, Any, Tuple
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import BacktestRequest
from app.services.risk_free_rate_calculator import RiskFreeRateCalculator
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestRiskFreeRateService:
    """백테스트 무위험수익률 관련 서비스"""
    
    async def _get_risk_free_rate_for_period(
        self,
        request: BacktestRequest,
        session: AsyncSession
    ) -> Tuple[Optional[pd.Series], Optional[Dict[str, Any]]]:
        """무위험 수익률 조회 - 트랜잭션 에러 시 독립 세션으로 재시도"""
        try:
            # RiskFreeRateCalculator를 사용해 무위험수익률 계산
            risk_free_calculator = RiskFreeRateCalculator(session)
            
            # 무위험수익률 계산 (연환산 값)
            annual_risk_free_rate = await risk_free_calculator.calculate_risk_free_rate(
                analysis_start=request.start,
                analysis_end=request.end,
                user_risk_free_rate=request.risk_free_rate
            )
            
            # 일별 수익률로 변환
            daily_rate = annual_risk_free_rate / 252.0  # 연환산 -> 일별 (거래일 기준)
            
            # 기간 동안의 시계열 생성
            dates = pd.date_range(start=request.start, end=request.end, freq='D')
            risk_free_rates = pd.Series([daily_rate] * len(dates), index=dates)
            
            # 무위험수익률 정보
            if request.risk_free_rate is not None:
                rate_type = 'USER_PROVIDED'
                selection_reason = 'user_specified'
            else:
                # 분석 기간에 따라 자동 선택된 국고채 타입 추론
                analysis_days = (request.end - request.start).days
                if analysis_days < 365:
                    rate_type = "TB1Y"
                elif analysis_days < 1095:
                    rate_type = "TB3Y"
                else:
                    rate_type = "TB5Y"
                selection_reason = f'auto_selected_by_period_{analysis_days}_days'
            
            risk_free_rate_info = {
                'rate_type': rate_type,
                'annual_rate': annual_risk_free_rate,
                'daily_rate': daily_rate,
                'selection_reason': selection_reason,
                'data_points': len(risk_free_rates),
                'avg_annual_rate': float(annual_risk_free_rate * 100)  # 백분율로 표시
            }
            
            logger.info(
                "Risk-free rate retrieved for backtest",
                rate_type=rate_type,
                data_points=len(risk_free_rates),
                avg_annual_rate=risk_free_rate_info['avg_annual_rate']
            )
            
            return risk_free_rates, risk_free_rate_info
            
        except Exception as e:
            logger.error(f"Failed to get risk-free rate: {str(e)}")
            # 데이터베이스 에러를 상위로 전파하여 백테스트 실패 처리
            raise Exception(f"Database error during risk-free rate retrieval: {str(e)}")

