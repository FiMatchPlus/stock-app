"""무위험 수익률 자동 결정 및 조회 서비스"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, asc, desc

from app.models.stock import RiskFreeRate
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RiskFreeRateService:
    """무위험 수익률 자동 결정 및 조회 서비스"""
    
    def __init__(self):
        # 금리 유형별 특성 정의 (확장 가능)
        self.rate_characteristics = {
            'TB1Y': {
                'maturity_days': 365,
                'description': '국고채 1년',
                'priority': 1,  # 백테스트 기간과 가장 가까운 만기
                'suitable_for': ['short_term', 'medium_term']
            },
            'TB3Y': {
                'maturity_days': 1095,
                'description': '국고채 3년', 
                'priority': 2,
                'suitable_for': ['medium_term', 'long_term']
            },
            'TB5Y': {
                'maturity_days': 1825,
                'description': '국고채 5년',
                'priority': 3,
                'suitable_for': ['long_term']
            }
        }
        
        # 백테스트 기간 분류 기준
        self.period_thresholds = {
            'short_term': 180,    # 6개월 이하
            'medium_term': 730,   # 2년 이하  
            'long_term': 1825     # 5년 이하
        }
    
    async def determine_risk_free_rate_type(
        self,
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession
    ) -> Tuple[str, Dict[str, any]]:
        """
        백테스트 기간을 분석하여 적절한 무위험 수익률 유형 결정
        
        Args:
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            session: 데이터베이스 세션
            
        Returns:
            Tuple[str, Dict[str, any]]: (선택된 금리 유형, 결정 정보)
        """
        try:
            # 백테스트 기간 계산 (일 단위)
            backtest_days = (end_date - start_date).days
            
            # 사용 가능한 금리 유형 조회
            available_rates = await self.get_available_rate_types(session)
            
            # 기간 분류
            period_classification = self._classify_backtest_period(backtest_days)
            
            # 적합한 금리 유형 후보 선택
            suitable_rates = self._find_suitable_rates(
                period_classification, available_rates
            )
            
            # 데이터 가용성 검증
            rate_availability = await self._check_rate_availability(
                suitable_rates, start_date, end_date, session
            )
            
            # 최적 금리 유형 결정
            selected_rate_type, selection_reason = self._select_best_rate(
                suitable_rates, rate_availability, backtest_days
            )
            
            # 결정 정보 구성
            decision_info = {
                'backtest_days': backtest_days,
                'period_classification': period_classification,
                'available_rates': available_rates,
                'suitable_rates': suitable_rates,
                'rate_availability': rate_availability,
                'selection_reason': selection_reason,
                'selected_rate_characteristics': self.rate_characteristics.get(selected_rate_type, {})
            }
            
            logger.info(
                "Risk-free rate type determined",
                selected_rate_type=selected_rate_type,
                backtest_days=backtest_days,
                period_classification=period_classification,
                selection_reason=selection_reason
            )
            
            return selected_rate_type, decision_info
            
        except Exception as e:
            logger.error(f"Failed to determine risk-free rate type: {str(e)}")
            # 기본값 반환
            return "TB3Y", {
                'backtest_days': (end_date - start_date).days,
                'period_classification': 'medium_term',
                'selection_reason': 'fallback_to_default',
                'error': str(e)
            }
    
    def _classify_backtest_period(self, backtest_days: int) -> str:
        """백테스트 기간을 분류"""
        if backtest_days <= self.period_thresholds['short_term']:
            return 'short_term'
        elif backtest_days <= self.period_thresholds['medium_term']:
            return 'medium_term'
        else:
            return 'long_term'
    
    def _find_suitable_rates(
        self, 
        period_classification: str, 
        available_rates: List[str]
    ) -> List[str]:
        """기간에 적합한 금리 유형 찾기"""
        suitable_rates = []
        
        for rate_type in available_rates:
            if rate_type in self.rate_characteristics:
                characteristics = self.rate_characteristics[rate_type]
                if period_classification in characteristics['suitable_for']:
                    suitable_rates.append(rate_type)
        
        # 우선순위별 정렬
        suitable_rates.sort(
            key=lambda x: self.rate_characteristics.get(x, {}).get('priority', 999)
        )
        
        return suitable_rates
    
    async def _check_rate_availability(
        self,
        rate_types: List[str],
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession
    ) -> Dict[str, Dict[str, any]]:
        """각 금리 유형별 데이터 가용성 확인"""
        availability = {}
        
        for rate_type in rate_types:
            try:
                # 해당 기간의 데이터 존재 여부 확인
                query = (
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
                
                result = await session.execute(query)
                rate_data = result.scalars().all()
                
                # 데이터 커버리지 계산
                expected_days = (end_date - start_date).days + 1
                actual_days = len(rate_data)
                coverage_ratio = actual_days / expected_days if expected_days > 0 else 0
                
                # 데이터 품질 평가
                is_available = coverage_ratio >= 0.8  # 80% 이상 커버리지
                
                availability[rate_type] = {
                    'is_available': is_available,
                    'coverage_ratio': coverage_ratio,
                    'expected_days': expected_days,
                    'actual_days': actual_days,
                    'data_range': {
                        'start': rate_data[0].datetime if rate_data else None,
                        'end': rate_data[-1].datetime if rate_data else None
                    } if rate_data else None
                }
                
            except Exception as e:
                logger.warning(f"Failed to check availability for {rate_type}: {str(e)}")
                availability[rate_type] = {
                    'is_available': False,
                    'error': str(e)
                }
        
        return availability
    
    def _select_best_rate(
        self,
        suitable_rates: List[str],
        rate_availability: Dict[str, Dict[str, any]],
        backtest_days: int
    ) -> Tuple[str, str]:
        """최적의 금리 유형 선택"""
        
        # 1순위: 데이터가 사용 가능한 금리 중 우선순위가 가장 높은 것
        available_rates = [
            rate for rate in suitable_rates 
            if rate_availability.get(rate, {}).get('is_available', False)
        ]
        
        if available_rates:
            selected_rate = available_rates[0]  # 이미 우선순위별 정렬됨
            return selected_rate, "best_available_data"
        
        # 2순위: 백테스트 기간과 만기가 가장 가까운 금리
        closest_rate = min(
            suitable_rates,
            key=lambda x: abs(
                backtest_days - self.rate_characteristics.get(x, {}).get('maturity_days', backtest_days)
            )
        )
        
        return closest_rate, "closest_maturity"
    
    async def get_risk_free_rate(
        self,
        rate_type: str,
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession
    ) -> pd.Series:
        """
        지정된 금리 유형의 기간별 수익률 조회
        
        Args:
            rate_type: 금리 유형 (TB1Y, TB3Y, TB5Y)
            start_date: 시작일
            end_date: 종료일
            session: 데이터베이스 세션
            
        Returns:
            pd.Series: 일별 금리 시계열 (날짜가 인덱스)
        """
        try:
            # 금리 데이터 조회
            query = (
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
            
            result = await session.execute(query)
            rate_data = result.scalars().all()
            
            if not rate_data:
                logger.warning(
                    f"No risk-free rate data found",
                    rate_type=rate_type,
                    start_date=start_date,
                    end_date=end_date
                )
                return pd.Series(dtype=float)
            
            # DataFrame으로 변환
            data = []
            for rate in rate_data:
                data.append({
                    'datetime': rate.datetime,
                    'rate': float(rate.rate),
                    'daily_rate': float(rate.daily_rate)
                })
            
            df = pd.DataFrame(data)
            df = df.set_index('datetime')
            df = df.sort_index()
            
            # 일별 금리 반환 (연율을 일별로 변환)
            daily_rates = df['daily_rate']
            
            logger.info(
                f"Risk-free rate retrieved",
                rate_type=rate_type,
                data_points=len(daily_rates),
                start_date=daily_rates.index[0] if not daily_rates.empty else None,
                end_date=daily_rates.index[-1] if not daily_rates.empty else None,
                avg_rate=float(daily_rates.mean()) if not daily_rates.empty else 0.0
            )
            
            return daily_rates
            
        except Exception as e:
            logger.error(
                f"Failed to get risk-free rate",
                rate_type=rate_type,
                error=str(e)
            )
            return pd.Series(dtype=float)
    
    async def get_available_rate_types(self, session: AsyncSession) -> List[str]:
        """사용 가능한 금리 유형 목록 조회"""
        try:
            query = select(RiskFreeRate.rate_type).distinct()
            result = await session.execute(query)
            rate_types = [row[0] for row in result.fetchall()]
            
            return sorted(rate_types)
            
        except Exception as e:
            logger.error(f"Failed to get available rate types: {str(e)}")
            return ["TB1Y", "TB3Y", "TB5Y"]  # 기본값
    
    async def get_rate_info(
        self,
        rate_type: str,
        session: AsyncSession
    ) -> Optional[Dict[str, any]]:
        """금리 유형 정보 조회"""
        try:
            # 가장 최근 데이터 조회
            query = (
                select(RiskFreeRate)
                .where(RiskFreeRate.rate_type == rate_type)
                .order_by(desc(RiskFreeRate.datetime))
                .limit(1)
            )
            
            result = await session.execute(query)
            latest_rate = result.scalar_one_or_none()
            
            if not latest_rate:
                return None
            
            # 데이터 범위 조회
            min_max_query = (
                select(RiskFreeRate.datetime)
                .where(RiskFreeRate.rate_type == rate_type)
                .order_by(asc(RiskFreeRate.datetime))
                .limit(1)
            )
            
            result = await session.execute(min_max_query)
            min_data = result.fetchone()
            
            max_query = (
                select(RiskFreeRate.datetime)
                .where(RiskFreeRate.rate_type == rate_type)
                .order_by(desc(RiskFreeRate.datetime))
                .limit(1)
            )
            
            result = await session.execute(max_query)
            max_data = result.fetchone()
            
            return {
                'rate_type': rate_type,
                'latest_rate': float(latest_rate.rate),
                'latest_daily_rate': float(latest_rate.daily_rate),
                'latest_date': latest_rate.datetime,
                'source': latest_rate.source,
                'data_range': {
                    'start_date': min_data[0] if min_data else None,
                    'end_date': max_data[0] if max_data else None
                },
                'characteristics': self.rate_characteristics.get(rate_type, {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get rate info: {str(e)}")
            return None
