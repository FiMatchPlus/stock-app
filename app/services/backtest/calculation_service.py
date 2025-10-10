"""백테스트 포트폴리오 계산 서비스"""

from decimal import Decimal
from typing import Dict, Tuple
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestCalculationService:
    """백테스트 포트폴리오 계산 서비스"""
    
    def _calculate_portfolio_returns(
        self, 
        returns_pivot: pd.DataFrame, 
        weights: Dict[str, float]
    ) -> pd.Series:
        """포트폴리오 수익률 계산 (벡터화)"""
        # 가중치 벡터 생성
        weight_vector = pd.Series(weights)
        
        # 수익률과 가중치 정렬
        common_stocks = returns_pivot.columns.intersection(weight_vector.index)
        returns_aligned = returns_pivot[common_stocks]
        weights_aligned = weight_vector[common_stocks]
        
        # 정규화 (가중치 합이 1이 되도록)
        weights_aligned = weights_aligned / weights_aligned.sum()
        
        # 포트폴리오 수익률 계산 (행렬 곱셈)
        portfolio_returns = returns_aligned.dot(weights_aligned)
        
        return portfolio_returns
    
    def _calculate_portfolio_values(
        self, 
        portfolio_returns: pd.Series, 
        initial_capital: Decimal
    ) -> pd.Series:
        """포트폴리오 가치 계산"""
        # 누적 수익률 계산
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # 포트폴리오 가치 계산
        portfolio_values = cumulative_returns * float(initial_capital)
        
        return portfolio_values
    
    def _calculate_portfolio_values_by_quantity(
        self, 
        price_pivot: pd.DataFrame, 
        quantities: Dict[str, int]
    ) -> Tuple[pd.Series, pd.Series]:
        """수량 기반 포트폴리오 가치 및 수익률 계산"""
        
        # 공통 종목만 필터링 (이미 상위에서 정의됨)
        common_stocks = price_pivot.columns.intersection(quantities.keys())
        price_data = price_pivot[common_stocks]
        
        # 각 종목별 가치 계산 (가격 × 수량)
        portfolio_values_daily = pd.Series(index=price_data.index, dtype=float)
        
        for date in price_data.index:
            daily_value = 0
            for stock in common_stocks:
                if pd.notna(price_data.loc[date, stock]):
                    stock_value = price_data.loc[date, stock] * quantities[stock]
                    daily_value += stock_value
            portfolio_values_daily[date] = daily_value
        
        # 일별 수익률 계산
        portfolio_returns = portfolio_values_daily.pct_change().fillna(0)
        
        return portfolio_values_daily, portfolio_returns

