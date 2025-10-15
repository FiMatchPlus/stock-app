"""거래 규칙 검증 서비스 - 손절/익절 규칙 처리"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np

from app.models.schemas import TradingRule, TradingRules, ExecutionLog
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TradingRulesService:
    """거래 규칙 검증 서비스"""
    
    def __init__(self):
        self.execution_logs = []
    
    async def check_trading_rules(
        self,
        date: datetime,
        portfolio_data: List[Dict[str, Any]],
        individual_values: Dict[str, float],
        individual_prices: Dict[str, float],
        individual_returns: Dict[str, float],
        quantities: Dict[str, int],
        benchmark_returns: Optional[pd.Series] = None,
        rules: Optional[TradingRules] = None
    ) -> Tuple[bool, List[ExecutionLog], str]:
        """
        거래 규칙 검증 및 실행
        
        Returns:
            Tuple[bool, List[ExecutionLog], str]: (실행여부, 실행로그, 상태)
        """
        if not rules:
            return False, [], "COMPLETED"
        
        execution_logs = []
        
        if rules.stopLoss:
            should_stop_loss, stop_logs = await self._check_stop_loss_rules(
                date=date,
                portfolio_data=portfolio_data,
                individual_values=individual_values,
                individual_prices=individual_prices,
                individual_returns=individual_returns,
                quantities=quantities,
                benchmark_returns=benchmark_returns,
                rules=rules.stopLoss
            )
            
            if should_stop_loss:
                execution_logs.extend(stop_logs)
                return True, execution_logs, "LIQUIDATED"
        
        if rules.takeProfit:
            should_take_profit, profit_logs = await self._check_take_profit_rules(
                date=date,
                portfolio_data=portfolio_data,
                individual_values=individual_values,
                individual_prices=individual_prices,
                individual_returns=individual_returns,
                quantities=quantities,
                rules=rules.takeProfit
            )
            
            if should_take_profit:
                execution_logs.extend(profit_logs)
                return True, execution_logs, "LIQUIDATED"
        
        return False, execution_logs, "COMPLETED"
    
    async def _check_stop_loss_rules(
        self,
        date: datetime,
        portfolio_data: List[Dict[str, Any]],
        individual_values: Dict[str, float],
        individual_prices: Dict[str, float],
        individual_returns: Dict[str, float],
        quantities: Dict[str, int],
        benchmark_returns: Optional[pd.Series],
        rules: List[TradingRule]
    ) -> Tuple[bool, List[ExecutionLog]]:
        """손절 규칙 체크"""
        
        execution_logs = []
        current_portfolio_value = sum(individual_values.values())
        
        for rule in rules:
            try:
                if rule.category == "BETA":
                    should_trigger, value, threshold = await self._check_beta_rule(
                        portfolio_data, benchmark_returns, rule.value
                    )
                
                elif rule.category == "MDD":
                    should_trigger, value, threshold = await self._check_mdd_rule(
                        portfolio_data, rule.value
                    )
                
                elif rule.category == "VAR":
                    should_trigger, value, threshold = await self._check_var_rule(
                        portfolio_data, rule.value
                    )
                
                elif rule.category == "LOSS_LIMIT":
                    should_trigger, value, threshold = await self._check_loss_limit_rule(
                        portfolio_data, rule.value
                    )
                
                else:
                    logger.warning(f"Unknown stop loss category: {rule.category}")
                    continue
                
                if should_trigger:
                    execution_logs.append(ExecutionLog(
                        date=date.isoformat(),
                        action="STOP_LOSS",
                        category=rule.category,
                        value=value,
                        threshold=threshold,
                        reason=f"{rule.category} 손절: {value:.4f} > {threshold:.4f}",
                        portfolio_value=current_portfolio_value
                    ))
                    
                    return True, execution_logs
            
            except Exception as e:
                logger.error(f"Error checking {rule.category} rule: {str(e)}")
                continue
        
        return False, execution_logs
    
    async def _check_take_profit_rules(
        self,
        date: datetime,
        portfolio_data: List[Dict[str, Any]],
        individual_values: Dict[str, float],
        individual_prices: Dict[str, float],
        individual_returns: Dict[str, float],
        quantities: Dict[str, int],
        rules: List[TradingRule]
    ) -> Tuple[bool, List[ExecutionLog]]:
        """익절 규칙 체크"""
        
        execution_logs = []
        current_portfolio_value = sum(individual_values.values())
        
        for rule in rules:
            try:
                if rule.category == "ONEPROFIT":
                    should_trigger, value, threshold = await self._check_oneprofit_rule(
                        individual_returns, rule.value
                    )
                
                else:
                    logger.warning(f"Unknown take profit category: {rule.category}")
                    continue
                
                if should_trigger:
                    execution_logs.append(ExecutionLog(
                        date=date.isoformat(),
                        action="TAKE_PROFIT",
                        category=rule.category,
                        value=value,
                        threshold=threshold,
                        reason=f"{rule.category} 익절: {value:.4f} > {threshold:.4f}",
                        portfolio_value=current_portfolio_value
                    ))
                    
                    return True, execution_logs
            
            except Exception as e:
                logger.error(f"Error checking {rule.category} rule: {str(e)}")
                continue
        
        return False, execution_logs
    
    async def _check_beta_rule(
        self,
        portfolio_data: List[Dict[str, Any]],
        benchmark_returns: Optional[pd.Series],
        threshold: float
    ) -> Tuple[bool, float, float]:
        """베타 규칙 체크"""
        
        if not benchmark_returns or benchmark_returns.empty:
            return False, 0.0, threshold
        
        portfolio_returns = pd.Series([
            data['daily_return'] for data in portfolio_data
        ])
        
        if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
            return False, 0.0, threshold
        
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns.iloc[-min_length:]
        benchmark_returns = benchmark_returns.iloc[-min_length:]
        
        try:
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            should_trigger = abs(beta) > threshold
            
            return should_trigger, abs(beta), threshold
        
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            return False, 0.0, threshold
    
    async def _check_mdd_rule(
        self,
        portfolio_data: List[Dict[str, Any]],
        threshold: float
    ) -> Tuple[bool, float, float]:
        """최대 낙폭(MDD) 규칙 체크"""
        
        if len(portfolio_data) < 2:
            return False, 0.0, threshold
        
        portfolio_values = pd.Series([
            data['portfolio_value'] for data in portfolio_data
        ])
        
        cumulative_returns = (1 + portfolio_values.pct_change().fillna(0)).cumprod()
        
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_drawdown = abs(drawdown.min())
        
        should_trigger = max_drawdown > threshold
        
        return should_trigger, max_drawdown, threshold
    
    async def _check_var_rule(
        self,
        portfolio_data: List[Dict[str, Any]],
        threshold: float
    ) -> Tuple[bool, float, float]:
        """VaR 규칙 체크"""
        
        if len(portfolio_data) < 10:
            return False, 0.0, threshold
        
        portfolio_returns = pd.Series([
            data['daily_return'] for data in portfolio_data
        ])
        
        var_95 = np.percentile(portfolio_returns, 5)
        var_95_abs = abs(var_95)
        
        should_trigger = var_95_abs > threshold
        
        return should_trigger, var_95_abs, threshold
    
    async def _check_oneprofit_rule(
        self,
        individual_returns: Dict[str, float],
        threshold: float
    ) -> Tuple[bool, float, float]:
        """단일 종목 수익률 규칙 체크"""
        
        if not individual_returns:
            return False, 0.0, threshold
        
        max_return = max(individual_returns.values())
        
        should_trigger = max_return > threshold
        
        return should_trigger, max_return, threshold
    
    async def _check_loss_limit_rule(
        self,
        portfolio_data: List[Dict[str, Any]],
        threshold: float
    ) -> Tuple[bool, float, float]:
        """손실 한계선 규칙 체크
        
        Args:
            portfolio_data: 포트폴리오 일별 데이터
            threshold: 손실 한계선 (음수, 예: -0.15는 -15%)
            
        Returns:
            Tuple[bool, float, float]: (손절 실행 여부, 현재 총 수익률, 임계값)
        """
        
        if len(portfolio_data) < 1:
            return False, 0.0, threshold
        
        initial_value = portfolio_data[0]['portfolio_value']
        
        current_value = portfolio_data[-1]['portfolio_value']
        
        if initial_value <= 0:
            return False, 0.0, threshold
        
        total_return = (current_value - initial_value) / initial_value
        
        should_trigger = total_return < threshold
        
        return should_trigger, total_return, threshold
