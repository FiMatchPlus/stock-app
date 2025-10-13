"""백테스트 성과 지표 계산 서비스"""

import asyncio
from decimal import Decimal
from typing import List, Dict, Any, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from app.models.schemas import BacktestMetrics
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestMetricsService:
    """백테스트 성과 지표 계산 서비스"""
    
    # ThreadPoolExecutor는 __init__에서 초기화되어야 함
    thread_pool: ThreadPoolExecutor
    
    async def _calculate_metrics(self, result_summary: List[Dict[str, Any]]) -> BacktestMetrics:
        """성과 지표 계산 (최적화된 버전)"""
        if not result_summary:
            raise ValueError("백테스트 결과 데이터가 없어 성과 지표를 계산할 수 없습니다.")
        
        # 포트폴리오 가치 기반 일별 수익률 계산
        returns = []
        prev_value = None
        initial_value = None
        final_value = None
        
        for rs in result_summary:
            portfolio_value = rs['portfolio_value']
            
            if initial_value is None:
                initial_value = portfolio_value
            final_value = portfolio_value
            
            if prev_value is None:
                # 첫날은 수익률 0
                daily_return = 0.0
            else:
                # 전일 대비 수익률
                daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
            
            returns.append(daily_return)
            prev_value = portfolio_value
        
        returns = np.array(returns)
        
        # 기본 통계량 계산 - 초기/최종 가치 기반
        total_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0.0
        
        logger.info(f"Portfolio return calculation: initial={initial_value:,.0f}, final={final_value:,.0f}, return={total_return*100:.2f}%")
        
        # 연환산 수익률
        days = len(returns)
        annualized_return = (1 + total_return) ** (252 / days) - 1
        
        # 변동성 (연환산)
        volatility = returns.std() * np.sqrt(252)
        
        # 샤프 비율 (무위험 수익률 고려)
        # TODO: 무위험 수익률을 매개변수로 받아서 계산하도록 수정 필요
        risk_free_rate = 0.0  # 임시로 0% 사용
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 최대 낙폭 계산
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # VaR/CVaR 계산 (병렬 처리)
        var_95, var_99, cvar_95, cvar_99 = await self._calculate_var_cvar(returns)
        
        # 승률 및 손익비
        win_rate, profit_loss_ratio = self._calculate_win_loss_metrics(returns)
        
        return BacktestMetrics(
            total_return=Decimal(str(total_return)),
            annualized_return=Decimal(str(annualized_return)),
            volatility=Decimal(str(volatility)),
            sharpe_ratio=Decimal(str(sharpe_ratio)),
            max_drawdown=Decimal(str(max_drawdown)),
            var_95=Decimal(str(var_95)),
            var_99=Decimal(str(var_99)),
            cvar_95=Decimal(str(cvar_95)),
            cvar_99=Decimal(str(cvar_99)),
            win_rate=Decimal(str(win_rate)),
            profit_loss_ratio=Decimal(str(profit_loss_ratio))
        )
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """최대 낙폭 계산"""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    async def _calculate_var_cvar(
        self, 
        returns: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """VaR/CVaR 계산 (병렬 처리)"""
        loop = asyncio.get_event_loop()
        
        # 정렬된 수익률 배열 생성 (공통 사용)
        sorted_returns = np.sort(returns)
        
        # VaR 계산 (병렬)
        var_95_task = loop.run_in_executor(
            self.thread_pool, 
            self._calculate_var, 
            sorted_returns, 0.05
        )
        var_99_task = loop.run_in_executor(
            self.thread_pool, 
            self._calculate_var, 
            sorted_returns, 0.01
        )
        
        # CVaR 계산 (병렬)
        cvar_95_task = loop.run_in_executor(
            self.thread_pool, 
            self._calculate_cvar, 
            sorted_returns, 0.05
        )
        cvar_99_task = loop.run_in_executor(
            self.thread_pool, 
            self._calculate_cvar, 
            sorted_returns, 0.01
        )
        
        # 결과 대기
        var_95, var_99, cvar_95, cvar_99 = await asyncio.gather(
            var_95_task, var_99_task, cvar_95_task, cvar_99_task
        )
        
        return var_95, var_99, cvar_95, cvar_99
    
    def _calculate_var(self, sorted_returns: np.ndarray, confidence_level: float) -> float:
        """VaR 계산"""
        index = int(confidence_level * len(sorted_returns))
        return sorted_returns[index]
    
    def _calculate_cvar(self, sorted_returns: np.ndarray, confidence_level: float) -> float:
        """CVaR 계산"""
        index = int(confidence_level * len(sorted_returns))
        if index == 0:
            return 0.0
        return sorted_returns[:index].mean() if not np.isnan(sorted_returns[:index].mean()) else 0.0
    
    def _calculate_win_loss_metrics(self, returns: np.ndarray) -> Tuple[float, float]:
        """승률 및 손익비 계산"""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0
        
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        return win_rate, profit_loss_ratio

