"""백테스트 메트릭스 변환 서비스"""

from decimal import Decimal
from typing import Optional, Dict, Any
from app.models.schemas import BacktestMetrics
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestMetricsConverterService:
    """백테스트 메트릭스 변환 서비스"""
    
    @staticmethod
    def convert_mongodb_metrics_to_backtest_metrics(
        metrics_data: Optional[Dict[str, Any]]
    ) -> Optional[BacktestMetrics]:
        """
        MongoDB에서 가져온 메트릭스 데이터를 BacktestMetrics 객체로 변환
        
        Args:
            metrics_data: MongoDB에서 가져온 메트릭스 데이터
            
        Returns:
            BacktestMetrics 객체 또는 None
        """
        if not metrics_data:
            return None
            
        try:
            return BacktestMetrics(
                total_return=Decimal(str(metrics_data.get('total_return', 0))),
                annualized_return=Decimal(str(metrics_data.get('annualized_return', 0))),
                volatility=Decimal(str(metrics_data.get('volatility', 0))),
                sharpe_ratio=Decimal(str(metrics_data.get('sharpe_ratio', 0))),
                max_drawdown=Decimal(str(metrics_data.get('max_drawdown', 0))),
                var_95=Decimal(str(metrics_data.get('var_95', 0))),
                var_99=Decimal(str(metrics_data.get('var_99', 0))),
                cvar_95=Decimal(str(metrics_data.get('cvar_95', 0))),
                cvar_99=Decimal(str(metrics_data.get('cvar_99', 0))),
                win_rate=Decimal(str(metrics_data.get('win_rate', 0))),
                profit_loss_ratio=Decimal(str(metrics_data.get('profit_loss_ratio', 0)))
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert metrics data to BacktestMetrics: {str(e)}")
            return None
    
    @staticmethod
    def safe_get_sharpe_ratio(metrics: Optional[BacktestMetrics]) -> float:
        """
        BacktestMetrics에서 샤프 비율을 안전하게 가져오기
        
        Args:
            metrics: BacktestMetrics 객체
            
        Returns:
            샤프 비율 (float) 또는 0.0
        """
        if metrics and metrics.sharpe_ratio is not None:
            try:
                return float(metrics.sharpe_ratio)
            except (ValueError, TypeError):
                logger.warning("Failed to convert sharpe_ratio to float")
        return 0.0
