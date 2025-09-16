"""포트폴리오 백테스트 조회 서비스"""

from datetime import timedelta
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import BacktestResponse, ResultSummary, PortfolioSnapshotResponse, BacktestMetrics
from app.services.backtest_metrics_converter_service import BacktestMetricsConverterService
from app.services.metrics_service import MetricsService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioBacktestQueryService:
    """포트폴리오 백테스트 조회 서비스"""
    
    def __init__(self, metrics_service: Optional[MetricsService] = None):
        self.metrics_service = metrics_service
    
    async def convert_snapshots_to_backtest_responses(
        self, 
        snapshots: List[PortfolioSnapshotResponse]
    ) -> List[BacktestResponse]:
        """
        포트폴리오 스냅샷들을 BacktestResponse로 변환
        
        Args:
            snapshots: 포트폴리오 스냅샷 리스트
            
        Returns:
            BacktestResponse 리스트
        """
        results = []
        
        try:
            logger.info(f"Converting {len(snapshots)} snapshots to BacktestResponse")
            
            for i, snapshot in enumerate(snapshots):
                try:
                    logger.debug(f"Processing snapshot {i+1}/{len(snapshots)}: {snapshot.id}")
                    
                    # 메트릭스 조회 및 변환
                    metrics = await self._get_metrics_for_snapshot(snapshot)
                    
                    # ResultSummary 생성
                    result_summary = self._create_result_summary(snapshot, metrics)
                    
                    # BacktestResponse 생성
                    backtest_response = BacktestResponse(
                        portfolio_snapshot=snapshot,
                        metrics=metrics,
                        result_summary=result_summary,
                        execution_time=snapshot.execution_time or 0.0
                    )
                    
                    results.append(backtest_response)
                    logger.debug(f"Successfully converted snapshot {snapshot.id}")
                    
                except Exception as snapshot_error:
                    logger.error(f"Failed to convert snapshot {snapshot.id}: {str(snapshot_error)}", 
                                exc_info=True)
                    raise
            
            logger.info(f"Successfully converted {len(results)} snapshots")
            return results
            
        except Exception as e:
            logger.error(f"Failed to convert snapshots to BacktestResponse: {str(e)}", 
                        exc_info=True)
            raise
    
    async def _get_metrics_for_snapshot(
        self, 
        snapshot: PortfolioSnapshotResponse
    ) -> Optional[object]:
        """
        스냅샷에 대한 메트릭스 조회 및 변환
        
        Args:
            snapshot: 포트폴리오 스냅샷
            
        Returns:
            BacktestMetrics 객체 또는 None
        """
        if not snapshot.metric_id or not self.metrics_service:
            return None
            
        try:
            metrics_data = await self.metrics_service.get_metrics(snapshot.metric_id)
            return BacktestMetricsConverterService.convert_mongodb_metrics_to_backtest_metrics(metrics_data)
        except Exception as e:
            logger.warning(f"Failed to load metrics for snapshot {snapshot.id}: {str(e)}")
            return None
    
    def _create_result_summary(
        self, 
        snapshot: PortfolioSnapshotResponse, 
        metrics: Optional[object]
    ) -> List[ResultSummary]:
        """
        ResultSummary 리스트 생성
        
        Args:
            snapshot: 포트폴리오 스냅샷
            metrics: 백테스트 메트릭스
            
        Returns:
            ResultSummary 리스트
        """
        result_summary = []
        sharpe_ratio = BacktestMetricsConverterService.safe_get_sharpe_ratio(metrics)
        
        # 백테스트 기간 동안의 일별 데이터 생성 (간소화)
        current_date = snapshot.start_at
        while current_date <= snapshot.end_at:
            result_summary.append(ResultSummary(
                date=current_date.isoformat(),
                portfolio_return=0.0,  # 실제로는 재계산 필요
                portfolio_value=float(snapshot.current_value),  # 실제로는 일별 가치 필요
                sharpe_ratio=sharpe_ratio
            ))
            current_date += timedelta(days=1)
        
        return result_summary
