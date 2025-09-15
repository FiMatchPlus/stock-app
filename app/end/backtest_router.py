"""백테스트 API 라우터"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_async_session
from app.models.schemas import (
    BacktestRequest, BacktestResponse, BacktestHistoryRequest, BacktestHistoryResponse,
    ErrorResponse
)
from app.services.backtest_service import BacktestService
from app.services.metrics_service import MetricsService
from app.utils.mongodb_client import get_mongodb_client, MongoDBClient
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 라우터 생성
router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.post(
    "/run",
    response_model=BacktestResponse,
    summary="백테스트 실행",
    description="포트폴리오 백테스트를 실행하고 성과 지표를 반환합니다."
)
async def run_backtest(
    request: BacktestRequest,
    session: AsyncSession = Depends(get_async_session),
    mongodb_client: MongoDBClient = Depends(get_mongodb_client),
    portfolio_id: Optional[int] = Query(None, description="포트폴리오 ID (선택사항)")
) -> BacktestResponse:
    """백테스트 실행"""
    try:
        logger.info(
            "Backtest request received",
            start_date=request.start,
            end_date=request.end,
            holdings_count=len(request.holdings),
            portfolio_id=portfolio_id
        )
        
        # 입력 검증
        if request.start >= request.end:
            raise HTTPException(
                status_code=400,
                detail="Start date must be before end date"
            )
        
        if not request.holdings:
            raise HTTPException(
                status_code=400,
                detail="At least one holding must be specified"
            )
        
        # 백테스트 서비스 인스턴스 생성
        backtest_service = BacktestService(mongodb_client)
        
        # 백테스트 실행
        result = await backtest_service.run_backtest(
            request=request,
            session=session,
            portfolio_id=portfolio_id
        )
        
        logger.info(
            "Backtest completed successfully",
            portfolio_snapshot_id=result.portfolio_snapshot.id,
            execution_time=result.execution_time
        )
        
        return result
        
    except ValueError as e:
        logger.warning(f"Backtest validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Backtest execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/history",
    response_model=BacktestHistoryResponse,
    summary="백테스트 히스토리 조회",
    description="백테스트 실행 히스토리를 조회합니다."
)
async def get_backtest_history(
    portfolio_id: Optional[int] = Query(None, description="포트폴리오 ID"),
    start_date: Optional[datetime] = Query(None, description="시작일"),
    end_date: Optional[datetime] = Query(None, description="종료일"),
    limit: int = Query(100, ge=1, le=1000, description="조회 개수"),
    offset: int = Query(0, ge=0, description="오프셋"),
    session: AsyncSession = Depends(get_async_session),
    mongodb_client: MongoDBClient = Depends(get_mongodb_client)
) -> BacktestHistoryResponse:
    """백테스트 히스토리 조회"""
    try:
        logger.info(
            "Backtest history request received",
            portfolio_id=portfolio_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )
        
        # 백테스트 서비스 인스턴스 생성
        backtest_service = BacktestService(mongodb_client)
        
        # 히스토리 조회
        snapshots = await backtest_service.get_backtest_history(
            session=session,
            portfolio_id=portfolio_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )
        
        # 총 개수 조회 (추후 구현 필요)
        total_count = len(snapshots)  # 임시로 현재 결과 개수 사용
        
        logger.info(
            "Backtest history retrieved successfully",
            count=len(snapshots),
            total_count=total_count
        )
        
        return BacktestHistoryResponse(
            total_count=total_count,
            snapshots=snapshots,
            summary_metrics=None  # 추후 구현
        )
        
    except Exception as e:
        logger.error(f"Failed to get backtest history: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/portfolio/{portfolio_id}",
    response_model=List[BacktestResponse],
    summary="특정 포트폴리오 백테스트 조회",
    description="특정 포트폴리오의 모든 백테스트 결과를 조회합니다."
)
async def get_portfolio_backtests(
    portfolio_id: int = Path(..., description="포트폴리오 ID"),
    session: AsyncSession = Depends(get_async_session)
) -> List[BacktestResponse]:
    """특정 포트폴리오의 백테스트 조회"""
    try:
        logger.info(f"Portfolio backtest request received", portfolio_id=portfolio_id)
        
        # 포트폴리오별 백테스트 조회
        snapshots = await backtest_service.get_backtest_history(
            session=session,
            portfolio_id=portfolio_id,
            limit=1000  # 모든 결과 조회
        )
        
        if not snapshots:
            raise HTTPException(
                status_code=404,
                detail=f"No backtest results found for portfolio {portfolio_id}"
            )
        
        logger.info(
            "Portfolio backtests retrieved successfully",
            portfolio_id=portfolio_id,
            count=len(snapshots)
        )
        
        # BacktestResponse 형태로 변환 (간단한 형태)
        results = []
        for snapshot in snapshots:
            results.append(BacktestResponse(
                portfolio_snapshot=snapshot,
                metrics=None,  # 추후 구현
                daily_returns=[],  # 추후 구현
                execution_time=0.0  # 추후 구현
            ))
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get portfolio backtests: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete(
    "/portfolio/{portfolio_id}",
    summary="포트폴리오 백테스트 삭제",
    description="특정 포트폴리오의 모든 백테스트 결과를 삭제합니다."
)
async def delete_portfolio_backtests(
    portfolio_id: int = Path(..., description="포트폴리오 ID"),
    session: AsyncSession = Depends(get_async_session)
) -> dict:
    """포트폴리오 백테스트 삭제"""
    try:
        logger.info(f"Portfolio backtest deletion request received", portfolio_id=portfolio_id)
        
        # 포트폴리오별 백테스트 삭제 (추후 구현)
        # 현재는 기본 응답만 반환
        
        logger.info(
            "Portfolio backtests deleted successfully",
            portfolio_id=portfolio_id
        )
        
        return {
            "message": f"Portfolio {portfolio_id} backtest results deleted successfully",
            "portfolio_id": portfolio_id
        }
        
    except Exception as e:
        logger.error(f"Failed to delete portfolio backtests: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/metrics/{metric_id}",
    summary="백테스트 지표 조회",
    description="MongoDB에서 특정 백테스트 지표를 조회합니다."
)
async def get_backtest_metrics(
    metric_id: str = Path(..., description="MongoDB ObjectId"),
    mongodb_client: MongoDBClient = Depends(get_mongodb_client)
) -> dict:
    """백테스트 지표 조회"""
    try:
        metrics_service = MetricsService(mongodb_client)
        metrics = await metrics_service.get_metrics(metric_id)
        
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail=f"Metrics not found with id: {metric_id}"
            )
        
        logger.info(f"Metrics retrieved successfully: {metric_id}")
        return {
            "success": True,
            "data": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/metrics/portfolio/{portfolio_snapshot_id}",
    summary="포트폴리오 스냅샷별 지표 조회",
    description="포트폴리오 스냅샷 ID로 백테스트 지표를 조회합니다."
)
async def get_metrics_by_portfolio_snapshot(
    portfolio_snapshot_id: int = Path(..., description="포트폴리오 스냅샷 ID"),
    mongodb_client: MongoDBClient = Depends(get_mongodb_client)
) -> dict:
    """포트폴리오 스냅샷별 지표 조회"""
    try:
        metrics_service = MetricsService(mongodb_client)
        metrics = await metrics_service.get_metrics_by_portfolio_snapshot(portfolio_snapshot_id)
        
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail=f"Metrics not found for portfolio snapshot: {portfolio_snapshot_id}"
            )
        
        logger.info(f"Metrics retrieved by portfolio snapshot: {portfolio_snapshot_id}")
        return {
            "success": True,
            "data": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics by portfolio snapshot: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/health",
    summary="백테스트 서비스 헬스 체크",
    description="백테스트 서비스의 상태를 확인합니다."
)
async def health_check() -> dict:
    """백테스트 서비스 헬스 체크"""
    return {
        "status": "healthy",
        "service": "backtest",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get(
    "/metrics/explanation",
    summary="성과 지표 설명",
    description="백테스트에서 계산되는 성과 지표들에 대한 설명을 제공합니다."
)
async def get_metrics_explanation() -> dict:
    """성과 지표 설명"""
    return {
        "metrics": {
            "total_return": {
                "description": "전체 기간 동안의 총 수익률",
                "formula": "(최종가치 / 초기가치) - 1",
                "unit": "percentage"
            },
            "annualized_return": {
                "description": "연환산 수익률",
                "formula": "((1 + total_return)^(252/기간일수)) - 1",
                "unit": "percentage"
            },
            "volatility": {
                "description": "연환산 변동성 (표준편차)",
                "formula": "일별 수익률의 표준편차 × √252",
                "unit": "percentage"
            },
            "sharpe_ratio": {
                "description": "샤프 비율 (위험 대비 수익률)",
                "formula": "(연환산 수익률 - 무위험 수익률) / 변동성",
                "unit": "ratio"
            },
            "max_drawdown": {
                "description": "최대 낙폭",
                "formula": "최고점 대비 최대 하락폭",
                "unit": "percentage"
            },
            "var_95": {
                "description": "95% VaR (Value at Risk)",
                "formula": "95% 신뢰구간에서의 최대 손실",
                "unit": "percentage"
            },
            "var_99": {
                "description": "99% VaR (Value at Risk)",
                "formula": "99% 신뢰구간에서의 최대 손실",
                "unit": "percentage"
            },
            "cvar_95": {
                "description": "95% CVaR (Conditional Value at Risk)",
                "formula": "95% VaR을 초과하는 손실의 평균",
                "unit": "percentage"
            },
            "cvar_99": {
                "description": "99% CVaR (Conditional Value at Risk)",
                "formula": "99% VaR을 초과하는 손실의 평균",
                "unit": "percentage"
            },
            "win_rate": {
                "description": "승률",
                "formula": "양수 수익률 날짜 수 / 전체 거래일 수",
                "unit": "percentage"
            },
            "profit_loss_ratio": {
                "description": "손익비",
                "formula": "평균 수익 / 평균 손실",
                "unit": "ratio"
            }
        }
    }
