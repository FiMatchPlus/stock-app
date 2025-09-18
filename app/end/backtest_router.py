"""백테스트 API 라우터"""

import time
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_async_session
from app.models.schemas import (
    BacktestRequest, BacktestResponse, BacktestErrorResponse, 
    BacktestDataError, MissingStockData
)
from app.exceptions import MissingStockPriceDataException, BacktestDataException
from app.services.backtest_service import BacktestService
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 라우터 생성
router = APIRouter(prefix="/backtest", tags=["backtest"])


# 의존성 주입 함수들
async def get_backtest_service() -> BacktestService:
    """백테스트 서비스 의존성 주입"""
    return BacktestService()


@router.post(
    "/run",
    response_model=BacktestResponse,
    responses={
        400: {"model": BacktestErrorResponse, "description": "잘못된 요청 또는 데이터 누락"},
        500: {"model": BacktestErrorResponse, "description": "서버 내부 오류"}
    },
    summary="백테스트 실행",
    description="포트폴리오 백테스트를 실행하고 성과 지표를 반환합니다."
)
async def run_backtest(
    request: BacktestRequest,
    session: AsyncSession = Depends(get_async_session),
    backtest_service: BacktestService = Depends(get_backtest_service),
    portfolio_id: Optional[int] = Query(None, description="포트폴리오 ID (선택사항)")
) -> BacktestResponse:
    """백테스트 실행"""
    start_time = time.time()
    
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
        
    except MissingStockPriceDataException as e:
        execution_time = time.time() - start_time
        
        logger.warning(
            "Backtest failed due to missing stock price data",
            missing_stocks_count=e.missing_stocks_count,
            total_stocks=e.total_stocks,
            requested_period=e.requested_period
        )
        
        # 구조화된 오류 응답 생성
        missing_data = [
            MissingStockData(**stock_data) 
            for stock_data in e.missing_stocks
        ]
        
        error_response = BacktestErrorResponse(
            success=False,
            error=BacktestDataError(
                error_type=e.error_type,
                message=e.message,
                missing_data=missing_data,
                requested_period=e.requested_period,
                total_stocks=e.total_stocks,
                missing_stocks_count=e.missing_stocks_count
            ),
            execution_time=execution_time
        )
        
        raise HTTPException(
            status_code=400,
            detail=error_response.model_dump()
        )
        
    except BacktestDataException as e:
        execution_time = time.time() - start_time
        
        logger.warning(f"Backtest data error: {str(e)}")
        
        # 기본 데이터 오류 응답
        error_response = BacktestErrorResponse(
            success=False,
            error=BacktestDataError(
                error_type=e.error_type,
                message=e.message,
                missing_data=[],
                requested_period=f"{request.start.strftime('%Y-%m-%d')} ~ {request.end.strftime('%Y-%m-%d')}",
                total_stocks=len(request.holdings),
                missing_stocks_count=0
            ),
            execution_time=execution_time
        )
        
        raise HTTPException(
            status_code=400,
            detail=error_response.model_dump()
        )
        
    except ValueError as e:
        execution_time = time.time() - start_time
        logger.warning(f"Backtest validation error: {str(e)}")
        
        error_response = BacktestErrorResponse(
            success=False,
            error=BacktestDataError(
                error_type="VALIDATION_ERROR",
                message=str(e),
                missing_data=[],
                requested_period=f"{request.start.strftime('%Y-%m-%d')} ~ {request.end.strftime('%Y-%m-%d')}",
                total_stocks=len(request.holdings),
                missing_stocks_count=0
            ),
            execution_time=execution_time
        )
        
        raise HTTPException(status_code=400, detail=error_response.model_dump())
        
    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        logger.error(
            "Backtest execution failed",
            error=str(e),
            error_type=type(e).__name__,
            request_start=request.start,
            request_end=request.end,
            holdings_count=len(request.holdings),
            portfolio_id=portfolio_id
        )
        
        # 일반적인 서버 오류 응답
        error_response = BacktestErrorResponse(
            success=False,
            error=BacktestDataError(
                error_type="INTERNAL_ERROR",
                message="백테스트 실행 중 예상치 못한 오류가 발생했습니다.",
                missing_data=[],
                requested_period=f"{request.start.strftime('%Y-%m-%d')} ~ {request.end.strftime('%Y-%m-%d')}",
                total_stocks=len(request.holdings),
                missing_stocks_count=0
            ),
            execution_time=execution_time
        )
        
        raise HTTPException(status_code=500, detail=error_response.model_dump())

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
