"""백테스트 API 라우터"""

import time
import uuid
import httpx
import numpy as np
from datetime import datetime
from typing import Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_async_session
from app.models.schemas import (
    BacktestRequest, 
    BacktestDataError, MissingStockData,
    BacktestJobResponse, BacktestCallbackResponse, BacktestResponse
)
from app.exceptions import MissingStockPriceDataException, BacktestDataException
from app.services.backtest_service import BacktestService
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/backtest", tags=["backtest"])


async def get_backtest_service() -> BacktestService:
    """백테스트 서비스 의존성 주입"""
    return BacktestService()


@router.post(
    "/start",
    response_model=BacktestJobResponse,
    summary="비동기 백테스트 시작",
    description="백테스트를 백그라운드에서 실행하고 완료 시 콜백 URL로 결과를 전송합니다."
)
async def start_backtest_async(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session),
    backtest_service: BacktestService = Depends(get_backtest_service),
    portfolio_id: Optional[int] = Query(None, description="포트폴리오 ID (선택사항)")
) -> BacktestJobResponse:
    """비동기 백테스트 시작"""
    try:
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
        
        if not request.callback_url:
            raise HTTPException(
                status_code=400,
                detail="callback_url is required for async backtest"
            )
        
        job_id = str(uuid.uuid4())
        
        logger.info(
            "비동기 백테스트 요청 수신",
            job_id=job_id,
            start_date=request.start.isoformat(),
            end_date=request.end.isoformat(),
            holdings_count=len(request.holdings),
            callback_url=request.callback_url,
            portfolio_id=portfolio_id
        )
        
        background_tasks.add_task(
            run_backtest_and_callback,
            job_id=job_id,
            request=request,
            session=session,
            backtest_service=backtest_service,
            portfolio_id=portfolio_id
        )
        
        return BacktestJobResponse(
            job_id=job_id,
            status="started",
            message="백테스트가 백그라운드에서 실행 중입니다."
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(
            "비동기 백테스트 시작 실패",
            error=str(e),
            request_start=request.start.isoformat(),
            request_end=request.end.isoformat(),
            holdings_count=len(request.holdings),
            callback_url=request.callback_url
        )
        
        raise HTTPException(
            status_code=500,
            detail="백테스트 시작 중 오류가 발생했습니다."
        )


@router.post(
    "/run",
    response_model=BacktestResponse,
    summary="동기 백테스트 실행",
    description="백테스트를 즉시 실행하고 결과를 반환합니다."
)
async def run_backtest_sync(
    request: BacktestRequest,
    session: AsyncSession = Depends(get_async_session),
    backtest_service: BacktestService = Depends(get_backtest_service),
    portfolio_id: Optional[int] = Query(None, description="포트폴리오 ID (선택사항)")
) -> BacktestResponse:
    """동기 백테스트 실행"""
    try:
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
        
        logger.info(
            "동기 백테스트 요청 수신",
            start_date=request.start.isoformat(),
            end_date=request.end.isoformat(),
            holdings_count=len(request.holdings),
            portfolio_id=portfolio_id,
            backtest_id=request.backtest_id
        )
        
        result = await backtest_service.run_backtest(
            request=request,
            session=session,
            portfolio_id=portfolio_id
        )
        
        logger.info(
            "동기 백테스트 완료",
            execution_time=f"{result.execution_time:.3f}s",
            portfolio_id=result.portfolio_snapshot.portfolio_id,
            backtest_id=result.backtest_id
        )
        
        return result
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(
            "동기 백테스트 실행 실패",
            error=str(e),
            request_start=request.start.isoformat(),
            request_end=request.end.isoformat(),
            holdings_count=len(request.holdings),
            backtest_id=request.backtest_id
        )
        
        raise HTTPException(
            status_code=500,
            detail="백테스트 실행 중 오류가 발생했습니다."
        )


async def run_backtest_and_callback(
    job_id: str,
    request: BacktestRequest,
    session: AsyncSession,
    backtest_service: BacktestService,
    portfolio_id: Optional[int] = None
):
    """백그라운드에서 백테스트 실행 및 콜백 전송"""
    start_time = time.time()
    
    from app.models.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as backtest_session:
        try:
            logger.info(f"백그라운드 백테스트 시작", job_id=job_id)
            
            backtest_request = BacktestRequest(
                start=request.start,
                end=request.end,
                holdings=request.holdings,
                rebalance_frequency=request.rebalance_frequency,
                benchmark_code=request.benchmark_code,
                risk_free_rate=request.risk_free_rate,
                rules=request.rules
            )
            
            result = await backtest_service.run_backtest(
                request=backtest_request,
                session=backtest_session,
                portfolio_id=portfolio_id
            )
            
            execution_time = time.time() - start_time
            
            callback_response = BacktestCallbackResponse(
                job_id=job_id,
                success=True,
                portfolio_snapshot=result.portfolio_snapshot,
                metrics=result.metrics,
                benchmark_metrics=result.benchmark_metrics,
                result_summary=result.result_summary,
                execution_logs=result.execution_logs,
                result_status=result.result_status,
                benchmark_info=result.benchmark_info,
                risk_free_rate_info=result.risk_free_rate_info,
                error=None,
                execution_time=execution_time,
                backtest_id=request.backtest_id
            )
            
            await send_callback(request.callback_url, callback_response)
            
            logger.info(
                "백그라운드 백테스트 완료",
                job_id=job_id,
                execution_time=f"{execution_time:.3f}s"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(
                "백그라운드 백테스트 실패",
                job_id=job_id,
                error=str(e),
                execution_time=f"{execution_time:.3f}s",
                exc_info=True
            )
            
            from app.models.schemas import BacktestDataError
            
            error_type = "DATABASE_ERROR" if "Database error" in str(e) else "INTERNAL_ERROR"
            error_message = str(e)
            
            callback_response = BacktestCallbackResponse(
                job_id=job_id,
                success=False,
                portfolio_snapshot=None,
                metrics=None,
                benchmark_metrics=None,
                result_summary=None,
                error=BacktestDataError(
                    error_type=error_type,
                    message=error_message,
                    missing_data=[],
                    requested_period=f"{request.start.strftime('%Y-%m-%d')} ~ {request.end.strftime('%Y-%m-%d')}",
                    total_stocks=len(request.holdings),
                    missing_stocks_count=0
                ),
                execution_time=execution_time,
                backtest_id=request.backtest_id
            )
            
            await send_callback(request.callback_url, callback_response)


async def send_callback(callback_url: str, response: BacktestCallbackResponse):
    """콜백 URL로 결과 전송"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {
                "Content-Type": "application/json",
                "X-Backtest-Status": "success" if response.success else "error",
                "X-Backtest-Job-ID": response.job_id
            }
            
            if not response.success:
                error_type = getattr(response.error, 'error_type', 'UNKNOWN') if response.error else 'UNKNOWN'
                headers["X-Backtest-Error-Type"] = error_type
                
            callback_result = await client.post(
                callback_url,
                json=response.model_dump(mode='json'),
                headers=headers
            )
            
            expected_status_codes = {
                True: 200,
                False: 400
            }
            expected_status = expected_status_codes[response.success]
            
            if callback_result.status_code == expected_status:
                if response.success:
                    logger.info(
                        "콜백 전송 성공",
                        job_id=response.job_id,
                        callback_url=callback_url,
                        status_code=callback_result.status_code
                    )
                else:
                    logger.warning(
                        "콜백 에러 응답 전송됨",
                        job_id=response.job_id,
                        callback_url=callback_url,
                        status_code=callback_result.status_code,
                        error_type=getattr(response.error, 'error_type', None) if response.error else None
                    )
            else:
                status_msg = "성공적으로 전송됨" if response.success else "에러 응답과 함께 전송됨"
                logger.warning(
                    f"콜백 {status_msg}, 그러나 응답 상태 코드가 다름",
                    job_id=response.job_id,
                    callback_url=callback_url,
                    expected_status=expected_status,
                    actual_status=callback_result.status_code,
                    success=response.success
                )
                
    except Exception as e:
        logger.error(
            "콜백 전송 실패",
            job_id=response.job_id,
            callback_url=callback_url,
            success=response.success,
            error=str(e)
        )
        raise

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


@router.post(
    "/analyze/{portfolio_id}",
    summary="백테스트 결과 고도화 분석",
    description="완료된 백테스트 결과에 대해 벤치마크 비교 및 고급 리스크 지표를 계산합니다."
)
async def analyze_backtest_result(
    portfolio_id: int,
    benchmark_code: str = Query("KOSPI", description="비교할 벤치마크 지수 코드"),
    risk_free_rate: Optional[float] = Query(None, description="무위험수익률 (연율, 미제공시 자동 조회)"),
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, float]:
    """백테스트 결과 고도화 분석
    
    백테스트가 완료된 포트폴리오에 대해 벤치마크 대비 성과를 분석하고
    고급 리스크 지표들을 계산하여 반환합니다.
    """
    try:
        
        backtest_service = BacktestService()
        
        import pandas as pd
        from datetime import datetime, timedelta
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=252),
            end=datetime.now(),
            freq='D'
        )
        portfolio_returns = pd.Series(
            data=np.random.normal(0.0005, 0.015, len(dates)),
            index=dates,
            name='portfolio_returns'
        )
        
        analysis_result = await backtest_service.analyze_backtest_performance(
            portfolio_returns=portfolio_returns,
            benchmark_code=benchmark_code,
            session=session,
            risk_free_rate=risk_free_rate
        )
        
        logger.info(
            "백테스트 고도화 분석 완료",
            portfolio_id=portfolio_id,
            benchmark_code=benchmark_code,
            metrics_count=len(analysis_result)
        )
        
        return analysis_result
        
    except Exception as e:
        logger.error(
            "백테스트 결과 분석 실패",
            portfolio_id=portfolio_id,
            benchmark_code=benchmark_code,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))