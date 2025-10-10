"""포트폴리오 분석 API 라우터 (도메인 분리)"""

import time
import json
import uuid
import httpx
from typing import Optional, List, Dict
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_async_session
from app.models.schemas import (
    AnalysisRequest,
    PortfolioAnalysisResponse,
    AnalysisJobResponse,
    AnalysisCallbackResponse,
    AnalysisCallbackMetadata,
    BenchmarkPriceResponse,
    RiskFreeRateResponse,
    ErrorResponse
)
from app.services.analysis_service import AnalysisService
from app.services.prices.data_collection_service import DataCollectionService
from app.repositories.benchmark_repository import BenchmarkRepository
from app.repositories.risk_free_rate_repository import RiskFreeRateRepository
from app.utils.logger import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix="/analysis", tags=["analysis"])


async def get_analysis_service() -> AnalysisService:
    return AnalysisService()




async def get_data_collection_service() -> DataCollectionService:
    return DataCollectionService()




@router.post(
    "/start",
    response_model=AnalysisJobResponse,
    summary="비동기 포트폴리오 분석 시작",
    description="포트폴리오 분석을 백그라운드에서 실행하고 완료 시 콜백 URL로 결과를 전송합니다."
)
async def start_analysis_async(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session),
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisJobResponse:
    """비동기 포트폴리오 분석 시작"""
    try:
        # 입력 검증
        if not request.holdings:
            raise HTTPException(
                status_code=400,
                detail="At least one holding must be specified"
            )
        
        # 콜백 URL 필수 확인
        if not request.callback_url:
            raise HTTPException(
                status_code=400,
                detail="callback_url is required for async analysis"
            )
        
        # 작업 ID 생성
        job_id = str(uuid.uuid4())
        
        logger.info(
            "Async analysis request received",
            job_id=job_id,
            holdings_count=len(request.holdings),
            lookback_years=request.lookback_years,
            benchmark=request.benchmark,
            callback_url=request.callback_url
        )
        
        # 백그라운드 작업으로 분석 실행
        background_tasks.add_task(
            run_analysis_and_callback,
            job_id=job_id,
            request=request,
            session=session,
            analysis_service=analysis_service
        )
        
        return AnalysisJobResponse(
            job_id=job_id,
            status="started",
            message="포트폴리오 분석이 백그라운드에서 실행 중입니다."
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(
            "Failed to start async analysis",
            error=str(e),
            holdings_count=len(request.holdings) if request.holdings else 0,
            callback_url=request.callback_url
        )
        
        raise HTTPException(
            status_code=500,
            detail="분석 시작 중 오류가 발생했습니다."
        )


 


async def run_analysis_and_callback(
    job_id: str,
    request: AnalysisRequest,
    session: AsyncSession,
    analysis_service: AnalysisService
):
    """백그라운드에서 포트폴리오 분석 실행 및 콜백 전송"""
    start_time = time.time()
    
    # 독립적인 세션 생성하여 트랜잭션 충돌 방지
    from app.models.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as analysis_session:
        try:
            logger.info(f"Starting background analysis", job_id=job_id)
            
            result = await analysis_service.run_analysis(
                request=request,
                session=analysis_session  # 새로운 독립 세션 사용
            )
            
            execution_time = time.time() - start_time
            
            # 성공 콜백 전송
            callback_response = AnalysisCallbackResponse(
                job_id=job_id,
                success=True,
                metadata=AnalysisCallbackMetadata(
                    risk_free_rate_used=result.metadata.risk_free_rate_used if result.metadata else None,
                    period=result.metadata.period if result.metadata else None,
                    notes=result.metadata.notes if result.metadata else None,
                    execution_time=execution_time,
                    portfolio_id=request.portfolio_id,
                    timestamp=datetime.utcnow()
                ),
                benchmark=result.benchmark,
                portfolios=result.portfolios,
                stock_details=result.stock_details,
            )
            
            await send_analysis_callback(request.callback_url, callback_response)
            
            logger.info(
                "Background analysis completed successfully",
                job_id=job_id,
                execution_time=f"{execution_time:.3f}s"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(
                "Background analysis failed",
                job_id=job_id,
                error=str(e),
                execution_time=f"{execution_time:.3f}s",
                exc_info=True
            )
            
            # 실패 콜백 전송
            callback_response = AnalysisCallbackResponse(
                job_id=job_id,
                success=False,
                metadata=AnalysisCallbackMetadata(
                    risk_free_rate_used=None,
                    period=None,
                    notes=None,
                    execution_time=execution_time,
                    portfolio_id=request.portfolio_id,
                    timestamp=datetime.utcnow()
                ),
                benchmark=None,
                portfolios=[],
                stock_details=None,
            )
            
            await send_analysis_callback(request.callback_url, callback_response)


async def send_analysis_callback(callback_url: str, response: AnalysisCallbackResponse):
    """콜백 URL로 분석 결과 전송"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 응답 상태에 따라 적절한 헤더와 상태 정보 추가
            headers = {
                "Content-Type": "application/json",
                "X-Analysis-Status": "success" if response.success else "error",
                "X-Analysis-Job-ID": response.job_id
            }
            
            # 에러 응답인 경우 추가 헤더 설정
            if not response.success:
                headers["X-Analysis-Error-Type"] = "ANALYSIS_ERROR"
                
            # 콜백 직전 실제 JSON 페이로드 로깅
            payload = response.model_dump(mode='json')
            logger.info(
                "Prepared analysis callback JSON payload",
                job_id=response.job_id,
                payload=json.dumps(payload, ensure_ascii=False, default=str)
            )

            callback_result = await client.post(
                callback_url,
                json=payload,
                headers=headers
            )
            
            # 성공/실패에 따른 로깅
            expected_status_codes = {
                True: 200,   # 성공: 200 OK
                False: 400   # 에러: 400 Bad Request
            }
            expected_status = expected_status_codes[response.success]
            
            if callback_result.status_code == expected_status:
                if response.success:
                    logger.info(
                        "Analysis callback sent successfully",
                        job_id=response.job_id,
                        callback_url=callback_url,
                        status_code=callback_result.status_code
                    )
                else:
                    logger.warning(
                        "Analysis callback sent with error response",
                        job_id=response.job_id,
                        callback_url=callback_url,
                        status_code=callback_result.status_code
                    )
            else:
                status_msg = "successfully" if response.success else "with error response"
                logger.warning(
                    f"Analysis callback {status_msg} but response status code differs",
                    job_id=response.job_id,
                    callback_url=callback_url,
                    expected_status=expected_status,
                    actual_status=callback_result.status_code,
                    success=response.success
                )
                
    except Exception as e:
        logger.error(
            "Failed to send analysis callback",
            job_id=response.job_id,
            callback_url=callback_url,
            success=response.success,
            error=str(e)
        )
        # 콜백 전송 실패 시 예외를 다시 발생시켜 상위에서 실패로 처리되도록 함
        raise


@router.get(
    "/benchmarks",
    response_model=List[str],
    summary="사용 가능한 벤치마크 목록 조회",
    description="분석에 사용할 수 있는 벤치마크 지수 목록을 반환합니다."
)
async def get_available_benchmarks(
    session: AsyncSession = Depends(get_async_session),
) -> List[str]:
    try:
        repo = BenchmarkRepository(session)
        benchmarks = await repo.get_available_benchmarks()
        return benchmarks
    except Exception as e:
        logger.error("Failed to retrieve available benchmarks", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/risk-free-rates",
    response_model=List[str],
    summary="사용 가능한 무위험수익률 유형 조회",
    description="분석에 사용할 수 있는 무위험수익률 유형 목록을 반환합니다."
)
async def get_available_risk_free_rates(
    session: AsyncSession = Depends(get_async_session),
) -> List[str]:
    try:
        repo = RiskFreeRateRepository(session)
        rate_types = await repo.get_available_rate_types()
        return rate_types
    except Exception as e:
        logger.error("Failed to retrieve available risk-free rate types", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/risk-free-rate/{rate_type}/current",
    response_model=Optional[RiskFreeRateResponse],
    summary="현재 무위험수익률 조회",
    description="지정된 유형의 현재 무위험수익률을 조회합니다."
)
async def get_current_risk_free_rate(
    rate_type: str,
    session: AsyncSession = Depends(get_async_session),
) -> Optional[RiskFreeRateResponse]:
    try:
        repo = RiskFreeRateRepository(session)
        rate_data = await repo.get_latest_risk_free_rate(rate_type)
        return rate_data
    except Exception as e:
        logger.error(f"Failed to retrieve current risk-free rate for {rate_type}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/benchmark/{index_code}/current",
    response_model=Optional[BenchmarkPriceResponse],
    summary="현재 벤치마크 가격 조회",
    description="지정된 지수의 현재 가격 정보를 조회합니다."
)
async def get_current_benchmark_price(
    index_code: str,
    session: AsyncSession = Depends(get_async_session),
) -> Optional[BenchmarkPriceResponse]:
    try:
        repo = BenchmarkRepository(session)
        price_data = await repo.get_latest_benchmark_price(index_code)
        return price_data
    except Exception as e:
        logger.error(f"Failed to retrieve current benchmark price for {index_code}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/data/validate",
    summary="데이터 무결성 검증",
    description="지정된 기간의 벤치마크 및 무위험수익률 데이터 무결성을 검증합니다."
)
async def validate_data_integrity(
    start_date: datetime,
    end_date: datetime,
    session: AsyncSession = Depends(get_async_session),
    data_service: DataCollectionService = Depends(get_data_collection_service),
) -> Dict:
    try:
        validation_result = await data_service.validate_data_integrity(
            session, start_date, end_date
        )
        return validation_result
    except Exception as e:
        logger.error("Failed to validate data integrity", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/data/update-benchmarks",
    summary="벤치마크 데이터 업데이트",
    description="벤치마크 지수 데이터를 최신으로 업데이트합니다."
)
async def update_benchmark_data(
    index_codes: Optional[List[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    session: AsyncSession = Depends(get_async_session),
    data_service: DataCollectionService = Depends(get_data_collection_service),
) -> Dict[str, int]:
    try:
        update_result = await data_service.update_benchmark_data(
            session, index_codes, start_date, end_date
        )
        return update_result
    except Exception as e:
        logger.error("Failed to update benchmark data", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/data/update-risk-free-rates",
    summary="무위험수익률 데이터 업데이트",
    description="무위험수익률 데이터를 최신으로 업데이트합니다."
)
async def update_risk_free_rates(
    rate_types: Optional[List[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    session: AsyncSession = Depends(get_async_session),
    data_service: DataCollectionService = Depends(get_data_collection_service),
) -> Dict[str, int]:
    try:
        update_result = await data_service.update_risk_free_rates(
            session, rate_types, start_date, end_date
        )
        return update_result
    except Exception as e:
        logger.error("Failed to update risk-free rates", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/health",
    summary="분석 서비스 헬스 체크",
    description="분석 서비스의 상태를 확인합니다."
)
async def health_check() -> dict:
    return {"status": "healthy", "service": "analysis"}


