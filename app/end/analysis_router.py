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
from app.repositories.benchmark_repository import BenchmarkRepository
from app.repositories.risk_free_rate_repository import RiskFreeRateRepository
from app.utils.logger import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix="/analysis", tags=["analysis"])


async def get_analysis_service() -> AnalysisService:
    return AnalysisService()




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
            "비동기 분석 요청 수신",
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
            "비동기 분석 시작 실패",
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
            logger.info(f"백그라운드 분석 시작", job_id=job_id)
            
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
                "백그라운드 분석 완료",
                job_id=job_id,
                execution_time=f"{execution_time:.3f}s"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(
                "백그라운드 분석 실패",
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
                "분석 콜백 JSON 페이로드 준비 완료",
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
                        "분석 콜백 전송 성공",
                        job_id=response.job_id,
                        callback_url=callback_url,
                        status_code=callback_result.status_code
                    )
                else:
                    logger.warning(
                        "분석 콜백 에러 응답 전송됨",
                        job_id=response.job_id,
                        callback_url=callback_url,
                        status_code=callback_result.status_code
                    )
            else:
                status_msg = "성공적으로 전송됨" if response.success else "에러 응답과 함께 전송됨"
                logger.warning(
                    f"분석 콜백 {status_msg}, 그러나 응답 상태 코드가 다름",
                    job_id=response.job_id,
                    callback_url=callback_url,
                    expected_status=expected_status,
                    actual_status=callback_result.status_code,
                    success=response.success
                )
                
    except Exception as e:
        logger.error(
            "분석 콜백 전송 실패",
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
        logger.error("사용 가능한 벤치마크 조회 실패", error=str(e))
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
        logger.error("사용 가능한 무위험수익률 유형 조회 실패", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/data/validate",
    summary="데이터 무결성 검증",
    description="지정된 기간의 벤치마크 및 무위험수익률 데이터 무결성을 검증합니다."
)
async def validate_data_integrity(
    start_date: datetime,
    end_date: datetime,
    session: AsyncSession = Depends(get_async_session),
) -> Dict:
    """벤치마크 및 무위험수익률 데이터 무결성 검증 (조회 전용)"""
    try:
        benchmark_repo = BenchmarkRepository(session)
        risk_free_repo = RiskFreeRateRepository(session)
        
        # 사용 가능한 벤치마크와 금리 유형 조회
        available_benchmarks = await benchmark_repo.get_available_benchmarks()
        available_rates = await risk_free_repo.get_available_rate_types()
        
        # 각 벤치마크별 데이터 coverage 확인
        benchmark_coverage = {}
        for benchmark in available_benchmarks:
            benchmark_data = await benchmark_repo.get_benchmark_prices(
                [benchmark], start_date, end_date
            )
            coverage_ratio = len(benchmark_data) / max((end_date - start_date).days, 1)
            benchmark_coverage[benchmark] = {
                'record_count': len(benchmark_data),
                'coverage_ratio': coverage_ratio,
                'complete': coverage_ratio > 0.8  # 80% 이상이면 완전한 것으로 간주
            }
        
        # 각 금리별 데이터 coverage 확인
        rate_coverage = {}
        for rate_type in available_rates:
            rate_data = await risk_free_repo.get_risk_free_rate_series(
                start_date, end_date, rate_type
            )
            coverage_ratio = len(rate_data) / max((end_date - start_date).days, 1)
            rate_coverage[rate_type] = {
                'record_count': len(rate_data),
                'coverage_ratio': coverage_ratio,
                'complete': coverage_ratio > 0.8
            }
        
        return {
            'validation_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': (end_date - start_date).days
            },
            'benchmark_coverage': benchmark_coverage,
            'risk_free_rate_coverage': rate_coverage,
            'overall_status': 'healthy' if all(
                cov['complete'] for cov in {**benchmark_coverage, **rate_coverage}.values()
            ) else 'incomplete'
        }
    except Exception as e:
        logger.error("데이터 무결성 검증 실패", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/health",
    summary="분석 서비스 헬스 체크",
    description="분석 서비스의 상태를 확인합니다."
)
async def health_check() -> dict:
    return {"status": "healthy", "service": "analysis"}


