"""포트폴리오 분석 API 라우터 (도메인 분리)"""

from typing import Optional, List, Dict
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_async_session
from app.models.schemas import (
    AnalysisRequest,
    EnhancedAnalysisResponse,
    BenchmarkPriceResponse,
    RiskFreeRateResponse
)
from app.services.analysis_service import AnalysisService
from app.services.data_collection_service import DataCollectionService
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
    "/run",
    response_model=EnhancedAnalysisResponse,
    summary="포트폴리오 분석 실행",
    description="보유 종목 수량 기반으로 MPT/CAPM 분석을 수행하고 결과를 반환합니다. 벤치마크 비교 분석도 포함됩니다."
)
async def run_analysis(
    request: AnalysisRequest,
    session: AsyncSession = Depends(get_async_session),
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> EnhancedAnalysisResponse:
    """포트폴리오 분석 실행
    
    기본 분석과 고급 분석을 모두 포함하여 포트폴리오 성과를 종합적으로 분석합니다.
    
    제공 기능:
    - 포트폴리오 최적화 (최소 변동성, 최대 샤프)
    - 기본 성과 지표 (기대수익률, 변동성, 샤프 비율 등)
    - 고급 리스크 지표 (베타, 알파, 트래킹 에러, 소르티노 비율 등)
    - 벤치마크 비교 분석 (선택적)
    """
    try:
        if not request.holdings:
            raise HTTPException(status_code=400, detail="At least one holding must be specified")

        result = await analysis_service.run_analysis(request, session)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to run analysis", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


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


