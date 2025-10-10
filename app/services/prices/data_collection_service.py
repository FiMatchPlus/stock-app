"""벤치마크 및 무위험수익률 데이터 수집 서비스"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import asyncio
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.stock import BenchmarkPrice, RiskFreeRate
from app.repositories.benchmark_repository import BenchmarkRepository
from app.repositories.risk_free_rate_repository import RiskFreeRateRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DataCollectionService:
    """벤치마크 및 무위험수익률 데이터 수집/업데이트 서비스"""
    
    def __init__(self):
        pass
    
    async def update_benchmark_data(
        self,
        session: AsyncSession,
        index_codes: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, int]:
        """벤치마크 데이터 업데이트"""
        if index_codes is None:
            index_codes = ["KOSPI", "KOSDAQ", "KRX100"]
        
        if end_date is None:
            end_date = datetime.utcnow()
        
        if start_date is None:
            start_date = end_date - timedelta(days=30)  # 기본 30일
        
        logger.info(f"Starting benchmark data update for {len(index_codes)} indices")
        
        # TODO: 실제 외부 API 연동 구현 필요
        # 현재는 구조만 제공
        
        update_counts = {}
        for index_code in index_codes:
            # 예시: 외부 API에서 데이터 수집 후 저장
            count = await self._collect_and_save_benchmark_data(
                session, index_code, start_date, end_date
            )
            update_counts[index_code] = count
        
        total_updated = sum(update_counts.values())
        logger.info(f"Benchmark data update completed. Total records updated: {total_updated}")
        
        return update_counts
    
    async def _collect_and_save_benchmark_data(
        self,
        session: AsyncSession,
        index_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """특정 지수의 벤치마크 데이터 수집 및 저장"""
        # TODO: 실제 외부 API 연동 구현
        # 예: 한국투자증권 API, Yahoo Finance API 등
        
        logger.info(f"Collecting benchmark data for {index_code}")
        
        # 임시 구현: 실제로는 외부 API 호출
        return 0
    
    async def update_risk_free_rates(
        self,
        session: AsyncSession,
        rate_types: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, int]:
        """무위험수익률 데이터 업데이트"""
        if rate_types is None:
            rate_types = ["CD91", "Treasury3Y", "BOK_BASE"]
        
        if end_date is None:
            end_date = datetime.utcnow()
        
        if start_date is None:
            start_date = end_date - timedelta(days=30)  # 기본 30일
        
        logger.info(f"Starting risk-free rate update for {len(rate_types)} rate types")
        
        update_counts = {}
        for rate_type in rate_types:
            count = await self._collect_and_save_risk_free_rate_data(
                session, rate_type, start_date, end_date
            )
            update_counts[rate_type] = count
        
        total_updated = sum(update_counts.values())
        logger.info(f"Risk-free rate update completed. Total records updated: {total_updated}")
        
        return update_counts
    
    async def _collect_and_save_risk_free_rate_data(
        self,
        session: AsyncSession,
        rate_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """특정 유형의 무위험수익률 데이터 수집 및 저장"""
        # TODO: 실제 외부 API 연동 구현
        # 예: 한국은행 ECOS API, 금융투자협회 API 등
        
        logger.info(f"Collecting risk-free rate data for {rate_type}")
        
        # 임시 구현: 실제로는 외부 API 호출
        return 0
    
    async def validate_data_integrity(
        self,
        session: AsyncSession,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, any]:
        """데이터 무결성 검증"""
        
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
                rate_type, start_date, end_date
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


