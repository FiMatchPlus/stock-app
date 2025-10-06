"""포트폴리오 분석 서비스 (MPT/CAPM) - 이동 윈도우 버전

개요:
- 3년 윈도우 크기로 1개월 간격으로 이동하며 포트폴리오 최적화 수행
- 최소 변동성 포트폴리오와 최대 샤프 포트폴리오의 비중을 각 시점별로 계산
- 백테스팅을 통해 전체 기간에 대한 성능 지표를 계산
- 최종 응답은 최근 시점의 비중과 백테스팅 기반 평균 성능 지표를 포함
- 벤치마크 비교와 고급 리스크 지표를 포함한 포트폴리오 분석을 제공합니다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import (
    AnalysisRequest,
    PortfolioAnalysisResponse,
    AnalysisJobResponse,
    AnalysisCallbackResponse,
    PortfolioWeights,
    AnalysisMetrics,
    BenchmarkComparison,
)
from app.models.stock import StockPrice
from app.repositories.benchmark_repository import BenchmarkRepository
from app.repositories.risk_free_rate_repository import RiskFreeRateRepository
from app.utils.logger import get_logger
from app.services.moving_window_analysis_service import MovingWindowAnalysisService


logger = get_logger(__name__)


class AnalysisService:
    """MPT/CAPM 기반 포트폴리오 분석 서비스 - 이동 윈도우 방식"""

    def __init__(self):
        self.moving_window_service = MovingWindowAnalysisService()

    async def run_analysis(
        self,
        request: AnalysisRequest,
        session: AsyncSession,
    ) -> PortfolioAnalysisResponse:
        """이동 윈도우 기반 포트폴리오 분석 실행

        3년 윈도우 크기로 1개월 간격으로 이동하며 최적화를 수행하고,
        백테스팅을 통해 검증된 성능 지표를 제공합니다.
        """
        
        # 이동 윈도우 서비스로 분석 위임
        result = await self.moving_window_service.run_analysis(request, session)
        return result


