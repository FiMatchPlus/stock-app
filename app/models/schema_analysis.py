"""Domain: Analysis related schemas."""

from __future__ import annotations

from typing import Optional, List, Dict
from datetime import datetime
from pydantic import BaseModel, Field
from app.models.schema_common import ErrorResponse, get_kst_now


class AnalysisHoldingInput(BaseModel):
    """분석 입력 보유 종목 스키마"""
    code: str = Field(..., description="종목코드")
    quantity: int = Field(..., ge=1, description="보유 수량 (주)")


class AnalysisRequest(BaseModel):
    """포트폴리오 분석 요청 스키마 (MPT/CAPM)"""
    holdings: List[AnalysisHoldingInput] = Field(..., min_items=1, description="보유 종목 목록")
    callback_url: str = Field(..., description="결과를 받을 콜백 URL (비동기 처리 시 필수)")
    portfolio_id: int = Field(..., description="포트폴리오 ID (콜백 시 그대로 반환)")
    lookback_years: int = Field(3, ge=1, le=10, description="과거 데이터 조회 연수 (3~5 추천)")
    benchmark: Optional[str] = Field(None, description="벤치마크 지수 코드 (예: KOSPI, KOSDAQ). 미제공 시 내부 추정")
    risk_free_rate: Optional[float] = Field(None, description="연간 무위험 수익률 (소수). 내부 계산")


class BetaAnalysis(BaseModel):
    """베타 분석 정보 (간소화)"""
    beta: float = Field(..., description="시장 베타 (벤치마크 대비 민감도)")
    r_square: float = Field(..., description="결정계수 (모델 설명력, 0~1)")
    alpha: float = Field(..., description="알파 (초과 수익률)")


class AnalysisMetrics(BaseModel):
    """포트폴리오 성과 지표"""
    expected_return: float = Field(..., description="기대수익률 (연환산)")
    std_deviation: float = Field(..., description="표준편차 (연환산)")
    jensen_alpha: float = Field(..., description="젠센 알파")
    tracking_error: float = Field(..., description="트래킹 에러")
    sharpe_ratio: float = Field(..., description="샤프 비율")
    treynor_ratio: float = Field(..., description="트레이너 비율")
    sortino_ratio: float = Field(..., description="소르티노 비율")
    calmar_ratio: float = Field(..., description="칼마 비율")
    information_ratio: float = Field(..., description="정보비율")
    max_drawdown: float = Field(..., description="최대 낙폭")
    downside_deviation: float = Field(..., description="하방편차")
    upside_beta: float = Field(..., description="상승 베타")
    downside_beta: float = Field(..., description="하락 베타")
    var_value: float = Field(..., description="VaR 95% (Value at Risk)")
    cvar_value: float = Field(..., description="CVaR 95% (Conditional Value at Risk)")
    correlation_with_benchmark: float = Field(..., description="벤치마크와의 상관관계")


class StockDetails(BaseModel):
    """종목별 상세 정보 (포트폴리오 분석 결과)"""
    expected_return: float = Field(..., description="기대수익률")
    volatility: float = Field(..., description="변동성")
    correlation_to_portfolio: float = Field(..., description="포트폴리오와의 상관관계")
    beta_analysis: Optional[BetaAnalysis] = Field(None, description="베타 분석 정보")


class BenchmarkComparison(BaseModel):
    """벤치마크 비교 결과"""
    benchmark_code: str = Field(..., description="벤치마크 지수 코드")
    benchmark_return: float = Field(..., description="벤치마크 수익률 (연환산)")
    benchmark_volatility: float = Field(..., description="벤치마크 변동성")
    excess_return: float = Field(..., description="초과 수익률")
    relative_volatility: float = Field(..., description="상대 변동성")
    security_selection: float = Field(..., description="종목선택 효과")
    timing_effect: float = Field(..., description="타이밍 효과")


class PortfolioData(BaseModel):
    """포트폴리오 데이터"""
    type: str = Field(..., description="포트폴리오 타입 (user, min_variance, max_sharpe)")
    weights: Dict[str, float] = Field(..., description="종목코드별 비중 (합계 1.0)")
    beta_analysis: Optional[BetaAnalysis] = Field(None, description="포트폴리오 베타 분석 정보")
    metrics: Optional[AnalysisMetrics] = Field(None, description="성과 지표")
    benchmark_comparison: Optional[BenchmarkComparison] = Field(None, description="벤치마크 비교 결과")


class AnalysisMetadata(BaseModel):
    """분석 메타데이터"""
    risk_free_rate_used: float = Field(..., description="사용된 무위험수익률")
    period: Dict[str, datetime] = Field(..., description="분석 기간")
    notes: Optional[str] = Field(None, description="참고 사항")
    execution_time: Optional[float] = Field(None, description="실행 시간 (초)")
    portfolio_id: Optional[int] = Field(None, description="클라이언트에서 제공한 포트폴리오 ID")
    timestamp: Optional[datetime] = Field(default_factory=get_kst_now, description="응답 생성 시각")


class BenchmarkInfo(BaseModel):
    """벤치마크 정보"""
    code: str = Field(..., description="벤치마크 지수 코드")
    benchmark_return: float = Field(..., alias="return", description="벤치마크 수익률 (연환산)")
    volatility: float = Field(..., description="벤치마크 변동성")


class PortfolioAnalysisResponse(BaseModel):
    """새로운 포트폴리오 분석 응답"""
    success: bool = Field(True, description="성공 여부")
    metadata: AnalysisMetadata = Field(..., description="분석 메타데이터")
    benchmark: Optional[BenchmarkInfo] = Field(None, description="벤치마크 정보")
    portfolios: List[PortfolioData] = Field(..., description="포트폴리오 데이터 목록")
    stock_details: Optional[Dict[str, StockDetails]] = Field(None, description="종목별 상세 정보")


class AnalysisJobResponse(BaseModel):
    """비동기 포트폴리오 분석 작업 시작 응답"""
    job_id: str = Field(..., description="작업 ID")
    status: str = Field(..., description="작업 상태 (started)")
    message: str = Field(..., description="상태 메시지")


class AnalysisCallbackMetadata(BaseModel):
    """포트폴리오 분석 완료 콜백 메타데이터 스키마"""
    risk_free_rate_used: Optional[float] = Field(None, description="사용된 무위험수익률")
    period: Optional[Dict[str, datetime]] = Field(None, description="분석 기간")
    notes: Optional[str] = Field(None, description="참고 사항")
    execution_time: float = Field(..., description="실행 시간 (초)")
    portfolio_id: Optional[int] = Field(None, description="클라이언트에서 제공한 포트폴리오 ID")
    timestamp: datetime = Field(default_factory=get_kst_now, description="완료 시각")


class AnalysisCallbackResponse(BaseModel):
    """포트폴리오 분석 완료 콜백 응답 스키마"""
    job_id: str = Field(..., description="작업 ID")
    success: Optional[bool] = Field(None, description="성공 여부")
    min_variance: Optional[Dict[str, float]] = Field(None, description="최소분산 포트폴리오 비중")
    max_sharpe: Optional[Dict[str, float]] = Field(None, description="최대 샤프 포트폴리오 비중")
    metrics: Optional[Dict[str, AnalysisMetrics]] = Field(None, description="향상된 성과 지표")
    benchmark_comparison: Optional[BenchmarkComparison] = Field(None, description="벤치마크 비교 결과")
    benchmark_info: Optional[BenchmarkInfo] = Field(None, description="벤치마크 정보")
    metadata: AnalysisCallbackMetadata = Field(..., description="분석 메타데이터")
    error: Optional[ErrorResponse] = Field(None, description="오류 상세 정보")


 

__all__ = [
    "AnalysisHoldingInput",
    "AnalysisRequest",
    "BetaAnalysis",
    "AnalysisMetrics",
    "StockDetails",
    "BenchmarkComparison",
    "PortfolioData",
    "AnalysisMetadata",
    "BenchmarkInfo",
    "PortfolioAnalysisResponse",
    "AnalysisJobResponse",
    "AnalysisCallbackMetadata",
    "AnalysisCallbackResponse",
]


