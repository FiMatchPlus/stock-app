"""Backtest related schemas."""

from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from app.models.schema_common import BacktestDataError, get_kst_now


class Holding(BaseModel):
    """보유 종목 스키마"""
    code: str = Field(..., description="종목코드")
    quantity: int = Field(..., ge=0, description="보유 수량 (주)")
    avg_price: Optional[Decimal] = Field(None, ge=0, description="평균 매수가 (선택사항)")
    current_value: Optional[Decimal] = Field(None, ge=0, description="현재 평가액 (선택사항)")


class TradingRule(BaseModel):
    """거래 규칙 스키마"""
    category: str = Field(..., description="규칙 카테고리")
    value: float = Field(..., description="임계값")
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        valid_categories = ['BETA', 'MDD', 'VAR', 'ONEPROFIT', 'LOSS_LIMIT']
        if v not in valid_categories:
            raise ValueError(f'category must be one of: {valid_categories}')
        return v


class TradingRules(BaseModel):
    """거래 규칙 스키마"""
    stopLoss: Optional[List[TradingRule]] = Field(None, description="손절 규칙 목록")
    takeProfit: Optional[List[TradingRule]] = Field(None, description="익절 규칙 목록")


class BacktestRequest(BaseModel):
    """백테스트 요청 스키마"""
    start: datetime = Field(..., description="시작일")
    end: datetime = Field(..., description="종료일")
    holdings: List[Holding] = Field(..., min_items=1, description="보유 종목 목록")
    rebalance_frequency: Optional[str] = Field("daily", description="리밸런싱 주기")
    callback_url: Optional[str] = Field(None, description="결과를 받을 콜백 URL (비동기 처리 시 필수)")
    rules: Optional[TradingRules] = Field(None, description="손절/익절 규칙")
    risk_free_rate: Optional[float] = Field(None, description="무위험 수익률 (연율, 미제공시 자동 결정)")
    benchmark_code: Optional[str] = Field("KOSPI", description="벤치마크 지수 코드 (미제공시 KOSPI 기본값)")
    backtest_id: Optional[int] = Field(None, description="클라이언트에서 제공하는 백테스트 ID (콜백 시 그대로 반환)")
    
    @field_validator('holdings')
    @classmethod
    def validate_holdings_quantities(cls, v):
        """보유 수량 검증"""
        for holding in v:
            if holding.quantity <= 0:
                raise ValueError(f'Quantity must be positive for {holding.code}')
        return v

    @field_validator('rebalance_frequency')
    @classmethod
    def validate_rebalance_frequency(cls, v):
        if v not in ['daily', 'weekly', 'monthly']:
            raise ValueError('rebalance_frequency must be one of: daily, weekly, monthly')
        return v


class BacktestMetrics(BaseModel):
    """백테스트 성과 지표 스키마"""
    total_return: Decimal = Field(..., description="총 수익률")
    annualized_return: Decimal = Field(..., description="연환산 수익률")
    volatility: Decimal = Field(..., description="변동성")
    sharpe_ratio: Decimal = Field(..., description="샤프 비율")
    max_drawdown: Decimal = Field(..., description="최대 낙폭")
    var_95: Decimal = Field(..., description="VaR 95%")
    var_99: Decimal = Field(..., description="VaR 99%")
    cvar_95: Decimal = Field(..., description="CVaR 95%")
    cvar_99: Decimal = Field(..., description="CVaR 99%")
    win_rate: Decimal = Field(..., description="승률")
    profit_loss_ratio: Decimal = Field(..., description="손익비")


class BenchmarkMetrics(BaseModel):
    """벤치마크 성과 지표 스키마"""
    benchmark_total_return: float = Field(..., description="벤치마크 총 수익률")
    benchmark_volatility: float = Field(..., description="벤치마크 수익률 변동성")
    benchmark_max_price: float = Field(..., description="벤치마크 최고가")
    benchmark_min_price: float = Field(..., description="벤치마크 최저가")
    alpha: float = Field(..., description="벤치마크 대비 포트폴리오 초과수익률")
    benchmark_daily_average: float = Field(..., description="벤치마크 일일 평균 수익률")


class HoldingSnapshotResponse(BaseModel):
    """보유 종목 스냅샷 응답 스키마"""
    id: int
    stock_id: str
    quantity: int

    class Config:
        from_attributes = True


class PortfolioSnapshotResponse(BaseModel):
    """포트폴리오 스냅샷 응답 스키마"""
    id: int
    portfolio_id: int
    base_value: Decimal
    current_value: Decimal
    start_at: datetime
    end_at: datetime
    created_at: datetime
    execution_time: float
    holdings: List[HoldingSnapshotResponse] = []

    class Config:
        from_attributes = True


class StockDailyData(BaseModel):
    """종목별 일별 데이터 스키마"""
    stock_code: str = Field(..., description="종목코드")
    date: str = Field(..., description="날짜 (ISO 형식)")
    close_price: float = Field(..., description="종가")
    daily_return: float = Field(..., description="일별 수익률")
    portfolio_weight: float = Field(..., description="포트폴리오 내 비중")
    portfolio_contribution: float = Field(..., description="포트폴리오 수익률 기여도")
    quantity: int = Field(..., description="보유 수량 (주)")


class ResultSummary(BaseModel):
    """결과 요약 데이터 스키마 (종목별 일별 데이터)"""
    date: str = Field(..., description="날짜 (ISO 형식)")
    stocks: List[StockDailyData] = Field(..., description="종목별 일별 데이터")


class ExecutionLog(BaseModel):
    """손절/익절 실행 로그"""
    date: str = Field(..., description="실행 날짜")
    action: str = Field(..., description="실행 액션: STOP_LOSS, TAKE_PROFIT")
    category: str = Field(..., description="규칙 카테고리")
    value: float = Field(..., description="실제 값")
    threshold: float = Field(..., description="임계값")
    reason: str = Field(..., description="실행 사유")
    portfolio_value: float = Field(..., description="포트폴리오 가치")


class BacktestResponse(BaseModel):
    """백테스트 응답 스키마"""
    success: bool = Field(True, description="성공 여부")
    portfolio_snapshot: PortfolioSnapshotResponse
    metrics: Optional[BacktestMetrics] = None
    benchmark_metrics: Optional[BenchmarkMetrics] = None
    result_summary: List[ResultSummary] = Field(..., description="결과 요약 데이터")
    execution_time: float = Field(..., description="실행 시간 (초)")
    execution_logs: List[ExecutionLog] = Field(default=[], description="손절/익절 실행 로그")
    result_status: str = Field("COMPLETED", description="결과 상태: COMPLETED, LIQUIDATED")
    benchmark_info: Optional[Dict[str, Any]] = Field(None, description="사용된 벤치마크 정보")
    risk_free_rate_info: Optional[Dict[str, Any]] = Field(None, description="사용된 무위험 수익률 정보")
    backtest_id: Optional[int] = Field(None, description="클라이언트에서 제공한 백테스트 ID")
    timestamp: datetime = Field(default_factory=get_kst_now, description="응답 생성 시각")


class BacktestJobResponse(BaseModel):
    """비동기 백테스트 작업 시작 응답"""
    job_id: str = Field(..., description="작업 ID")
    status: str = Field(..., description="작업 상태 (started)")
    message: str = Field(..., description="상태 메시지")


class BacktestCallbackResponse(BaseModel):
    """백테스트 완료 콜백 응답 스키마"""
    job_id: str = Field(..., description="작업 ID")
    success: Optional[bool] = Field(None, description="성공 여부")
    portfolio_snapshot: Optional[PortfolioSnapshotResponse] = Field(None, description="포트폴리오 스냅샷")
    metrics: Optional[BacktestMetrics] = Field(None, description="성과 지표")
    benchmark_metrics: Optional[BenchmarkMetrics] = Field(None, description="벤치마크 성과 지표")
    result_summary: Optional[List[ResultSummary]] = Field(None, description="결과 요약 데이터")
    execution_logs: Optional[List[ExecutionLog]] = Field(None, description="손절/익절 실행 로그")
    result_status: Optional[str] = Field(None, description="결과 상태")
    benchmark_info: Optional[Dict[str, Any]] = Field(None, description="사용된 벤치마크 정보")
    risk_free_rate_info: Optional[Dict[str, Any]] = Field(None, description="사용된 무위험 수익률 정보")
    error: Optional[BacktestDataError] = Field(None, description="오류 상세 정보")
    execution_time: float = Field(..., description="실행 시간 (초)")
    backtest_id: Optional[int] = Field(None, description="클라이언트에서 제공한 백테스트 ID")
    timestamp: datetime = Field(default_factory=get_kst_now, description="완료 시각")


__all__ = [
    "Holding",
    "TradingRule",
    "TradingRules",
    "BacktestRequest",
    "BacktestMetrics",
    "BenchmarkMetrics",
    "HoldingSnapshotResponse",
    "PortfolioSnapshotResponse",
    "StockDailyData",
    "ResultSummary",
    "ExecutionLog",
    "BacktestResponse",
    "BacktestJobResponse",
    "BacktestCallbackResponse",
]
