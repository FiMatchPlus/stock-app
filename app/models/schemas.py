"""Pydantic 스키마 정의"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, field_validator

# 한국 시간대 설정
KST = timezone(timedelta(hours=9))

def get_kst_now():
    return datetime.now(KST)


class StockCreate(BaseModel):
    """종목 생성 스키마"""
    ticker: str = Field(..., max_length=20, description="단축코드")
    name: str = Field(..., max_length=100, description="한글종목명")
    eng_name: Optional[str] = Field(None, max_length=100, description="영어이름")
    isin: str = Field(..., max_length=20, description="표준코드")
    region: str = Field(default="KR", max_length=20, description="지역")
    currency: str = Field(default="KRW", max_length=3, description="통화")
    major_code: Optional[str] = Field(None, max_length=100, description="지수업종대분류코드명")
    medium_code: Optional[str] = Field(None, max_length=100, description="지수업종중분류코드명")
    minor_code: Optional[str] = Field(None, max_length=100, description="지수업종소분류코드명")
    market: str = Field(..., max_length=20, description="시장구분")
    exchange: Optional[str] = Field(None, max_length=20, description="거래소구분")
    is_active: str = Field(default="Y", max_length=1, description="매매가능여부")
    industry_code: Optional[int] = Field(None, description="표준산업분류코드")
    industry_name: Optional[str] = Field(None, max_length=100, description="표준산업분류코드명")
    type: Optional[str] = Field(None, max_length=50, description="상품종류")

    @field_validator('is_active')
    @classmethod
    def validate_is_active(cls, v):
        if v not in ['Y', 'N']:
            raise ValueError('is_active must be Y or N')
        return v


class StockResponse(BaseModel):
    """종목 응답 스키마"""
    id: int
    ticker: str
    name: str
    eng_name: Optional[str]
    isin: str
    region: str
    currency: str
    major_code: Optional[str]
    medium_code: Optional[str]
    minor_code: Optional[str]
    market: str
    exchange: Optional[str]
    is_active: str
    industry_code: Optional[int]
    industry_name: Optional[str]
    type: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class StockPriceCreate(BaseModel):
    """주가 데이터 생성 스키마"""
    stock_code: str = Field(..., max_length=10, description="종목코드")
    timestamp: datetime = Field(..., description="시간", alias="datetime")
    interval_unit: str = Field(..., max_length=5, description="시간간격")
    open_price: Decimal = Field(..., ge=0, description="시가")
    high_price: Decimal = Field(..., ge=0, description="고가")
    low_price: Decimal = Field(..., ge=0, description="저가")
    close_price: Decimal = Field(..., ge=0, description="종가")
    volume: int = Field(..., ge=0, description="거래량")
    change_amount: Decimal = Field(..., description="전일대비")

    @field_validator('interval_unit')
    @classmethod
    def validate_interval_unit(cls, v):
        if v not in ['1m', '1d', '1W', '1Y']:
            raise ValueError('interval_unit must be one of: 1m, 1d, 1W, 1Y')
        return v


class StockPriceResponse(BaseModel):
    """주가 데이터 응답 스키마"""
    id: int
    stock_code: str
    timestamp: datetime = Field(..., alias="datetime")
    interval_unit: str
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: int
    change_amount: Decimal
    change_rate: Decimal

    class Config:
        from_attributes = True


class StockPriceCollectionRequest(BaseModel):
    """주가 데이터 수집 요청 스키마"""
    symbols: List[str] = Field(..., max_items=100, description="종목코드 목록 (최대 100개)")
    interval: str = Field(default="1d", description="시간간격")
    start_date: Optional[datetime] = Field(None, description="시작일")
    end_date: Optional[datetime] = Field(None, description="종료일")

    @field_validator('interval')
    @classmethod
    def validate_interval(cls, v):
        if v not in ['1m', '1d', '1W', '1Y']:
            raise ValueError('interval must be one of: 1m, 1d, 1W, 1Y')
        return v


class StockPriceQueryRequest(BaseModel):
    """주가 데이터 조회 요청 스키마"""
    symbol: str = Field(..., description="종목코드")
    interval: Optional[str] = Field(None, description="시간간격")
    start_date: Optional[datetime] = Field(None, description="시작일")
    end_date: Optional[datetime] = Field(None, description="종료일")
    limit: Optional[int] = Field(100, ge=1, le=1000, description="조회 개수")
    offset: Optional[int] = Field(0, ge=0, description="오프셋")


class AggregateResult(BaseModel):
    """집계 결과 스키마"""
    symbol: str
    interval: str
    count: int
    avg_price: Decimal
    min_price: Decimal
    max_price: Decimal
    total_volume: int
    volatility: Decimal
    correlation: Optional[Dict[str, Decimal]] = None
    calculated_at: datetime


# WebSocket 스키마 제거됨


class ErrorResponse(BaseModel):
    """에러 응답 스키마"""
    error: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 에러 정보")
    timestamp: datetime = Field(default_factory=get_kst_now, description="에러 발생 시간")


class MissingStockData(BaseModel):
    """누락된 주가 데이터 정보"""
    stock_code: str = Field(..., description="종목코드")
    start_date: str = Field(..., description="요청된 시작일")
    end_date: str = Field(..., description="요청된 종료일")
    available_date_range: Optional[str] = Field(None, description="사용 가능한 날짜 범위")


class BacktestDataError(BaseModel):
    """백테스트 데이터 오류 정보"""
    error_type: str = Field(..., description="오류 유형")
    message: str = Field(..., description="오류 메시지")
    missing_data: List[MissingStockData] = Field(..., description="누락된 데이터 목록")
    requested_period: str = Field(..., description="요청된 기간")
    total_stocks: int = Field(..., description="총 요청 종목 수")
    missing_stocks_count: int = Field(..., description="데이터 누락 종목 수")
    timestamp: datetime = Field(default_factory=get_kst_now, description="오류 발생 시간")


class BacktestDataError(BaseModel):
    """백테스트 데이터 오류 정보"""
    error_type: str = Field(..., description="오류 유형")
    message: str = Field(..., description="오류 메시지")
    missing_data: List[MissingStockData] = Field(..., description="누락된 데이터 목록")
    requested_period: str = Field(..., description="요청된 기간")
    total_stocks: int = Field(..., description="총 요청 종목 수")
    missing_stocks_count: int = Field(..., description="데이터 누락 종목 수")
    timestamp: datetime = Field(default_factory=get_kst_now, description="오류 발생 시간")


# 백테스트 관련 스키마
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
        valid_categories = ['BETA', 'MDD', 'VAR', 'ONEPROFIT']
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


class BacktestJobResponse(BaseModel):
    """비동기 백테스트 작업 시작 응답"""
    job_id: str = Field(..., description="작업 ID")
    status: str = Field(..., description="작업 상태 (started)")
    message: str = Field(..., description="상태 메시지")


class BacktestCallbackResponse(BaseModel):
    """백테스트 완료 콜백 응답 스키마"""
    job_id: str = Field(..., description="작업 ID")
    # 성공 시: BacktestResponse와 동일한 필드들
    success: Optional[bool] = Field(None, description="성공 여부")
    portfolio_snapshot: Optional['PortfolioSnapshotResponse'] = Field(None, description="포트폴리오 스냅샷")
    metrics: Optional['BacktestMetrics'] = Field(None, description="성과 지표")
    benchmark_metrics: Optional['BenchmarkMetrics'] = Field(None, description="벤치마크 성과 지표")
    result_summary: Optional[List['ResultSummary']] = Field(None, description="결과 요약 데이터")
    execution_logs: Optional[List['ExecutionLog']] = Field(None, description="손절/익절 실행 로그")
    result_status: Optional[str] = Field(None, description="결과 상태")
    benchmark_info: Optional[Dict[str, Any]] = Field(None, description="사용된 벤치마크 정보")
    risk_free_rate_info: Optional[Dict[str, Any]] = Field(None, description="사용된 무위험 수익률 정보")
    # 실패 시: 에러 정보 포함
    error: Optional['BacktestDataError'] = Field(None, description="오류 상세 정보")
    # 공통 필드
    execution_time: float = Field(..., description="실행 시간 (초)")
    backtest_id: Optional[int] = Field(None, description="클라이언트에서 제공한 백테스트 ID")
    timestamp: datetime = Field(default_factory=get_kst_now, description="완료 시각")


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


# ------------------------------
# Analysis (MPT/CAPM) Schemas
# ------------------------------

class AnalysisHoldingInput(BaseModel):
    """분석 입력 보유 종목 스키마"""
    stock_code: str = Field(..., description="종목코드")
    quantity: int = Field(..., ge=1, description="보유 수량 (주)")


class AnalysisRequest(BaseModel):
    """포트폴리오 분석 요청 스키마 (MPT/CAPM)"""
    holdings: List[AnalysisHoldingInput] = Field(..., min_items=1, description="보유 종목 목록")
    lookback_years: int = Field(3, ge=1, le=10, description="과거 데이터 조회 연수 (3~5 추천)")
    benchmark: Optional[str] = Field(None, description="벤치마크 지수 코드 (예: KOSPI, KOSDAQ). 미제공 시 내부 추정")
    risk_free_rate: Optional[float] = Field(None, description="연간 무위험 수익률 (소수). 미제공 시 0 가정")


class PortfolioWeights(BaseModel):
    """종목별 비중"""
    weights: Dict[str, float] = Field(..., description="종목코드별 비중 (합계 1.0)")


# ------------------------------
# Benchmark & Risk-Free Rate Schemas
# ------------------------------

class BenchmarkPriceResponse(BaseModel):
    """벤치마크 가격 응답 스키마"""
    id: int
    index_code: str
    datetime: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    change_amount: Decimal
    change_rate: Decimal
    volume: int
    trading_value: Decimal
    market_cap: Optional[Decimal]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class RiskFreeRateResponse(BaseModel):
    """무위험수익률 응답 스키마"""
    id: int
    rate_type: str
    datetime: datetime
    rate: Decimal
    daily_rate: Decimal
    source: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ------------------------------
# Enhanced Analysis Schemas
# ------------------------------

class EnhancedAnalysisMetrics(BaseModel):
    """향상된 포트폴리오 성과 지표"""
    # 기본 지표
    expected_return: float = Field(..., description="기대수익률 (연환산)")
    std_deviation: float = Field(..., description="표준편차 (연환산)")
    
    # 벤치마크 대비 지표
    beta: float = Field(..., description="베타 (벤치마크 대비)")
    alpha: float = Field(..., description="알파 (CAPM 기준)")
    jensen_alpha: float = Field(..., description="젠센 알파")
    tracking_error: float = Field(..., description="트래킹 에러")
    
    # 위험조정 수익률 지표
    sharpe_ratio: float = Field(..., description="샤프 비율")
    treynor_ratio: float = Field(..., description="트레이너 비율")
    sortino_ratio: float = Field(..., description="소르티노 비율")
    calmar_ratio: float = Field(..., description="칼마 비율")
    information_ratio: float = Field(..., description="정보비율")
    
    # 리스크 지표
    max_drawdown: float = Field(..., description="최대 낙폭")
    downside_deviation: float = Field(..., description="하방편차")
    upside_beta: float = Field(..., description="상승 베타")
    downside_beta: float = Field(..., description="하락 베타")
    
    # 벤치마크 상관관계
    correlation_with_benchmark: float = Field(..., description="벤치마크와의 상관관계")


class BenchmarkComparison(BaseModel):
    """벤치마크 비교 결과"""
    benchmark_code: str = Field(..., description="벤치마크 지수 코드")
    benchmark_return: float = Field(..., description="벤치마크 수익률 (연환산)")
    benchmark_volatility: float = Field(..., description="벤치마크 변동성")
    
    # 초과 성과
    excess_return: float = Field(..., description="초과 수익률")
    relative_volatility: float = Field(..., description="상대 변동성")
    
    # 성과 기여도 분석
    security_selection: float = Field(..., description="종목선택 효과")
    timing_effect: float = Field(..., description="타이밍 효과")


class EnhancedAnalysisResponse(BaseModel):
    """향상된 포트폴리오 분석 응답"""
    success: bool = Field(True, description="성공 여부")
    min_variance: PortfolioWeights = Field(..., description="최소분산 포트폴리오 비중")
    max_sharpe: PortfolioWeights = Field(..., description="최대 샤프 포트폴리오 비중")
    metrics: Dict[str, EnhancedAnalysisMetrics] = Field(..., description="향상된 성과 지표")
    benchmark_comparison: Optional[BenchmarkComparison] = Field(None, description="벤치마크 비교 결과")
    risk_free_rate_used: float = Field(..., description="사용된 무위험수익률")
    analysis_period: Dict[str, datetime] = Field(..., description="분석 기간")
    notes: Optional[str] = Field(None, description="참고 사항")