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


# 백테스트 관련 스키마
class Holding(BaseModel):
    """보유 종목 스키마"""
    code: str = Field(..., description="종목코드")
    quantity: int = Field(..., ge=0, description="보유 수량 (주)")
    avg_price: Optional[Decimal] = Field(None, ge=0, description="평균 매수가 (선택사항)")
    current_value: Optional[Decimal] = Field(None, ge=0, description="현재 평가액 (선택사항)")


class BacktestRequest(BaseModel):
    """백테스트 요청 스키마"""
    start: datetime = Field(..., description="시작일")
    end: datetime = Field(..., description="종료일")
    holdings: List[Holding] = Field(..., min_items=1, description="보유 종목 목록")
    rebalance_frequency: Optional[str] = Field("daily", description="리밸런싱 주기")

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


class StockDailyData(BaseModel):
    """종목별 일별 데이터 스키마"""
    stock_code: str = Field(..., description="종목코드")
    date: str = Field(..., description="날짜 (ISO 형식)")
    close_price: float = Field(..., description="종가")
    daily_return: float = Field(..., description="일별 수익률")
    portfolio_weight: float = Field(..., description="포트폴리오 내 비중")
    portfolio_contribution: float = Field(..., description="포트폴리오 수익률 기여도")
    value: float = Field(..., description="보유 가치")


class ResultSummary(BaseModel):
    """결과 요약 데이터 스키마 (종목별 일별 데이터)"""
    date: str = Field(..., description="날짜 (ISO 형식)")
    stocks: List[StockDailyData] = Field(..., description="종목별 일별 데이터")


class BacktestResponse(BaseModel):
    """백테스트 응답 스키마"""
    portfolio_snapshot: PortfolioSnapshotResponse
    metrics: Optional[BacktestMetrics] = None
    result_summary: List[ResultSummary] = Field(..., description="결과 요약 데이터")
    execution_time: float = Field(..., description="실행 시간 (초)")

