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


class WebSocketMessage(BaseModel):
    """WebSocket 메시지 스키마"""
    type: str = Field(..., description="메시지 타입")
    symbol: Optional[str] = Field(None, description="종목코드")
    data: Optional[Dict[str, Any]] = Field(None, description="데이터")
    timestamp: datetime = Field(default_factory=get_kst_now, description="타임스탬프")


class ErrorResponse(BaseModel):
    """에러 응답 스키마"""
    error: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 에러 정보")
    timestamp: datetime = Field(default_factory=get_kst_now, description="에러 발생 시간")

