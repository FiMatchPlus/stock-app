"""Common/shared schemas."""

from datetime import datetime, timezone, timedelta
from typing import Optional, List
from pydantic import BaseModel, Field

KST = timezone(timedelta(hours=9))

def get_kst_now():
    return datetime.now(KST)


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


__all__ = [
    "KST",
    "get_kst_now",
    "ErrorResponse",
    "MissingStockData",
    "BacktestDataError",
]
