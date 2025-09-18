"""백테스트 관련 커스텀 예외 클래스"""

from typing import List, Dict, Any, Optional
from datetime import datetime


class BacktestDataException(Exception):
    """백테스트 데이터 관련 예외 기본 클래스"""
    
    def __init__(
        self,
        message: str,
        error_type: str = "DATA_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        super().__init__(self.message)


class MissingStockPriceDataException(BacktestDataException):
    """주가 데이터 누락 예외"""
    
    def __init__(
        self,
        missing_stocks: List[Dict[str, Any]],
        requested_period: str,
        total_stocks: int,
        message: Optional[str] = None
    ):
        self.missing_stocks = missing_stocks
        self.requested_period = requested_period
        self.total_stocks = total_stocks
        self.missing_stocks_count = len(missing_stocks)
        
        # 기본 메시지 생성
        if message is None:
            if self.missing_stocks_count == 1:
                stock_info = missing_stocks[0]
                message = (
                    f"종목 '{stock_info['stock_code']}'의 "
                    f"{requested_period} 기간 주가 데이터를 찾을 수 없습니다."
                )
            else:
                stock_codes = [stock['stock_code'] for stock in missing_stocks[:3]]
                display_codes = ', '.join(stock_codes)
                if self.missing_stocks_count > 3:
                    display_codes += f" 외 {self.missing_stocks_count - 3}개"
                
                message = (
                    f"{self.missing_stocks_count}개 종목({display_codes})의 "
                    f"{requested_period} 기간 주가 데이터를 찾을 수 없습니다."
                )
        
        details = {
            "missing_stocks": missing_stocks,
            "requested_period": requested_period,
            "total_stocks": total_stocks,
            "missing_stocks_count": self.missing_stocks_count
        }
        
        super().__init__(
            message=message,
            error_type="MISSING_STOCK_PRICE_DATA",
            details=details
        )


class InsufficientDataException(BacktestDataException):
    """데이터 부족 예외"""
    
    def __init__(
        self,
        stock_code: str,
        available_days: int,
        required_days: int,
        message: Optional[str] = None
    ):
        self.stock_code = stock_code
        self.available_days = available_days
        self.required_days = required_days
        
        if message is None:
            message = (
                f"종목 '{stock_code}'의 데이터가 부족합니다. "
                f"필요: {required_days}일, 사용가능: {available_days}일"
            )
        
        details = {
            "stock_code": stock_code,
            "available_days": available_days,
            "required_days": required_days
        }
        
        super().__init__(
            message=message,
            error_type="INSUFFICIENT_DATA",
            details=details
        )


class BacktestValidationException(BacktestDataException):
    """백테스트 입력 검증 예외"""
    
    def __init__(
        self,
        validation_errors: List[str],
        message: Optional[str] = None
    ):
        self.validation_errors = validation_errors
        
        if message is None:
            if len(validation_errors) == 1:
                message = f"입력 검증 오류: {validation_errors[0]}"
            else:
                message = f"입력 검증 오류: {len(validation_errors)}개의 문제가 발견되었습니다."
        
        details = {
            "validation_errors": validation_errors
        }
        
        super().__init__(
            message=message,
            error_type="VALIDATION_ERROR",
            details=details
        )
