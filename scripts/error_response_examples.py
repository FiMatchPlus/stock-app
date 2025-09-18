"""백테스트 오류 응답 예시 모음"""

import json
import sys
import os
from datetime import datetime

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.schemas import (
    BacktestErrorResponse, BacktestDataError, MissingStockData
)


def print_json_response(title: str, response_dict: dict):
    """JSON 응답을 예쁘게 출력"""
    print(f"\n{title}")
    print("=" * len(title))
    print(json.dumps(response_dict, indent=2, ensure_ascii=False, default=str))
    print()


def example_1_single_stock_missing():
    """예시 1: 단일 종목 데이터 누락"""
    missing_data = [
        MissingStockData(
            stock_code="005930",  # 삼성전자
            start_date="2024-01-01",
            end_date="2024-01-31",
            available_date_range="2023-01-01 ~ 2023-12-31"
        )
    ]
    
    error_response = BacktestErrorResponse(
        success=False,
        error=BacktestDataError(
            error_type="MISSING_STOCK_PRICE_DATA",
            message="종목 '005930'의 2024-01-01 ~ 2024-01-31 기간 주가 데이터를 찾을 수 없습니다.",
            missing_data=missing_data,
            requested_period="2024-01-01 ~ 2024-01-31",
            total_stocks=1,
            missing_stocks_count=1
        ),
        execution_time=0.045,
        request_id="req_001"
    )
    
    return error_response.model_dump()


def example_2_multiple_stocks_missing():
    """예시 2: 다중 종목 데이터 누락"""
    missing_data = [
        MissingStockData(
            stock_code="005930",  # 삼성전자
            start_date="2024-01-01",
            end_date="2024-01-31",
            available_date_range="2023-01-01 ~ 2023-12-31"
        ),
        MissingStockData(
            stock_code="000660",  # SK하이닉스
            start_date="2024-01-01",
            end_date="2024-01-31",
            available_date_range="2022-06-01 ~ 2023-11-30"
        ),
        MissingStockData(
            stock_code="035420",  # NAVER
            start_date="2024-01-01",
            end_date="2024-01-31",
            available_date_range=None  # 데이터 전혀 없음
        )
    ]
    
    error_response = BacktestErrorResponse(
        success=False,
        error=BacktestDataError(
            error_type="MISSING_STOCK_PRICE_DATA",
            message="3개 종목(005930, 000660, 035420)의 2024-01-01 ~ 2024-01-31 기간 주가 데이터를 찾을 수 없습니다.",
            missing_data=missing_data,
            requested_period="2024-01-01 ~ 2024-01-31",
            total_stocks=5,
            missing_stocks_count=3
        ),
        execution_time=0.089,
        request_id="req_002"
    )
    
    return error_response.model_dump()


def example_3_no_data_at_all():
    """예시 3: 모든 종목 데이터 없음"""
    missing_data = [
        MissingStockData(
            stock_code="005930",
            start_date="2025-01-01",
            end_date="2025-01-31",
            available_date_range="2020-01-01 ~ 2024-12-31"
        ),
        MissingStockData(
            stock_code="000660",
            start_date="2025-01-01",
            end_date="2025-01-31",
            available_date_range="2020-01-01 ~ 2024-12-31"
        )
    ]
    
    error_response = BacktestErrorResponse(
        success=False,
        error=BacktestDataError(
            error_type="MISSING_STOCK_PRICE_DATA",
            message="2개 종목(005930, 000660)의 2025-01-01 ~ 2025-01-31 기간 주가 데이터를 찾을 수 없습니다.",
            missing_data=missing_data,
            requested_period="2025-01-01 ~ 2025-01-31",
            total_stocks=2,
            missing_stocks_count=2
        ),
        execution_time=0.023,
        request_id="req_003"
    )
    
    return error_response.model_dump()


def example_4_validation_error():
    """예시 4: 입력 검증 오류"""
    error_response = BacktestErrorResponse(
        success=False,
        error=BacktestDataError(
            error_type="VALIDATION_ERROR",
            message="Start date must be before end date",
            missing_data=[],
            requested_period="2024-01-31 ~ 2024-01-01",  # 잘못된 날짜 순서
            total_stocks=2,
            missing_stocks_count=0
        ),
        execution_time=0.001,
        request_id="req_004"
    )
    
    return error_response.model_dump()


def example_5_empty_holdings_error():
    """예시 5: 보유 종목 없음 오류"""
    error_response = BacktestErrorResponse(
        success=False,
        error=BacktestDataError(
            error_type="VALIDATION_ERROR",
            message="At least one holding must be specified",
            missing_data=[],
            requested_period="2024-01-01 ~ 2024-01-31",
            total_stocks=0,
            missing_stocks_count=0
        ),
        execution_time=0.002,
        request_id="req_005"
    )
    
    return error_response.model_dump()


def example_6_insufficient_data():
    """예시 6: 데이터 부족 오류"""
    error_response = BacktestErrorResponse(
        success=False,
        error=BacktestDataError(
            error_type="INSUFFICIENT_DATA",
            message="종목 '005930'의 데이터가 부족합니다. 필요: 30일, 사용가능: 5일",
            missing_data=[],
            requested_period="2024-01-01 ~ 2024-01-31",
            total_stocks=1,
            missing_stocks_count=0
        ),
        execution_time=0.067,
        request_id="req_006"
    )
    
    return error_response.model_dump()


def example_7_database_connection_error():
    """예시 7: 데이터베이스 연결 오류"""
    error_response = BacktestErrorResponse(
        success=False,
        error=BacktestDataError(
            error_type="INTERNAL_ERROR",
            message="백테스트 실행 중 예상치 못한 오류가 발생했습니다.",
            missing_data=[],
            requested_period="2024-01-01 ~ 2024-01-31",
            total_stocks=3,
            missing_stocks_count=0
        ),
        execution_time=0.156,
        request_id="req_007"
    )
    
    return error_response.model_dump()


def example_8_invalid_stock_quantity():
    """예시 8: 잘못된 종목 수량 오류"""
    error_response = BacktestErrorResponse(
        success=False,
        error=BacktestDataError(
            error_type="VALIDATION_ERROR",
            message="Quantity must be positive for 005930",
            missing_data=[],
            requested_period="2024-01-01 ~ 2024-01-31",
            total_stocks=1,
            missing_stocks_count=0
        ),
        execution_time=0.003,
        request_id="req_008"
    )
    
    return error_response.model_dump()


def example_9_partial_data_missing():
    """예시 9: 일부 종목만 데이터 누락 (혼합 상황)"""
    missing_data = [
        MissingStockData(
            stock_code="123456",  # 존재하지 않는 종목
            start_date="2024-01-01",
            end_date="2024-01-31",
            available_date_range=None
        )
    ]
    
    error_response = BacktestErrorResponse(
        success=False,
        error=BacktestDataError(
            error_type="MISSING_STOCK_PRICE_DATA",
            message="1개 종목(123456)의 2024-01-01 ~ 2024-01-31 기간 주가 데이터를 찾을 수 없습니다.",
            missing_data=missing_data,
            requested_period="2024-01-01 ~ 2024-01-31",
            total_stocks=3,  # 3개 중 1개만 누락
            missing_stocks_count=1
        ),
        execution_time=0.078,
        request_id="req_009"
    )
    
    return error_response.model_dump()


def example_10_many_stocks_missing():
    """예시 10: 많은 종목 데이터 누락 (표시 제한)"""
    missing_data = []
    stock_codes = ["005930", "000660", "035420", "005380", "051910", "006400", "035720"]
    
    for code in stock_codes:
        missing_data.append(
            MissingStockData(
                stock_code=code,
                start_date="2024-01-01",
                end_date="2024-01-31",
                available_date_range="2023-01-01 ~ 2023-12-31"
            )
        )
    
    error_response = BacktestErrorResponse(
        success=False,
        error=BacktestDataError(
            error_type="MISSING_STOCK_PRICE_DATA",
            message="7개 종목(005930, 000660, 035420 외 4개)의 2024-01-01 ~ 2024-01-31 기간 주가 데이터를 찾을 수 없습니다.",
            missing_data=missing_data,
            requested_period="2024-01-01 ~ 2024-01-31",
            total_stocks=10,
            missing_stocks_count=7
        ),
        execution_time=0.134,
        request_id="req_010"
    )
    
    return error_response.model_dump()


def main():
    """모든 오류 응답 예시 출력"""
    print("백테스트 오류 응답 예시 모음")
    print("=" * 50)
    
    examples = [
        ("1. 단일 종목 데이터 누락", example_1_single_stock_missing()),
        ("2. 다중 종목 데이터 누락", example_2_multiple_stocks_missing()),
        ("3. 모든 종목 데이터 없음", example_3_no_data_at_all()),
        ("4. 입력 검증 오류 (날짜 순서)", example_4_validation_error()),
        ("5. 보유 종목 없음 오류", example_5_empty_holdings_error()),
        ("6. 데이터 부족 오류", example_6_insufficient_data()),
        ("7. 데이터베이스 연결 오류", example_7_database_connection_error()),
        ("8. 잘못된 종목 수량 오류", example_8_invalid_stock_quantity()),
        ("9. 일부 종목만 데이터 누락", example_9_partial_data_missing()),
        ("10. 많은 종목 데이터 누락", example_10_many_stocks_missing())
    ]
    
    for title, response in examples:
        print_json_response(title, response)
    
    print("=" * 50)
    print("총 10가지 오류 응답 예시를 확인했습니다.")
    print("\n오류 타입별 분류:")
    print("- MISSING_STOCK_PRICE_DATA: 주가 데이터 누락 (예시 1, 2, 3, 9, 10)")
    print("- VALIDATION_ERROR: 입력 검증 오류 (예시 4, 5, 8)")
    print("- INSUFFICIENT_DATA: 데이터 부족 (예시 6)")
    print("- INTERNAL_ERROR: 서버 내부 오류 (예시 7)")


if __name__ == "__main__":
    main()
