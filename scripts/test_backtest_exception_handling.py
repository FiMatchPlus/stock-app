"""백테스트 예외 처리 테스트"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import List

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.schemas import BacktestRequest, Holding
from app.exceptions import MissingStockPriceDataException


def test_missing_stock_price_exception():
    """주가 데이터 누락 예외 테스트"""
    print("=== 주가 데이터 누락 예외 테스트 ===")
    
    # 테스트 데이터 준비
    missing_stocks = [
        {
            'stock_code': '005930',  # 삼성전자
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'available_date_range': '2023-01-01 ~ 2023-12-31'
        },
        {
            'stock_code': '000660',  # SK하이닉스
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'available_date_range': None  # 데이터 전혀 없음
        }
    ]
    
    requested_period = "2024-01-01 ~ 2024-01-31"
    total_stocks = 3
    
    # 예외 생성 및 테스트
    try:
        raise MissingStockPriceDataException(
            missing_stocks=missing_stocks,
            requested_period=requested_period,
            total_stocks=total_stocks
        )
    except MissingStockPriceDataException as e:
        print(f"예외 타입: {e.error_type}")
        print(f"메시지: {e.message}")
        print(f"누락 종목 수: {e.missing_stocks_count}")
        print(f"전체 종목 수: {e.total_stocks}")
        print(f"요청 기간: {e.requested_period}")
        
        print("\n누락된 종목 상세:")
        for i, stock in enumerate(e.missing_stocks, 1):
            print(f"  {i}. 종목코드: {stock['stock_code']}")
            print(f"     요청기간: {stock['start_date']} ~ {stock['end_date']}")
            if stock['available_date_range']:
                print(f"     사용가능: {stock['available_date_range']}")
            else:
                print(f"     사용가능: 데이터 없음")
            print()


def test_backtest_request_validation():
    """백테스트 요청 검증 테스트"""
    print("=== 백테스트 요청 검증 테스트 ===")
    
    # 올바른 요청
    try:
        valid_request = BacktestRequest(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            holdings=[
                Holding(code="005930", quantity=10),
                Holding(code="000660", quantity=5)
            ]
        )
        print("올바른 요청 생성 성공")
        print(f"   시작일: {valid_request.start}")
        print(f"   종료일: {valid_request.end}")
        print(f"   보유종목: {len(valid_request.holdings)}개")
        
    except Exception as e:
        print(f"올바른 요청 생성 실패: {e}")
    
    # 잘못된 요청 (수량이 0)
    try:
        invalid_request = BacktestRequest(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            holdings=[
                Holding(code="005930", quantity=0),  # 잘못된 수량
            ]
        )
        print("잘못된 요청이 통과됨")
        
    except Exception as e:
        print(f"잘못된 요청 차단됨: {e}")


def create_sample_error_response():
    """샘플 오류 응답 생성"""
    print("=== 샘플 오류 응답 생성 ===")
    
    from app.models.schemas import (
        BacktestErrorResponse, BacktestDataError, MissingStockData
    )
    
    # 누락 데이터 생성
    missing_data = [
        MissingStockData(
            stock_code="005930",
            start_date="2024-01-01",
            end_date="2024-01-31",
            available_date_range="2023-01-01 ~ 2023-12-31"
        ),
        MissingStockData(
            stock_code="000660",
            start_date="2024-01-01", 
            end_date="2024-01-31",
            available_date_range=None
        )
    ]
    
    # 오류 응답 생성
    error_response = BacktestErrorResponse(
        success=False,
        error=BacktestDataError(
            error_type="MISSING_STOCK_PRICE_DATA",
            message="2개 종목(005930, 000660)의 2024-01-01 ~ 2024-01-31 기간 주가 데이터를 찾을 수 없습니다.",
            missing_data=missing_data,
            requested_period="2024-01-01 ~ 2024-01-31",
            total_stocks=3,
            missing_stocks_count=2
        ),
        execution_time=0.123
    )
    
    # JSON으로 변환하여 출력
    print("구조화된 오류 응답:")
    import json
    response_dict = error_response.model_dump()
    print(json.dumps(response_dict, indent=2, ensure_ascii=False))
    
    return error_response


def main():
    """메인 테스트 함수"""
    print("백테스트 예외 처리 테스트 시작\n")
    
    test_missing_stock_price_exception()
    print("\n" + "="*50 + "\n")
    
    test_backtest_request_validation()
    print("\n" + "="*50 + "\n")
    
    create_sample_error_response()
    print("\n" + "="*50 + "\n")
    
    print("모든 테스트 완료!")


if __name__ == "__main__":
    main()
