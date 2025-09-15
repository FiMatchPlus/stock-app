#!/usr/bin/env python3
"""백테스트 API 테스트 스크립트"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
import httpx

# 테스트 데이터
TEST_BACKTEST_REQUEST = {
    "start": (datetime.now() - timedelta(days=30)).isoformat(),
    "end": datetime.now().isoformat(),
    "holdings": [
        {"code": "005930", "weight": 0.5},  # 삼성전자
        {"code": "000660", "weight": 0.3},  # SK하이닉스
        {"code": "035420", "weight": 0.2}   # NAVER
    ],
    "initial_capital": 1000000,
    "rebalance_frequency": "daily"
}

BASE_URL = "http://localhost:8000"

async def test_backtest_api():
    """백테스트 API 테스트"""
    async with httpx.AsyncClient() as client:
        try:
            # 1. 헬스 체크
            print("1. Testing backtest health check...")
            response = await client.get(f"{BASE_URL}/backtest/health")
            print(f"Health check status: {response.status_code}")
            print(f"Response: {response.json()}")
            print()
            
            # 2. 성과 지표 설명 조회
            print("2. Testing metrics explanation...")
            response = await client.get(f"{BASE_URL}/backtest/metrics/explanation")
            print(f"Metrics explanation status: {response.status_code}")
            if response.status_code == 200:
                metrics = response.json()
                print(f"Available metrics: {list(metrics['metrics'].keys())}")
            print()
            
            # 3. 백테스트 실행
            print("3. Testing backtest execution...")
            response = await client.post(
                f"{BASE_URL}/backtest/run",
                json=TEST_BACKTEST_REQUEST
            )
            print(f"Backtest execution status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Portfolio snapshot ID: {result['portfolio_snapshot']['id']}")
                print(f"Execution time: {result['execution_time']:.3f}s")
                
                if result['metrics']:
                    metrics = result['metrics']
                    print(f"Total return: {metrics['total_return']}")
                    print(f"Volatility: {metrics['volatility']}")
                    print(f"Sharpe ratio: {metrics['sharpe_ratio']}")
                    print(f"Max drawdown: {metrics['max_drawdown']}")
                
                print(f"Daily returns count: {len(result['daily_returns'])}")
                
            else:
                print(f"Error: {response.text}")
            print()
            
            # 4. 백테스트 히스토리 조회
            print("4. Testing backtest history...")
            response = await client.get(f"{BASE_URL}/backtest/history")
            print(f"History status: {response.status_code}")
            
            if response.status_code == 200:
                history = response.json()
                print(f"Total snapshots: {history['total_count']}")
                print(f"Returned snapshots: {len(history['snapshots'])}")
            
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Test failed: {str(e)}")

async def test_with_invalid_data():
    """잘못된 데이터로 테스트"""
    print("\n5. Testing with invalid data...")
    
    async with httpx.AsyncClient() as client:
        # 잘못된 가중치 합계
        invalid_request = {
            "start": (datetime.now() - timedelta(days=7)).isoformat(),
            "end": datetime.now().isoformat(),
            "holdings": [
                {"code": "005930", "weight": 0.6},
                {"code": "000660", "weight": 0.6}  # 합계가 1.2
            ],
            "initial_capital": 1000000
        }
        
        response = await client.post(
            f"{BASE_URL}/backtest/run",
            json=invalid_request
        )
        
        print(f"Invalid data test status: {response.status_code}")
        if response.status_code != 200:
            print(f"Expected error: {response.json()}")

if __name__ == "__main__":
    print("Starting backtest API tests...")
    print("Make sure the server is running on http://localhost:8000")
    print("=" * 60)
    
    asyncio.run(test_backtest_api())
    asyncio.run(test_with_invalid_data())
    
    print("\n" + "=" * 60)
    print("Test completed!")

