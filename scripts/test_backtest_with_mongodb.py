#!/usr/bin/env python3
"""MongoDB 연동 백테스트 테스트 스크립트"""

import asyncio
import json
from datetime import datetime, timedelta
import httpx

# 테스트 데이터 (2025년 1월 2일부터 8월까지)
TEST_BACKTEST_REQUEST = {
    "start": "2025-01-02T00:00:00",
    "end": "2025-08-31T23:59:59",
    "holdings": [
        {"code": "005930", "weight": 0.5},  # 삼성전자
        {"code": "000660", "weight": 0.3},  # SK하이닉스
        {"code": "035420", "weight": 0.2}   # NAVER
    ],
    "initial_capital": 1000000,
    "rebalance_frequency": "daily"
}

BASE_URL = "http://localhost:8000"

async def test_backtest_with_mongodb():
    """MongoDB 연동 백테스트 테스트"""
    async with httpx.AsyncClient() as client:
        try:
            # 1. 백테스트 실행
            print("1. 백테스트 실행 중...")
            response = await client.post(
                f"{BASE_URL}/backtest/run",
                json=TEST_BACKTEST_REQUEST
            )
            
            if response.status_code == 200:
                result = response.json()
                portfolio_snapshot = result['portfolio_snapshot']
                metric_id = portfolio_snapshot.get('metric_id')
                
                print(f"백테스트 실행 성공")
                print(f"   - Portfolio Snapshot ID: {portfolio_snapshot['id']}")
                print(f"   - Metric ID: {metric_id}")
                print(f"   - 실행 시간: {result['execution_time']:.3f}초")
                
                if result['metrics']:
                    metrics = result['metrics']
                    print(f"   - 총 수익률: {metrics['total_return']}")
                    print(f"   - 변동성: {metrics['volatility']}")
                    print(f"   - 샤프 비율: {metrics['sharpe_ratio']}")
                    print(f"   - 최대 낙폭: {metrics['max_drawdown']}")
                
                # 2. MongoDB에서 metrics 조회
                if metric_id:
                    print(f"\n2. MongoDB에서 metrics 조회 중...")
                    metrics_response = await client.get(
                        f"{BASE_URL}/backtest/metrics/{metric_id}"
                    )
                    
                    if metrics_response.status_code == 200:
                        metrics_data = metrics_response.json()
                        print("MongoDB metrics 조회 성공")
                        print(f"   - Metric ID: {metrics_data['data']['_id']}")
                        print(f"   - Portfolio Snapshot ID: {metrics_data['data']['portfolio_snapshot_id']}")
                        print(f"   - 총 수익률: {metrics_data['data']['total_return']}")
                        print(f"   - 샤프 비율: {metrics_data['data']['sharpe_ratio']}")
                    else:
                        print(f"MongoDB metrics 조회 실패: {metrics_response.status_code}")
                        print(f"   응답: {metrics_response.text}")
                
                # 3. Portfolio Snapshot ID로 metrics 조회
                print(f"\n3. Portfolio Snapshot ID로 metrics 조회 중...")
                portfolio_metrics_response = await client.get(
                    f"{BASE_URL}/backtest/metrics/portfolio/{portfolio_snapshot['id']}"
                )
                
                if portfolio_metrics_response.status_code == 200:
                    portfolio_metrics_data = portfolio_metrics_response.json()
                    print("Portfolio Snapshot ID로 metrics 조회 성공")
                    print(f"   - Metric ID: {portfolio_metrics_data['data']['_id']}")
                else:
                    print(f"Portfolio Snapshot ID로 metrics 조회 실패: {portfolio_metrics_response.status_code}")
                
                # 4. 백테스트 히스토리 조회
                print(f"\n4. 백테스트 히스토리 조회 중...")
                history_response = await client.get(f"{BASE_URL}/backtest/history")
                
                if history_response.status_code == 200:
                    history_data = history_response.json()
                    print("백테스트 히스토리 조회 성공")
                    print(f"   - 총 스냅샷 수: {history_data['total_count']}")
                    
                    if history_data['snapshots']:
                        latest_snapshot = history_data['snapshots'][0]
                        print(f"   - 최신 스냅샷 ID: {latest_snapshot['id']}")
                        print(f"   - Metric ID: {latest_snapshot.get('metric_id', 'None')}")
                else:
                    print(f"백테스트 히스토리 조회 실패: {history_response.status_code}")
                
            else:
                print(f"백테스트 실행 실패: {response.status_code}")
                print(f"   응답: {response.text}")
                
        except Exception as e:
            print(f"테스트 실패: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("MongoDB 연동 백테스트 테스트 시작...")
    print("=" * 60)
    
    asyncio.run(test_backtest_with_mongodb())
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
