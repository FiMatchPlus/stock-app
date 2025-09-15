#!/usr/bin/env python3
"""백테스트 디버그 스크립트"""

import asyncio
from datetime import datetime
from app.services.backtest_service import BacktestService
from app.models.schemas import BacktestRequest, Holding
from app.models.database import get_async_session
from app.utils.mongodb_client import mongodb_client, MONGODB_URL, MONGODB_DATABASE

async def debug_backtest():
    """백테스트 디버그"""
    try:
        # MongoDB 연결
        print("1. MongoDB 연결 중...")
        await mongodb_client.connect(MONGODB_URL, MONGODB_DATABASE)
        print("✅ MongoDB 연결 성공")
        
        # 백테스트 서비스 생성
        print("2. 백테스트 서비스 생성 중...")
        backtest_service = BacktestService(mongodb_client)
        print("✅ 백테스트 서비스 생성 성공")
        
        # 테스트 요청 생성
        print("3. 테스트 요청 생성 중...")
        request = BacktestRequest(
            start=datetime(2025, 1, 2),
            end=datetime(2025, 1, 3),
            holdings=[
                Holding(code="005930", weight=1.0)
            ],
            initial_capital=1000000
        )
        print("✅ 테스트 요청 생성 성공")
        
        # 데이터베이스 세션 생성
        print("4. 데이터베이스 세션 생성 중...")
        async for session in get_async_session():
            try:
                # 백테스트 실행
                print("5. 백테스트 실행 중...")
                result = await backtest_service.run_backtest(
                    request=request,
                    session=session,
                    portfolio_id=None
                )
                
                print("✅ 백테스트 실행 성공!")
                print(f"   - Portfolio Snapshot ID: {result.portfolio_snapshot.id}")
                print(f"   - Metric ID: {result.portfolio_snapshot.metric_id}")
                print(f"   - 실행 시간: {result.execution_time:.3f}초")
                
                if result.metrics:
                    metrics = result.metrics
                    print(f"   - 총 수익률: {metrics.total_return}")
                    print(f"   - 변동성: {metrics.volatility}")
                    print(f"   - 샤프 비율: {metrics.sharpe_ratio}")
                
                break
                
            except Exception as e:
                print(f"❌ 백테스트 실행 실패: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                await session.close()
        
    except Exception as e:
        print(f"❌ 디버그 실패: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # MongoDB 연결 해제
        await mongodb_client.disconnect()
        print("MongoDB 연결 해제")

if __name__ == "__main__":
    print("백테스트 디버그 시작...")
    print("=" * 50)
    
    asyncio.run(debug_backtest())
    
    print("=" * 50)
    print("디버그 완료!")
