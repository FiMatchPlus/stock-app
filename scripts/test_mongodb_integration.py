#!/usr/bin/env python3
"""MongoDB 연동 테스트 스크립트"""

import asyncio
import os
from datetime import datetime, timedelta
from decimal import Decimal

from app.models.schemas import BacktestMetrics
from app.services.metrics_service import MetricsService
from app.utils.mongodb_client import mongodb_client, MONGODB_URL, MONGODB_DATABASE

async def test_mongodb_integration():
    """MongoDB 연동 테스트"""
    try:
        print("1. MongoDB 연결 테스트...")
        await mongodb_client.connect(MONGODB_URL, MONGODB_DATABASE)
        print("MongoDB 연결 성공")
        
        print("\n2. Metrics 서비스 테스트...")
        metrics_service = MetricsService(mongodb_client)
        
        # 테스트용 metrics 생성
        test_metrics = BacktestMetrics(
            total_return=Decimal("0.15"),
            annualized_return=Decimal("0.12"),
            volatility=Decimal("0.20"),
            sharpe_ratio=Decimal("0.60"),
            max_drawdown=Decimal("-0.08"),
            var_95=Decimal("-0.03"),
            var_99=Decimal("-0.05"),
            cvar_95=Decimal("-0.04"),
            cvar_99=Decimal("-0.06"),
            win_rate=Decimal("0.55"),
            profit_loss_ratio=Decimal("1.25")
        )
        
        print("3. Metrics 저장 테스트...")
        metric_id = await metrics_service.save_metrics(test_metrics, 1)
        print(f"Metrics 저장 성공 - ID: {metric_id}")
        
        print("4. Metrics 조회 테스트...")
        retrieved_metrics = await metrics_service.get_metrics(metric_id)
        if retrieved_metrics:
            print("Metrics 조회 성공")
            print(f"   - 총 수익률: {retrieved_metrics['total_return']}")
            print(f"   - 샤프 비율: {retrieved_metrics['sharpe_ratio']}")
            print(f"   - 최대 낙폭: {retrieved_metrics['max_drawdown']}")
        else:
            print("Metrics 조회 실패")
        
        print("5. Portfolio Snapshot ID로 조회 테스트...")
        portfolio_metrics = await metrics_service.get_metrics_by_portfolio_snapshot(1)
        if portfolio_metrics:
            print("Portfolio Snapshot ID로 조회 성공")
            print(f"   - Metric ID: {portfolio_metrics['_id']}")
        else:
            print("Portfolio Snapshot ID로 조회 실패")
        
        print("6. Metrics 업데이트 테스트...")
        updated_metrics = BacktestMetrics(
            total_return=Decimal("0.18"),
            annualized_return=Decimal("0.15"),
            volatility=Decimal("0.18"),
            sharpe_ratio=Decimal("0.83"),
            max_drawdown=Decimal("-0.06"),
            var_95=Decimal("-0.02"),
            var_99=Decimal("-0.04"),
            cvar_95=Decimal("-0.03"),
            cvar_99=Decimal("-0.05"),
            win_rate=Decimal("0.60"),
            profit_loss_ratio=Decimal("1.35")
        )
        
        update_success = await metrics_service.update_metrics(metric_id, updated_metrics)
        if update_success:
            print("Metrics 업데이트 성공")
        else:
            print("Metrics 업데이트 실패")
        
        print("7. 히스토리 조회 테스트...")
        history = await metrics_service.get_metrics_history(1, limit=10)
        print(f"히스토리 조회 성공 - {len(history)}개 항목")
        
        print("8. Metrics 삭제 테스트...")
        delete_success = await metrics_service.delete_metrics(metric_id)
        if delete_success:
            print("Metrics 삭제 성공")
        else:
            print("Metrics 삭제 실패")
        
        print("\n🎉 모든 MongoDB 연동 테스트 완료!")
        
    except Exception as e:
        print(f"테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # MongoDB 연결 해제
        await mongodb_client.disconnect()
        print("MongoDB 연결 해제")

if __name__ == "__main__":
    print("MongoDB 연동 테스트 시작...")
    print("=" * 50)
    
    asyncio.run(test_mongodb_integration())
    
    print("=" * 50)
    print("테스트 완료!")
