#!/usr/bin/env python3
"""손실 한계선 규칙 테스트 스크립트"""

import json
from datetime import datetime, timedelta

# 손실 한계선 규칙 테스트 JSON 예시
def generate_test_request():
    """손실 한계선을 포함한 백테스트 요청 예시 생성"""
    
    # 현재 날짜 기준으로 1년 전부터 테스트
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    test_request = {
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "holdings": [
            {
                "code": "005930",  # 삼성전자
                "quantity": 100
            },
            {
                "code": "000660",  # SK하이닉스
                "quantity": 50
            }
        ],
        "rebalance_frequency": "daily",
        "callback_url": "http://localhost:8080/test/callback",
        "rules": {
            "stopLoss": [
                {
                    "category": "LOSS_LIMIT",
                    "value": -0.10  # -10% 손실 시 손절
                },
                {
                    "category": "MDD",
                    "value": 0.15   # 15% 최대낙폭 시 손절
                }
            ],
            "takeProfit": [
                {
                    "category": "ONEPROFIT",
                    "value": 0.20   # 20% 수익 시 익절
                }
            ]
        },
        "risk_free_rate": 0.035,
        "benchmark_code": "KOSPI",
        "backtest_id": 99999
    }
    
    return test_request

def print_usage_examples():
    """사용 예시 출력"""
    
    print("=" * 60)
    print("손실 한계선 (LOSS_LIMIT) 규칙 사용 가이드")
    print("=" * 60)
    
    print("\n1. 기본 사용법:")
    print("""
{
  "rules": {
    "stopLoss": [
      {
        "category": "LOSS_LIMIT",
        "value": -0.10        // -10% 손실 시 손절
      }
    ]
  }
}
    """)
    
    print("\n2. 다양한 손실 한계선 설정:")
    print("""
// 보수적 설정 (5% 손실)
{"category": "LOSS_LIMIT", "value": -0.05}

// 일반적 설정 (10% 손실)  
{"category": "LOSS_LIMIT", "value": -0.10}

// 공격적 설정 (20% 손실)
{"category": "LOSS_LIMIT", "value": -0.20}
    """)
    
    print("\n3. 다른 손절 조건과 함께 사용:")
    print("""
{
  "rules": {
    "stopLoss": [
      {
        "category": "LOSS_LIMIT",
        "value": -0.15        // 절대 손실 15%
      },
      {
        "category": "MDD", 
        "value": 0.12         // 최대낙폭 12%
      },
      {
        "category": "VAR",
        "value": 0.05         // VaR 5%
      }
    ]
  }
}
    """)
    
    print("\n4. 손실 한계선의 특징:")
    print("- 초기 투자금액 대비 절대적인 손실 비율")
    print("- MDD와 달리 중간 고점이 아닌 투자 원금 기준")
    print("- 리밸런싱과 관계없이 전체 투자 기간 동안의 누적 손실")
    print("- 값은 항상 음수로 입력 (예: -0.10 = -10%)")
    
    print("\n5. 실행 로그 예시:")
    print("""
{
  "date": "2024-03-15",
  "action": "STOP_LOSS",
  "category": "LOSS_LIMIT", 
  "value": -0.1234,         // 현재 수익률: -12.34%
  "threshold": -0.10,       // 설정 임계값: -10%
  "reason": "LOSS_LIMIT 손절: -0.1234 < -0.1000",
  "portfolio_value": 8766000
}
    """)

def main():
    """메인 실행 함수"""
    
    # 사용 가이드 출력
    print_usage_examples()
    
    print("\n" + "=" * 60)
    print("테스트 요청 JSON 생성")
    print("=" * 60)
    
    # 테스트 요청 생성
    test_request = generate_test_request()
    
    # JSON 출력
    print("\n테스트용 백테스트 요청:")
    print(json.dumps(test_request, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("API 테스트 방법")
    print("=" * 60)
    
    print("""
1. 동기 백테스트 테스트:
curl -X POST "http://localhost:8000/backtest/run" \\
  -H "Content-Type: application/json" \\
  -d '{
    "start": "2023-01-01T00:00:00",
    "end": "2024-01-01T00:00:00", 
    "holdings": [{"code": "005930", "quantity": 100}],
    "rules": {
      "stopLoss": [{"category": "LOSS_LIMIT", "value": -0.10}]
    }
  }'

2. 비동기 백테스트 테스트:
curl -X POST "http://localhost:8000/backtest/start" \\
  -H "Content-Type: application/json" \\
  -d '{
    "start": "2023-01-01T00:00:00",
    "end": "2024-01-01T00:00:00",
    "holdings": [{"code": "005930", "quantity": 100}],
    "callback_url": "http://localhost:8080/callback",
    "rules": {
      "stopLoss": [{"category": "LOSS_LIMIT", "value": -0.15}]
    }
  }'
    """)

if __name__ == "__main__":
    main()
