"""
손절/익절 규칙을 사용한 백테스트 예시

이 파일은 새로운 손절/익절 기능을 사용하는 방법을 보여줍니다.
"""

from datetime import datetime
from app.models.schemas import BacktestRequest, Holding, TradingRule, TradingRules
from app.services.benchmark_service import BenchmarkService

# 예시 1: 베타 손절 규칙
def create_beta_stop_loss_example():
    """베타가 1.5를 초과하면 손절하는 백테스트"""
    
    request = BacktestRequest(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),
        holdings=[
            Holding(code="005930", quantity=100, avg_price=70000),  # 삼성전자
            Holding(code="000660", quantity=50, avg_price=400000)   # SK하이닉스
        ],
        rules=TradingRules(
            stopLoss=[
                TradingRule(category="BETA", value=1.5)  # 베타가 1.5 초과시 손절
            ]
        )
    )
    
    return request

# 예시 2: MDD 손절 규칙
def create_mdd_stop_loss_example():
    """최대 낙폭이 15%를 초과하면 손절하는 백테스트"""
    
    request = BacktestRequest(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),
        holdings=[
            Holding(code="005930", quantity=100, avg_price=70000),
            Holding(code="000660", quantity=50, avg_price=400000)
        ],
        rules=TradingRules(
            stopLoss=[
                TradingRule(category="MDD", value=0.15)  # MDD가 15% 초과시 손절
            ]
        )
    )
    
    return request

# 예시 3: VaR 손절 규칙
def create_var_stop_loss_example():
    """95% VaR이 5%를 초과하면 손절하는 백테스트"""
    
    request = BacktestRequest(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),
        holdings=[
            Holding(code="005930", quantity=100, avg_price=70000),
            Holding(code="000660", quantity=50, avg_price=400000)
        ],
        rules=TradingRules(
            stopLoss=[
                TradingRule(category="VAR", value=0.05)  # VaR이 5% 초과시 손절
            ]
        )
    )
    
    return request

# 예시 4: 단일 종목 수익률 익절 규칙
def create_oneprofit_take_profit_example():
    """단일 종목이 30% 수익을 달성하면 익절하는 백테스트"""
    
    request = BacktestRequest(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),
        holdings=[
            Holding(code="005930", quantity=100, avg_price=70000),
            Holding(code="000660", quantity=50, avg_price=400000)
        ],
        rules=TradingRules(
            takeProfit=[
                TradingRule(category="ONEPROFIT", value=0.30)  # 단일 종목 30% 익절
            ]
        )
    )
    
    return request

# 예시 5: 복합 규칙 (손절 + 익절)
def create_complex_rules_example():
    """복합 규칙: MDD 15% 손절 + 단일 종목 25% 익절"""
    
    request = BacktestRequest(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),
        holdings=[
            Holding(code="005930", quantity=100, avg_price=70000),
            Holding(code="000660", quantity=50, avg_price=400000),
            Holding(code="035420", quantity=30, avg_price=1000000)  # 네이버
        ],
        rules=TradingRules(
            stopLoss=[
                TradingRule(category="MDD", value=0.15),    # MDD 15% 손절
                TradingRule(category="VAR", value=0.08),    # VaR 8% 손절
                TradingRule(category="BETA", value=2.0)     # 베타 2.0 손절
            ],
            takeProfit=[
                TradingRule(category="ONEPROFIT", value=0.25)  # 단일 종목 25% 익절
            ]
        )
    )
    
    return request

# 예시 6: API 요청 JSON 형태
def create_api_request_example():
    """실제 API 요청에서 사용할 JSON 형태"""
    
    return {
        "start": "2023-01-01T00:00:00",
        "end": "2023-12-31T23:59:59",
        "holdings": [
            {
                "code": "005930",
                "quantity": 100,
                "avg_price": 70000
            },
            {
                "code": "000660", 
                "quantity": 50,
                "avg_price": 400000
            }
        ],
        "rebalance_frequency": "daily",
        "rules": {
            "stopLoss": [
                {
                    "category": "MDD",
                    "value": 0.15
                },
                {
                    "category": "VAR", 
                    "value": 0.05
                }
            ],
            "takeProfit": [
                {
                    "category": "ONEPROFIT",
                    "value": 0.30
                }
            ]
        },
        "risk_free_rate": 3.25
    }

# 예시 7: 응답 JSON 형태
def create_api_response_example():
    """백테스트 응답 JSON 형태"""
    
    return {
        "success": True,
        "portfolio_snapshot": {
            "id": 12345,
            "portfolio_id": 67890,
            "base_value": 10000000,
            "current_value": 11500000,
            "start_at": "2023-01-01T00:00:00",
            "end_at": "2023-12-31T23:59:59",
            "created_at": "2023-12-31T23:59:59",
            "execution_time": 2.5,
            "holdings": [
                {
                    "id": 1,
                    "stock_id": "005930",
                    "quantity": 0  # 손절로 인해 청산됨
                }
            ]
        },
        "metrics": {
            "total_return": 0.15,
            "annualized_return": 0.15,
            "volatility": 0.20,
            "sharpe_ratio": 0.75,
            "max_drawdown": -0.12,
            "var_95": -0.03,
            "var_99": -0.05,
            "cvar_95": -0.04,
            "cvar_99": -0.06,
            "win_rate": 0.55,
            "profit_loss_ratio": 1.2
        },
        "result_summary": [
            # 일별 데이터...
        ],
        "execution_time": 2.5,
        "request_id": "req_1704067800000",
        "execution_logs": [
            {
                "date": "2023-06-15T00:00:00",
                "action": "STOP_LOSS",
                "category": "MDD",
                "value": 0.16,
                "threshold": 0.15,
                "reason": "MDD 손절: 0.1600 > 0.1500",
                "portfolio_value": 9500000
            }
        ],
        "result_status": "LIQUIDATED",
        "benchmark_info": {
            "benchmark_code": "KOSPI",
            "latest_price": 2450.0,
            "latest_date": "2023-12-31T00:00:00",
            "data_range": {
                "start_date": "2023-01-01T00:00:00",
                "end_date": "2023-12-31T00:00:00"
            },
            "latest_change_rate": 0.02
        },
        "risk_free_rate_info": {
            "rate_type": "TB3Y",
            "avg_annual_rate": 3.25,
            "data_points": 252,
            "decision_info": {
                "backtest_days": 365,
                "period_classification": "medium_term",
                "selection_reason": "best_available_data"
            },
            "rate_info": {
                "rate_type": "TB3Y",
                "latest_rate": 3.25,
                "latest_date": "2023-12-29T00:00:00",
                "source": "BOK",
                "characteristics": {
                    "maturity_days": 1095,
                    "description": "국고채 3년",
                    "priority": 2
                }
            }
        },
        "timestamp": "2024-01-15T10:30:00+09:00"
    }

# 예시 8: 벤치마크 자동 결정 예시
def create_benchmark_determination_examples():
    """벤치마크 자동 결정 예시"""
    
    examples = {
        "KOSPI 중심 포트폴리오": {
            "holdings": [
                {"code": "005930", "quantity": 100},  # 삼성전자 (KOSPI)
                {"code": "000660", "quantity": 50},   # SK하이닉스 (KOSPI)
                {"code": "035420", "quantity": 30},   # 네이버 (KOSPI)
                {"code": "207940", "quantity": 20}    # 삼성바이오로직스 (KOSPI)
            ],
            "expected_benchmark": "KOSPI",
            "reason": "KOSPI 종목이 100%"
        },
        
        "KOSDAQ 중심 포트폴리오": {
            "holdings": [
                {"code": "035720", "quantity": 100},  # 카카오 (KOSDAQ)
                {"code": "086520", "quantity": 50},   # 에코프로 (KOSDAQ)
                {"code": "251370", "quantity": 30},   # 와이엠티 (KOSDAQ)
                {"code": "091990", "quantity": 20}    # 셀트리온헬스케어 (KOSDAQ)
            ],
            "expected_benchmark": "KOSDAQ",
            "reason": "KOSDAQ 종목이 100%"
        },
        
        "혼합 포트폴리오 (KOSPI 우세)": {
            "holdings": [
                {"code": "005930", "quantity": 100},  # 삼성전자 (KOSPI)
                {"code": "000660", "quantity": 50},   # SK하이닉스 (KOSPI)
                {"code": "035720", "quantity": 20},   # 카카오 (KOSDAQ)
                {"code": "086520", "quantity": 10}    # 에코프로 (KOSDAQ)
            ],
            "expected_benchmark": "KOSPI",
            "reason": "KOSPI 종목이 75%, KOSDAQ 종목이 25%"
        },
        
        "혼합 포트폴리오 (KOSDAQ 우세)": {
            "holdings": [
                {"code": "005930", "quantity": 30},   # 삼성전자 (KOSPI)
                {"code": "000660", "quantity": 20},   # SK하이닉스 (KOSPI)
                {"code": "035720", "quantity": 100},  # 카카오 (KOSDAQ)
                {"code": "086520", "quantity": 50}    # 에코프로 (KOSDAQ)
            ],
            "expected_benchmark": "KOSDAQ",
            "reason": "KOSPI 종목이 25%, KOSDAQ 종목이 75%"
        },
        
        "균형 포트폴리오": {
            "holdings": [
                {"code": "005930", "quantity": 50},   # 삼성전자 (KOSPI)
                {"code": "000660", "quantity": 30},   # SK하이닉스 (KOSPI)
                {"code": "035720", "quantity": 50},   # 카카오 (KOSDAQ)
                {"code": "086520", "quantity": 30}    # 에코프로 (KOSDAQ)
            ],
            "expected_benchmark": "KOSPI",
            "reason": "KOSPI와 KOSDAQ이 비슷하므로 기본값인 KOSPI 선택"
        }
    }
    
    return examples

# 예시 9: 무위험 수익률 자동 결정 예시
def create_risk_free_rate_determination_examples():
    """무위험 수익률 자동 결정 예시"""
    
    examples = {
        "단기 백테스트 (6개월)": {
            "backtest_days": 180,
            "period_classification": "short_term",
            "expected_rate_type": "TB1Y",
            "reason": "단기 백테스트이므로 1년 만기 국고채가 가장 적합"
        },
        
        "중기 백테스트 (1년)": {
            "backtest_days": 365,
            "period_classification": "medium_term", 
            "expected_rate_type": "TB1Y",
            "reason": "중기 백테스트이므로 1년 만기 국고채가 적합"
        },
        
        "장기 백테스트 (3년)": {
            "backtest_days": 1095,
            "period_classification": "long_term",
            "expected_rate_type": "TB3Y", 
            "reason": "장기 백테스트이므로 3년 만기 국고채가 적합"
        },
        
        "매우 장기 백테스트 (5년)": {
            "backtest_days": 1825,
            "period_classification": "long_term",
            "expected_rate_type": "TB5Y",
            "reason": "매우 장기 백테스트이므로 5년 만기 국고채가 가장 적합"
        }
    }
    
    return examples

if __name__ == "__main__":
    # 예시 실행
    print("=== 손절/익절 규칙 백테스트 예시 ===")
    
    examples = [
        ("베타 손절", create_beta_stop_loss_example),
        ("MDD 손절", create_mdd_stop_loss_example), 
        ("VaR 손절", create_var_stop_loss_example),
        ("단일 종목 익절", create_oneprofit_take_profit_example),
        ("복합 규칙", create_complex_rules_example)
    ]
    
    for name, func in examples:
        print(f"\n--- {name} ---")
        request = func()
        print(f"손절 규칙: {[rule.category + '=' + str(rule.value) for rule in (request.rules.stopLoss or [])]}")
        print(f"익절 규칙: {[rule.category + '=' + str(rule.value) for rule in (request.rules.takeProfit or [])]}")
    
    print(f"\n--- API 요청 예시 ---")
    api_request = create_api_request_example()
    print(f"API 요청: {api_request}")
    
    print(f"\n--- API 응답 예시 ---")
    api_response = create_api_response_example()
    print(f"결과 상태: {api_response['result_status']}")
    print(f"실행 로그 수: {len(api_response['execution_logs'])}")
    print(f"사용된 벤치마크: {api_response['benchmark_info']['benchmark_code']}")
    if api_response['execution_logs']:
        print(f"첫 번째 실행 로그: {api_response['execution_logs'][0]}")
    
    print(f"\n--- 벤치마크 자동 결정 예시 ---")
    benchmark_examples = create_benchmark_determination_examples()
    for name, example in benchmark_examples.items():
        print(f"\n{name}:")
        print(f"  구성: {len(example['holdings'])}개 종목")
        print(f"  예상 벤치마크: {example['expected_benchmark']}")
        print(f"  이유: {example['reason']}")
    
    print(f"\n--- 무위험 수익률 자동 결정 예시 ---")
    risk_free_examples = create_risk_free_rate_determination_examples()
    for name, example in risk_free_examples.items():
        print(f"\n{name}:")
        print(f"  백테스트 기간: {example['backtest_days']}일")
        print(f"  기간 분류: {example['period_classification']}")
        print(f"  예상 금리 유형: {example['expected_rate_type']}")
        print(f"  이유: {example['reason']}")
