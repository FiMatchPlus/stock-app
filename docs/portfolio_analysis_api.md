# 포트폴리오 상세 분석 API 문서

## 개요

FiMatchPlus 포트폴리오 분석 API는 **이동 윈도우(Moving Window)** 기반의 MPT(현대 포트폴리오 이론) 최적화 및 백테스팅 분석을 제공합니다. 3년 윈도우 크기로 1개월 간격으로 이동하며 최적화를 수행하고, 백테스팅을 통해 검증된 성능 지표를 제공합니다.

## 주요 특징

- **이동 윈도우 최적화**: 3년 윈도우 크기로 1개월 간격으로 이동하며 포트폴리오 최적화 수행
- **백테스팅 검증**: 전체 기간에 대한 백테스팅을 통해 성능 지표의 신뢰성 확보
- **현실적 비중**: 최근 시점의 최적 비중을 제공하여 실제 투자에 활용 가능
- **종합적 분석**: 기본 지표부터 고급 리스크 지표까지 포괄적인 성과 분석

## API 엔드포인트

### POST /analysis/start (비동기)

비동기 포트폴리오 분석을 시작합니다.

#### 요청 (Request)

**URL**: `POST /analysis/start`

**Content-Type**: `application/json`

**요청 스키마**:

```json
{
  "holdings": [
    {
      "stock_code": "string (6자리 숫자)",
      "quantity": "integer (최소 1)"
    }
  ],
  "lookback_years": "integer (1-10, 기본값: 3)",
  "benchmark": "string (선택사항, 예: KOSPI)",
  "risk_free_rate": "number (선택사항, 0-1)",
  "callback_url": "string (필수, 비동기 처리 시)",
  "analysis_id": "integer (선택사항, 클라이언트 ID)"
}
```

**요청 예시**:

```json
{
  "holdings": [
    {"stock_code": "005930", "quantity": 100},
    {"stock_code": "000660", "quantity": 50},
    {"stock_code": "035420", "quantity": 30}
  ],
  "lookback_years": 5,
  "benchmark": "KOSPI",
  "risk_free_rate": 0.035,
  "callback_url": "https://your-server.com/analysis/callback",
  "analysis_id": 12345
}
```

#### 응답 (Response)

**성공 응답 (200 OK)**:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "started",
  "message": "포트폴리오 분석이 백그라운드에서 실행 중입니다."
}
```

**에러 응답 (400 Bad Request)**:

```json
{
  "error": "callback_url is required for async analysis",
  "detail": "비동기 분석을 위해서는 콜백 URL이 필요합니다.",
  "timestamp": "2024-01-01T10:00:00Z"
}
```

#### 콜백 응답

분석이 완료되면 지정된 콜백 URL로 결과가 전송됩니다.

**성공 콜백**:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "success": true,
  "min_variance": {
    "weights": {
      "005930": 0.45,
      "000660": 0.35,
      "035420": 0.20
    }
  },
  "max_sharpe": {
    "weights": {
      "005930": 0.50,
      "000660": 0.30,
      "035420": 0.20
    }
  },
  "metrics": {
    "min_variance": {
      "expected_return": 0.085,
      "sharpe_ratio": 0.33,
      "max_drawdown": -0.22,
      "var_value": -0.15,
      "cvar_value": -0.19
      // ... 기타 백테스팅 기반 지표들
    },
    "max_sharpe": {
      "expected_return": 0.092,
      "sharpe_ratio": 0.32,
      "max_drawdown": -0.25,
      "var_value": -0.18,
      "cvar_value": -0.23
      // ... 기타 백테스팅 기반 지표들
    }
  },
  "execution_time": 2.5,
  "analysis_id": 12345,
  "timestamp": "2024-01-01T10:02:30Z"
}
```

**실패 콜백**:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "success": false,
  "error": {
    "error": "Analysis failed",
    "detail": "Insufficient data for requested period",
    "timestamp": "2024-01-01T10:01:15Z"
  },
  "execution_time": 1.2,
  "analysis_id": 12345,
  "timestamp": "2024-01-01T10:01:15Z"
}
```

### POST /analysis/run (동기)

포트폴리오 분석을 즉시 실행하고 결과를 반환합니다.

#### 요청 (Request)

**URL**: `POST /analysis/run`

**Content-Type**: `application/json`

**요청 스키마**:

```json
{
  "holdings": [
    {
      "stock_code": "string (6자리 숫자)",
      "quantity": "integer (최소 1)"
    }
  ],
  "lookback_years": "integer (1-10, 기본값: 3)",
  "benchmark": "string (선택사항, 예: KOSPI)",
  "risk_free_rate": "number (선택사항, 0-1)"
}
```

**필드 설명**:

| 필드명 | 타입 | 필수 | 설명 | 예시 |
|--------|------|------|------|------|
| `holdings` | Array | ✅ | 보유 종목 목록 (최소 1개 이상) | - |
| `holdings[].stock_code` | String | ✅ | 종목코드 (6자리 숫자) | "005930" |
| `holdings[].quantity` | Integer | ✅ | 보유 수량 (주) | 100 |
| `lookback_years` | Integer | ❌ | 과거 데이터 조회 연수 (3~5 추천, 최소 5년 권장) | 5 |
| `benchmark` | String | ❌ | 벤치마크 지수 코드 (미제공 시 KOSPI 자동 사용) | "KOSPI" |
| `risk_free_rate` | Number | ❌ | 연간 무위험 수익률 (미제공 시 자동 조회) | 0.035 |

**요청 예시**:

```json
{
  "holdings": [
    {
      "stock_code": "005930",
      "quantity": 100
    },
    {
      "stock_code": "000660",
      "quantity": 50
    },
    {
      "stock_code": "035420",
      "quantity": 30
    }
  ],
  "lookback_years": 5,
  "benchmark": "KOSPI",
  "risk_free_rate": 0.035
}
```

#### 응답 (Response)

**성공 응답 (200 OK)**:

```json
{
  "success": true,
  "min_variance": {
    "weights": {
      "005930": 0.45,
      "000660": 0.35,
      "035420": 0.20
    }
  },
  "max_sharpe": {
    "weights": {
      "005930": 0.50,
      "000660": 0.30,
      "035420": 0.20
    }
  },
  "metrics": {
    "min_variance": {
      "expected_return": 0.085,
      "std_deviation": 0.15,
      "beta": 0.95,
      "alpha": 0.012,
      "jensen_alpha": 0.008,
      "tracking_error": 0.08,
      "sharpe_ratio": 0.33,
      "treynor_ratio": 0.053,
      "sortino_ratio": 0.42,
      "calmar_ratio": 0.28,
      "information_ratio": 0.15,
      "max_drawdown": -0.22,
      "downside_deviation": 0.12,
      "upside_beta": 1.05,
      "downside_beta": 0.85,
      "correlation_with_benchmark": 0.78
    },
    "max_sharpe": {
      "expected_return": 0.092,
      "std_deviation": 0.18,
      "beta": 1.02,
      "alpha": 0.018,
      "jensen_alpha": 0.014,
      "tracking_error": 0.10,
      "sharpe_ratio": 0.32,
      "treynor_ratio": 0.056,
      "sortino_ratio": 0.48,
      "calmar_ratio": 0.32,
      "information_ratio": 0.18,
      "max_drawdown": -0.25,
      "downside_deviation": 0.14,
      "upside_beta": 1.08,
      "downside_beta": 0.92,
      "correlation_with_benchmark": 0.82
    }
  },
  "benchmark_comparison": {
    "benchmark_code": "KOSPI",
    "benchmark_return": 0.065,
    "benchmark_volatility": 0.16,
    "excess_return": 0.027,
    "relative_volatility": 1.125,
    "security_selection": 0.019,
    "timing_effect": 0.008
  },
  "risk_free_rate_used": 0.035,
  "analysis_period": {
    "start": "2019-01-01T00:00:00Z",
    "end": "2024-01-01T00:00:00Z"
  },
  "notes": "Analysis based on KOSPI benchmark and 3-year rolling window optimization."
}
```

**응답 필드 설명**:

| 필드명 | 타입 | 설명 |
|--------|------|------|
| `success` | Boolean | 분석 성공 여부 |
| `min_variance` | Object | 최소 분산 포트폴리오 비중 (최근 시점 기준) |
| `min_variance.weights` | Object | 종목코드별 비중 (합계 1.0) |
| `max_sharpe` | Object | 최대 샤프 비율 포트폴리오 비중 (최근 시점 기준) |
| `max_sharpe.weights` | Object | 종목코드별 비중 (합계 1.0) |
| `metrics` | Object | 백테스팅 기반 평균 성능 지표 |
| `metrics.min_variance` | Object | 최소 분산 포트폴리오의 백테스팅 성능 지표 |
| `metrics.max_sharpe` | Object | 최대 샤프 포트폴리오의 백테스팅 성능 지표 |
| `benchmark_comparison` | Object | 벤치마크 비교 분석 결과 (벤치마크 제공 시) |
| `risk_free_rate_used` | Number | 분석에 사용된 무위험수익률 |
| `analysis_period` | Object | 분석 기간 (백테스팅 전체 기간) |
| `analysis_period.start` | String | 분석 시작일 (ISO 8601 형식) |
| `analysis_period.end` | String | 분석 종료일 (ISO 8601 형식) |
| `notes` | String | 참고 사항 |

#### 성능 지표 상세 설명

**기본 지표**:
- `expected_return`: 기대수익률 (연환산)
- `std_deviation`: 표준편차 (연환산)

**벤치마크 대비 지표**:
- `beta`: 베타 (벤치마크 대비)
- `alpha`: 알파 (CAPM 기준)
- `jensen_alpha`: 젠센 알파
- `tracking_error`: 트래킹 에러

**위험조정 수익률 지표**:
- `sharpe_ratio`: 샤프 비율
- `treynor_ratio`: 트레이너 비율
- `sortino_ratio`: 소르티노 비율
- `calmar_ratio`: 칼마 비율
- `information_ratio`: 정보비율

**리스크 지표**:
- `max_drawdown`: 최대 낙폭
- `downside_deviation`: 하방편차
- `upside_beta`: 상승 베타
- `downside_beta`: 하락 베타
- `var_value`: VaR 95% (Value at Risk)
- `cvar_value`: CVaR 95% (Conditional Value at Risk)

**벤치마크 상관관계**:
- `correlation_with_benchmark`: 벤치마크와의 상관관계

#### 에러 응답

**400 Bad Request**:
```json
{
  "error": "At least one holding must be specified",
  "detail": "포트폴리오에 최소 1개 이상의 종목이 필요합니다.",
  "timestamp": "2024-01-01T10:00:00Z"
}
```

**500 Internal Server Error**:
```json
{
  "error": "Internal server error",
  "detail": "분석 중 오류가 발생했습니다.",
  "timestamp": "2024-01-01T10:00:00Z"
}
```

## 분석 방법론

### 이동 윈도우 최적화

1. **윈도우 크기**: 3년 (36개월)
2. **이동 간격**: 1개월
3. **최적화 방법**:
   - 최소 분산 포트폴리오: 포트폴리오 분산 최소화
   - 최대 샤프 포트폴리오: 샤프 비율 최대화

### 백테스팅 기반 성능 지표

1. **백테스팅 수행**: 각 시점의 최적 비중을 다음 기간의 실제 수익률에 적용
2. **성능 지표 계산**: 전체 백테스팅 기간에 대한 평균/총합으로 계산
3. **검증된 지표**: 모델의 장기적인 일반 성능을 대변

### 응답 로직

- **최근 비중**: 가장 최근 시점의 3년 윈도우로 계산된 비중 제공
- **백테스팅 지표**: 백테스팅 기간 전체의 성과 지표 평균 제공
- **현실적 추천**: 실제 투자에 활용 가능한 검증된 비중과 성능 지표

## 사용 예시

### 동기 분석 (cURL)

```bash
curl -X POST "http://localhost:8000/analysis/run" \
  -H "Content-Type: application/json" \
  -d '{
    "holdings": [
      {"stock_code": "005930", "quantity": 100},
      {"stock_code": "000660", "quantity": 50},
      {"stock_code": "035420", "quantity": 30}
    ],
    "lookback_years": 5,
    "benchmark": "KOSPI",
    "risk_free_rate": 0.035
  }'
```

### 비동기 분석 (cURL)

```bash
curl -X POST "http://localhost:8000/analysis/start" \
  -H "Content-Type: application/json" \
  -d '{
    "holdings": [
      {"stock_code": "005930", "quantity": 100},
      {"stock_code": "000660", "quantity": 50},
      {"stock_code": "035420", "quantity": 30}
    ],
    "lookback_years": 5,
    "benchmark": "KOSPI",
    "risk_free_rate": 0.035,
    "callback_url": "https://your-server.com/analysis/callback",
    "analysis_id": 12345
  }'
```

### Python 예시

#### 동기 분석

```python
import requests
import json

# 동기 분석
url = "http://localhost:8000/analysis/run"
data = {
    "holdings": [
        {"stock_code": "005930", "quantity": 100},
        {"stock_code": "000660", "quantity": 50},
        {"stock_code": "035420", "quantity": 30}
    ],
    "lookback_years": 5,
    "benchmark": "KOSPI",
    "risk_free_rate": 0.035
}

response = requests.post(url, json=data)
result = response.json()

print(f"최소 분산 포트폴리오 비중: {result['min_variance']['weights']}")
print(f"최대 샤프 포트폴리오 비중: {result['max_sharpe']['weights']}")
print(f"샤프 비율: {result['metrics']['max_sharpe']['sharpe_ratio']}")
```

#### 비동기 분석

```python
import requests
import json
from flask import Flask, request, jsonify

# 비동기 분석 시작
url = "http://localhost:8000/analysis/start"
data = {
    "holdings": [
        {"stock_code": "005930", "quantity": 100},
        {"stock_code": "000660", "quantity": 50},
        {"stock_code": "035420", "quantity": 30}
    ],
    "lookback_years": 5,
    "benchmark": "KOSPI",
    "risk_free_rate": 0.035,
    "callback_url": "https://your-server.com/analysis/callback",
    "analysis_id": 12345
}

response = requests.post(url, json=data)
job_info = response.json()

print(f"작업 ID: {job_info['job_id']}")
print(f"상태: {job_info['status']}")

# 콜백 서버 (Flask 예시)
app = Flask(__name__)

@app.route('/analysis/callback', methods=['POST'])
def handle_analysis_callback():
    callback_data = request.json
    
    if callback_data['success']:
        print("분석 완료!")
        print(f"최소 분산 포트폴리오 비중: {callback_data['min_variance']['weights']}")
        print(f"최대 샤프 포트폴리오 비중: {callback_data['max_sharpe']['weights']}")
        print(f"샤프 비율: {callback_data['metrics']['max_sharpe']['sharpe_ratio']}")
        print(f"실행 시간: {callback_data['execution_time']}초")
    else:
        print(f"분석 실패: {callback_data['error']['detail']}")
    
    return jsonify({"status": "received"}), 200

if __name__ == '__main__':
    app.run(port=5000)
```

## 주의사항

1. **데이터 요구사항**: 최소 5년 이상의 데이터가 필요합니다.
2. **윈도우 크기**: 3년 윈도우 크기는 시장 변화를 반영하면서도 안정적인 추정을 위한 최적값입니다.
3. **백테스팅 결과**: 제공되는 성능 지표는 과거 데이터 기반 백테스팅 결과이며, 미래 성과를 보장하지 않습니다.
4. **비중 활용**: 최근 시점의 비중은 현재 시장 상황을 반영하므로 투자 결정 시 참고자료로 활용하세요.
