# 시계열 임베딩 모델 API

주가 데이터를 이용한 시계열 임베딩 모델을 생성하고 Elasticsearch에 저장하는 API입니다.

## 주요 기능

### 1. 시계열 임베딩 모델
- **AutoEncoder 기반**: 시계열 데이터를 저차원 벡터로 압축
- **LSTM 기반**: 순환 신경망을 이용한 시계열 패턴 학습
- **금융 피처 통합**: 변동성, 베타, 섹터 등 추가 피처 포함

### 2. 금융 피처 추출
- **변동성**: 연환산 변동성 계산
- **베타**: KOSPI 대비 베타 계산
- **섹터**: 산업 분류 정보
- **기술적 지표**: RSI, MACD, 볼린저 밴드 등

### 3. Elasticsearch 저장 및 검색
- **벡터 검색**: 코사인 유사도 기반 유사 종목 검색
- **필터 검색**: 섹터, 변동성, 베타 등으로 필터링
- **통계 조회**: 임베딩 데이터 통계 정보

## API 엔드포인트

### 임베딩 생성 및 저장

#### 1. 임베딩 생성
```http
POST /embeddings/generate
Content-Type: application/json

{
  "symbols": ["005930", "000660", "035420"],
  "sequence_length": 60,
  "embedding_dim": 64,
  "model_type": "autoencoder",
  "include_features": true
}
```

#### 2. 임베딩 생성 및 저장
```http
POST /embeddings/generate-and-store
Content-Type: application/json

{
  "symbols": ["005930", "000660", "035420"],
  "sequence_length": 60,
  "embedding_dim": 64,
  "model_type": "autoencoder",
  "include_features": true
}
```

### 임베딩 검색

#### 1. 유사 임베딩 검색
```http
POST /embeddings/search
Content-Type: application/json

{
  "query_embedding": [0.1, 0.2, 0.3, ...],
  "top_k": 10,
  "filters": {
    "sector": "반도체",
    "volatility_range": [0.1, 0.5]
  }
}
```

#### 2. 특정 종목 임베딩 조회
```http
GET /embeddings/symbol/005930
```

#### 3. 필터 기반 검색
```http
GET /embeddings/search-by-filters?sector=반도체&volatility_min=0.1&volatility_max=0.5&size=50
```

### 임베딩 관리

#### 1. 임베딩 업데이트
```http
PUT /embeddings/symbol/005930?sequence_length=60&embedding_dim=64&model_type=autoencoder&include_features=true
```

#### 2. 임베딩 삭제
```http
DELETE /embeddings/symbol/005930
```

#### 3. 배치 업데이트
```http
POST /embeddings/batch-update
Content-Type: application/json

{
  "symbols": ["005930", "000660", "035420"],
  "sequence_length": 60,
  "embedding_dim": 64,
  "model_type": "autoencoder",
  "include_features": true
}
```

### 모델 훈련

#### 1. 모델 훈련
```http
POST /embeddings/train
Content-Type: application/json

{
  "symbols": ["005930", "000660", "035420", "207940", "006400"],
  "sequence_length": 60,
  "embedding_dim": 64,
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "validation_split": 0.2
}
```

### 통계 및 헬스 체크

#### 1. 임베딩 통계
```http
GET /embeddings/statistics
```

#### 2. 서비스 헬스 체크
```http
GET /embeddings/health
```

## 환경 설정

### 1. 환경 변수 설정
```bash
# Elasticsearch 설정
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_USERNAME=
ELASTICSEARCH_PASSWORD=

# 기존 설정들...
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/stockdb
REDIS_URL=redis://localhost:6379
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=stockone19
```

### 2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. Elasticsearch 실행
```bash
# Docker로 Elasticsearch 실행
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.11.0
```

## 데이터 구조

### 임베딩 문서 (Elasticsearch)
```json
{
  "symbol": "005930",
  "embedding": [0.1, 0.2, 0.3, ...],
  "features": {
    "symbol": "005930",
    "volatility": 0.25,
    "beta": 1.2,
    "sector": "반도체",
    "market_cap": 500000000000,
    "pe_ratio": 15.5,
    "pb_ratio": 2.1,
    "dividend_yield": 0.02
  },
  "model_info": {
    "model_id": "uuid-string",
    "embedding_dim": 64,
    "sequence_length": 60,
    "model_type": "autoencoder"
  },
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

## 사용 예시

### 1. 기본 임베딩 생성 및 저장
```python
import requests

# 임베딩 생성 및 저장
response = requests.post(
    "http://localhost:8000/embeddings/generate-and-store",
    json={
        "symbols": ["005930", "000660", "035420"],
        "sequence_length": 60,
        "embedding_dim": 64,
        "model_type": "autoencoder",
        "include_features": True
    }
)

print(response.json())
```

### 2. 유사 종목 검색
```python
# 특정 종목의 임베딩 조회
response = requests.get("http://localhost:8000/embeddings/symbol/005930")
target_embedding = response.json()["embedding"]

# 유사 종목 검색
search_response = requests.post(
    "http://localhost:8000/embeddings/search",
    json={
        "query_embedding": target_embedding,
        "top_k": 10,
        "filters": {
            "sector": "반도체"
        }
    }
)

similar_stocks = search_response.json()
for stock in similar_stocks:
    print(f"{stock['symbol']}: {stock['score']:.4f}")
```

### 3. 필터 기반 검색
```python
# 반도체 섹터에서 변동성이 높은 종목들 검색
response = requests.get(
    "http://localhost:8000/embeddings/search-by-filters",
    params={
        "sector": "반도체",
        "volatility_min": 0.3,
        "size": 20
    }
)

stocks = response.json()
for stock in stocks:
    print(f"{stock['symbol']}: {stock['features']['volatility']:.3f}")
```

## 주의사항

1. **데이터베이스 수정 금지**: DDL 관련 작업은 직접 수행하지 않습니다.
2. **Elasticsearch 인덱스**: 자동으로 생성되며, 임베딩 차원에 따라 동적으로 조정됩니다.
3. **모델 훈련**: 충분한 데이터(최소 10개 종목)가 필요합니다.
4. **메모리 사용량**: 대량의 임베딩 생성 시 메모리 사용량을 고려하세요.

## 추가 데이터 요구사항

현재 구현에서 추가로 필요한 데이터:

1. **KOSPI 지수 데이터**: 베타 계산을 위해 필요
2. **재무제표 데이터**: PER, PBR, 배당수익률 계산을 위해 필요
3. **발행주식수 데이터**: 시가총액 계산을 위해 필요

이러한 데이터가 추가되면 더 정확한 금융 피처를 추출할 수 있습니다.
