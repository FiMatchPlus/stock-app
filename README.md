# Portfolio Analysis & Backtest Engine

포트폴리오 최적화 및 백테스트 서비스

## 프로젝트 구조

```
model_test_v1/
│
├── app/                                # 메인 애플리케이션
│   ├── main.py                        # FastAPI 애플리케이션 진입점
│   ├── run.py                         # 서버 실행 스크립트
│   │
│   ├── config/                        # 설정 관리
│   │   ├── __init__.py
│   │   └── settings.py               # 환경변수 및 앱 설정
│   │
│   ├── end/                           # API 엔드포인트 (Router)
│   │   ├── __init__.py
│   │   ├── analysis_router.py        # 포트폴리오 분석 API
│   │   ├── backtest_router.py        # 백테스트 API
│   │   └── stock_router.py           # 주가 데이터 수집 API
│   │
│   ├── models/                        # 데이터 모델 및 스키마
│   │   ├── __init__.py
│   │   ├── database.py               # 데이터베이스 연결 설정
│   │   ├── stock.py                  # SQLAlchemy ORM 모델
│   │   ├── schemas.py                # 공통 Pydantic 스키마
│   │   ├── schema_common.py          # 공통 스키마 정의
│   │   ├── schema_stock.py           # 주가 관련 스키마
│   │   ├── schema_analysis.py        # 분석 관련 스키마
│   │   └── schema_backtest.py        # 백테스트 관련 스키마
│   │
│   ├── repositories/                  # 데이터 접근 계층 (Repository Pattern)
│   │   ├── __init__.py
│   │   ├── base.py                   # 기본 Repository 클래스
│   │   ├── stock_repository.py       # 종목 데이터 Repository
│   │   ├── stock_price_repository.py # 주가 데이터 Repository
│   │   ├── benchmark_repository.py   # 벤치마크 데이터 Repository
│   │   └── risk_free_rate_repository.py  # 무위험수익률 Repository
│   │
│   ├── services/                      # 비즈니스 로직 계층
│   │   │
│   │   ├── analysis_service.py       # 포트폴리오 분석 메인 서비스
│   │   ├── analysis/                 # 포트폴리오 분석 세부 서비스
│   │   │   ├── __init__.py
│   │   │   ├── benchmark_service.py  # 벤치마크 비교 서비스
│   │   │   ├── beta_service.py       # 베타 계산 서비스
│   │   │   ├── compose_service.py    # 결과 조합 서비스
│   │   │   ├── data_service.py       # 데이터 준비 서비스
│   │   │   ├── metrics_service.py    # 성과 지표 계산 서비스
│   │   │   └── optimization_service.py  # 포트폴리오 최적화 서비스
│   │   │
│   │   ├── backtest_service.py       # 백테스트 메인 서비스
│   │   ├── backtest/                 # 백테스트 세부 서비스
│   │   │   ├── __init__.py
│   │   │   ├── analysis_service.py   # 백테스트 분석 서비스
│   │   │   ├── benchmark_service.py  # 벤치마크 비교 서비스
│   │   │   ├── calculation_service.py # 수익률 계산 서비스
│   │   │   ├── data_service.py       # 데이터 준비 서비스
│   │   │   ├── execution_service.py  # 백테스트 실행 서비스
│   │   │   ├── metrics_service.py    # 성과 지표 계산 서비스
│   │   │   ├── risk_free_rate_service.py  # 무위험수익률 서비스
│   │   │   └── trading_rules_service.py   # 거래 규칙 서비스
│   │   │
│   │   ├── prices/                   # 주가 데이터 관리 서비스
│   │   │   ├── __init__.py
│   │   │   ├── external_api_service.py    # 외부 API 서비스
│   │   │   ├── naver_crawling_service.py  # 네이버 금융 크롤링 서비스
│   │   │   ├── scheduler_service.py       # 스케줄러 서비스 (매일 자동 수집)
│   │   │   ├── stock_price_service.py     # 주가 데이터 서비스
│   │   │   └── stock_service.py           # 종목 마스터 서비스
│   │   │
│   │   ├── benchmark_service.py      # 벤치마크 자동 결정 서비스
│   │   ├── parallel_processing_service.py  # 병렬 처리 서비스
│   │   └── risk_free_rate_calculator.py    # 무위험수익률 계산 서비스
│   │
│   ├── exceptions/                    # 커스텀 예외
│   │   ├── __init__.py
│   │   └── backtest_exceptions.py    # 백테스트 관련 예외
│   │
│   └── utils/                         # 유틸리티
│       ├── __init__.py
│       └── logger.py                 # 구조화 로깅 설정
│
├── requirements.txt                   # Python 패키지 의존성
├── Dockerfile                        # Docker 이미지 빌드 설정
└── README.md                         # 프로젝트 문서
```

## 아키텍처 계층

### 1. **API Layer** (`app/end/`)
- FastAPI 라우터를 통한 RESTful API 제공
- 요청 검증 및 응답 직렬화

### 2. **Service Layer** (`app/services/`)
- 비즈니스 로직 구현
- 도메인별 서비스 분리 (분석, 백테스트, 주가 관리)
- 모듈화된 세부 서비스 구조

### 3. **Repository Layer** (`app/repositories/`)
- 데이터베이스 접근 추상화
- Repository 패턴 적용
- 재사용 가능한 데이터 접근 메서드

### 4. **Model Layer** (`app/models/`)
- SQLAlchemy ORM 모델 (데이터베이스 테이블 정의)
- Pydantic 스키마 (API 요청/응답 검증)


## 기술 스택

- **Framework**: FastAPI 0.104.1
- **Database**: PostgreSQL (with SQLAlchemy 2.0.23, asyncpg)
- **Data Processing**: pandas, numpy, scipy
- **Scheduler**: APScheduler 3.10.4
- **Web Scraping**: BeautifulSoup4 4.12.2
- **Logging**: structlog 23.2.0
- **Server**: Uvicorn 0.24.0

