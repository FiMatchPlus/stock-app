"""FastAPI 주가 데이터 수집 서버 메인 애플리케이션"""

from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time

from app.config import settings
from app.end import stock_router, backtest_router, analysis_router
from app.services.parallel_processing_service import ParallelProcessingService
from app.services.scheduler_service import scheduler_service
from app.utils.redis_client import close_redis_client
from app.utils.mongodb_client import mongodb_client, MONGODB_URL, MONGODB_DATABASE
from app.utils.logger import get_logger, log_api_request

logger = get_logger(__name__)

# 전역 병렬처리 서비스 인스턴스
parallel_service = ParallelProcessingService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시 실행
    logger.info("Starting Server", version=settings.app_version)
    
    # MongoDB 연결
    try:
        await mongodb_client.connect(MONGODB_URL, MONGODB_DATABASE)
        logger.info("MongoDB connected successfully")
    except Exception as e:
        logger.error("Failed to connect to MongoDB", error=str(e))
        # MongoDB 연결 실패해도 서버는 계속 실행
    
    # 스케줄러 시작
    try:
        scheduler_service.start()
        logger.info("Scheduler service started successfully")
    except Exception as e:
        logger.error("Failed to start scheduler service", error=str(e))
        raise
    
    
    yield
    
    # 종료 시 실행
    logger.info("Shutting down Server")
    
    # 스케줄러 중지
    try:
        scheduler_service.stop()
        logger.info("Scheduler service stopped")
    except Exception as e:
        logger.error("Failed to stop scheduler service", error=str(e))
    
    # 병렬처리 서비스 정리
    try:
        await parallel_service.cleanup()
        logger.info("Parallel processing service cleaned up")
    except Exception as e:
        logger.error("Failed to cleanup parallel processing service", error=str(e))
    
    # MongoDB 연결 해제
    try:
        await mongodb_client.disconnect()
        logger.info("MongoDB client closed")
    except Exception as e:
        logger.error("Failed to close MongoDB client", error=str(e))
    
    # Redis 연결 해제
    try:
        await close_redis_client()
        logger.info("Redis client closed")
    except Exception as e:
        logger.error("Failed to close Redis client", error=str(e))


# FastAPI 애플리케이션 생성
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="고성능 주가 데이터 수집 및 실시간 스트리밍 서버",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# CORS Preflight 요청 처리 미들웨어 (CORS 미들웨어보다 먼저 실행)
@app.middleware("http")
async def handle_cors_preflight(request: Request, call_next):
    """CORS preflight 요청을 처리하는 미들웨어"""
    if request.method == "OPTIONS":
        # Origin 헤더 확인
        origin = request.headers.get("origin", "*")
        
        return JSONResponse(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Accept, Origin",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Max-Age": "86400",  # 24시간
            }
        )
    
    response = await call_next(request)
    return response


# API 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """모든 API 요청을 로깅하는 미들웨어"""
    start_time = time.time()
    
    # 요청 로깅
    log_api_request(
        logger,
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        request_id=request.headers.get("x-request-id")
    )
    
    # 요청 처리
    response = await call_next(request)
    
    # 에러 응답에도 CORS 헤더 추가 (예외 처리기에서 누락된 경우 대비)
    if response.status_code >= 400:
        origin = request.headers.get("origin", "*")
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    
    # 응답 로깅
    process_time = time.time() - start_time
    logger.info(
        "API Response",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=f"{process_time:.3f}s",
        request_id=request.headers.get("x-request-id")
    )
    
    return response

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# HTTPException 처리기 (FastAPI 기본 처리기를 우선 사용)
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTPException 처리 - CORS 헤더 추가"""
    origin = request.headers.get("origin", "*")
    
    response = JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
    
    # CORS 헤더 추가
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response


# 전역 예외 처리기 (HTTPException은 제외)
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 예외 처리 - HTTPException은 제외"""
    # HTTPException은 이미 위에서 처리되므로 여기서는 제외
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        method=request.method
    )
    
    origin = request.headers.get("origin", "*")
    
    response = JSONResponse(
        status_code=500,
        content={
            "detail": str(exc) if settings.debug else "서버 내부 오류가 발생했습니다."
        }
    )
    
    # CORS 헤더 추가
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response


# 라우터 등록
app.include_router(stock_router)
app.include_router(backtest_router)
app.include_router(analysis_router)


# 헬스 체크 엔드포인트
@app.get("/health", tags=["health"])
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.utcnow().isoformat()
    }


# 루트 엔드포인트
@app.get("/", tags=["root"])
async def root():
    """루트 엔드포인트"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs_url": "/docs" if settings.debug else "Documentation not available in production",
        "health_check": "/health"
    }


# 병렬처리 관련 엔드포인트
@app.post("/aggregate/calculate", tags=["aggregate"])
async def calculate_aggregates(data: dict):
    """집계 연산 실행 (테스트용)"""
    try:
        # 실제 구현에서는 데이터베이스에서 데이터를 가져와야 함
        stock_data = data.get("stock_data", [])
        
        if not stock_data:
            raise HTTPException(status_code=400, detail="No stock data provided")
        
        # 병렬처리로 집계 연산 실행
        results = await parallel_service.calculate_stock_aggregates(stock_data)
        
        return {
            "message": "Aggregates calculated successfully",
            "count": len(results),
            "results": [result.model_dump(mode='json') for result in results]
        }
    
    except Exception as e:
        logger.error("Failed to calculate aggregates", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/indicators/calculate", tags=["indicators"])
async def calculate_technical_indicators(data: dict):
    """기술적 지표 계산 (테스트용)"""
    try:
        stock_data = data.get("stock_data", [])
        
        if not stock_data:
            raise HTTPException(status_code=400, detail="No stock data provided")
        
        # 병렬처리로 기술적 지표 계산
        results = await parallel_service.calculate_technical_indicators(stock_data)
        
        return {
            "message": "Technical indicators calculated successfully",
            "count": len(results),
            "results": results
        }
    
    except Exception as e:
        logger.error("Failed to calculate technical indicators", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/correlation/calculate", tags=["correlation"])
async def calculate_correlation_matrix(data: dict):
    """상관관계 행렬 계산 (테스트용)"""
    try:
        stock_data = data.get("stock_data", [])
        
        if not stock_data:
            raise HTTPException(status_code=400, detail="No stock data provided")
        
        # 병렬처리로 상관관계 행렬 계산
        results = await parallel_service.calculate_correlation_matrix(stock_data)
        
        return {
            "message": "Correlation matrix calculated successfully",
            "symbols": list(results.keys()),
            "matrix": results
        }
    
    except Exception as e:
        logger.error("Failed to calculate correlation matrix", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# 스케줄러 관련 엔드포인트
@app.get("/scheduler/status", tags=["scheduler"])
async def get_scheduler_status():
    """스케줄러 상태 조회"""
    try:
        status = scheduler_service.get_scheduler_status()
        return {
            "success": True,
            "data": status
        }
    except Exception as e:
        logger.error("Failed to get scheduler status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scheduler/crawl", tags=["scheduler"])
async def run_manual_crawling(stock_codes: list[str] = None):
    """수동 크롤링 실행"""
    try:
        result = await scheduler_service.run_manual_crawling(stock_codes)
        return result
    except Exception as e:
        logger.error("Failed to run manual crawling", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scheduler/start", tags=["scheduler"])
async def start_scheduler():
    """스케줄러 시작"""
    try:
        scheduler_service.start()
        return {
            "success": True,
            "message": "Scheduler started successfully"
        }
    except Exception as e:
        logger.error("Failed to start scheduler", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scheduler/stop", tags=["scheduler"])
async def stop_scheduler():
    """스케줄러 중지"""
    try:
        scheduler_service.stop()
        return {
            "success": True,
            "message": "Scheduler stopped successfully"
        }
    except Exception as e:
        logger.error("Failed to stop scheduler", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )
