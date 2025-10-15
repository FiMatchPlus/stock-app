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
from app.services.prices.scheduler_service import scheduler_service
from app.utils.logger import get_logger, log_api_request

logger = get_logger(__name__)

parallel_service = ParallelProcessingService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    logger.info("서버 시작", version=settings.app_version)
    
    try:
        scheduler_service.start()
        logger.info("스케줄러 서비스 시작 성공")
    except Exception as e:
        logger.error("스케줄러 서비스 시작 실패", error=str(e))
        raise
    
    
    yield
    
    logger.info("서버 종료")
    
    try:
        scheduler_service.stop()
        logger.info("스케줄러 서비스 중지됨")
    except Exception as e:
        logger.error("스케줄러 서비스 중지 실패", error=str(e))
    
    try:
        await parallel_service.cleanup()
        logger.info("병렬 처리 서비스 정리 완료")
    except Exception as e:
        logger.error("병렬 처리 서비스 정리 실패", error=str(e))


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="고성능 주가 데이터 수집 및 실시간 스트리밍 서버",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

@app.middleware("http")
async def handle_cors_preflight(request: Request, call_next):
    """CORS preflight 요청을 처리하는 미들웨어"""
    if request.method == "OPTIONS":
        origin = request.headers.get("origin", "*")
        
        return JSONResponse(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Accept, Origin",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Max-Age": "86400",
            }
        )
    
    response = await call_next(request)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """API 요청을 로깅 미들웨어"""
    start_time = time.time()
    
    log_api_request(
        logger,
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        request_id=request.headers.get("x-request-id")
    )
    
    response = await call_next(request)
    
    if response.status_code >= 400:
        origin = request.headers.get("origin", "*")
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    
    process_time = time.time() - start_time
    logger.info(
        "API 응답",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=f"{process_time:.3f}s",
        request_id=request.headers.get("x-request-id")
    )
    
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTPException 처리 - CORS 헤더 추가"""
    origin = request.headers.get("origin", "*")
    
    response = JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
    
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 예외 처리 - HTTPException은 제외"""
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    
    logger.error(
        "처리되지 않은 예외",
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
    
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response


app.include_router(stock_router)
app.include_router(backtest_router)
app.include_router(analysis_router)


@app.get("/health", tags=["health"])
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/", tags=["root"])
async def root():
    """루트 엔드포인트"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs_url": "/docs" if settings.debug else "Documentation not available in production",
        "health_check": "/health"
    }


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
        logger.error("스케줄러 상태 조회 실패", error=str(e))
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
        logger.error("스케줄러 시작 실패", error=str(e))
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
        logger.error("스케줄러 중지 실패", error=str(e))
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
