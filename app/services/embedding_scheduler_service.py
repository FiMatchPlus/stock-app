"""임베딩 자동 업데이트 스케줄러 서비스"""

from datetime import datetime, timedelta, time
from typing import List, Dict, Any
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.services.embedding_storage_service import EmbeddingStorageService
from app.repositories.stock_repository import StockRepository
from app.repositories.stock_price_repository import StockPriceRepository
from app.models.database import get_async_session
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingSchedulerService:
    """임베딩 자동 업데이트 스케줄러"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.embedding_service = EmbeddingStorageService()
        self.is_running = False
    
    def start(self):
        """스케줄러를 시작합니다."""
        try:
            if not self.is_running:
                # KST 새벽 4시에 실행 (매일)
                self.scheduler.add_job(
                    self.daily_embedding_update,
                    CronTrigger(hour=4, minute=0, timezone='Asia/Seoul'),
                    id='daily_embedding_update',
                    name='Daily Embedding Update',
                    replace_existing=True
                )
                
                # 매주 일요일 새벽 3시에 전체 모델 재학습 (선택사항)
                self.scheduler.add_job(
                    self.weekly_model_retrain,
                    CronTrigger(day_of_week=6, hour=3, minute=0, timezone='Asia/Seoul'),
                    id='weekly_model_retrain',
                    name='Weekly Model Retrain',
                    replace_existing=True
                )
                
                self.scheduler.start()
                self.is_running = True
                logger.info("Embedding scheduler started successfully")
            else:
                logger.warning("Embedding scheduler is already running")
                
        except Exception as e:
            logger.error("Failed to start embedding scheduler", error=str(e))
            raise
    
    def stop(self):
        """스케줄러를 중지합니다."""
        try:
            if self.is_running:
                self.scheduler.shutdown()
                self.is_running = False
                logger.info("Embedding scheduler stopped")
            else:
                logger.warning("Embedding scheduler is not running")
                
        except Exception as e:
            logger.error("Failed to stop embedding scheduler", error=str(e))
    
    async def daily_embedding_update(self):
        """일일 임베딩 업데이트 작업"""
        try:
            logger.info("Starting daily embedding update")
            start_time = datetime.now()
            
            # 1. 활성 종목 목록 조회
            active_stocks = await self._get_active_stocks()
            logger.info(f"Found {len(active_stocks)} active stocks")
            
            if not active_stocks:
                logger.warning("No active stocks found")
                return
            
            # 2. 어제 데이터가 있는 종목들 필터링
            valid_stocks = await self._filter_stocks_with_yesterday_data(active_stocks)
            logger.info(f"Found {len(valid_stocks)} stocks with valid yesterday data")
            
            if not valid_stocks:
                logger.warning("No stocks with valid yesterday data found")
                return
            
            # 3. 배치로 임베딩 업데이트
            result = await self.embedding_service.batch_update_embeddings(
                symbols=valid_stocks,
                sequence_length=60,
                embedding_dim=64,
                model_type="autoencoder",
                include_features=True
            )
            
            # 4. 결과 로깅
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(
                "Daily embedding update completed",
                duration=f"{duration:.2f}s",
                total_stocks=len(active_stocks),
                valid_stocks=len(valid_stocks),
                indexed=result.get("indexed", 0),
                failed=result.get("failed", 0)
            )
            
        except Exception as e:
            logger.error("Failed to perform daily embedding update", error=str(e))
    
    async def weekly_model_retrain(self):
        """주간 모델 재학습 작업 (선택사항)"""
        try:
            logger.info("Starting weekly model retrain")
            
            # 활성 종목 중 상위 100개 선택 (성능 고려)
            active_stocks = await self._get_active_stocks()
            if len(active_stocks) > 100:
                active_stocks = active_stocks[:100]
            
            # 모델 재학습 (백그라운드에서 실행)
            from app.services.embedding_model_service import EmbeddingModelService
            from app.models.schemas import ModelTrainingRequest
            
            model_service = EmbeddingModelService()
            
            training_request = ModelTrainingRequest(
                symbols=active_stocks,
                sequence_length=60,
                embedding_dim=64,
                epochs=50,  # 주간 재학습이므로 에포크 수 줄임
                batch_size=32,
                learning_rate=0.001,
                validation_split=0.2
            )
            
            # 비동기로 모델 훈련 실행
            asyncio.create_task(model_service.train_model(training_request))
            
            logger.info("Weekly model retrain task started in background")
            
        except Exception as e:
            logger.error("Failed to start weekly model retrain", error=str(e))
    
    async def _get_active_stocks(self) -> List[str]:
        """활성 종목 목록을 조회합니다."""
        try:
            from app.services.stock_service import StockService
            stock_service = StockService()
            
            # 모든 활성 종목 조회 (페이지네이션으로 처리)
            all_stocks = []
            page = 0
            page_size = 1000
            
            while True:
                stocks = await stock_service.get_stocks(
                    skip=page * page_size,
                    limit=page_size,
                    is_active="Y"
                )
                
                if not stocks:
                    break
                    
                all_stocks.extend(stocks)
                page += 1
                
                # 너무 많은 데이터 방지
                if page > 100:  # 최대 10만개 종목
                    break
            
            return [stock.ticker for stock in all_stocks]
            
        except Exception as e:
            logger.error("Failed to get active stocks", error=str(e))
            return []
    
    async def _filter_stocks_with_yesterday_data(self, symbols: List[str]) -> List[str]:
        """어제 데이터가 있고 가격이 유효한 종목들을 필터링합니다."""
        try:
            # 어제 날짜 계산 (KST 기준)
            yesterday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            valid_stocks = []
            
            # 배치로 처리 (한 번에 50개씩)
            batch_size = 50
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                
                # 각 종목에 대해 어제 데이터 확인
                for symbol in batch_symbols:
                    try:
                        from app.services.stock_price_service import StockPriceService
                        stock_price_service = StockPriceService()
                        
                        # 어제 데이터 조회
                        yesterday_prices = await stock_price_service.get_stock_prices(
                            symbol=symbol,
                            start_date=yesterday,
                            end_date=today,
                            interval="1d"
                        )
                        
                        if yesterday_prices and len(yesterday_prices) > 0:
                            # 가장 최근 데이터 확인
                            latest_price = yesterday_prices[-1]
                            
                            # 가격 유효성 검사
                            if (latest_price.close_price > 0 and 
                                latest_price.open_price > 0 and 
                                latest_price.high_price > 0 and 
                                latest_price.low_price > 0 and
                                latest_price.volume >= 0):
                                
                                valid_stocks.append(symbol)
                                logger.debug(f"Valid data found for {symbol}")
                            else:
                                logger.warning(f"Invalid price data for {symbol}: {latest_price.close_price}")
                        else:
                            logger.warning(f"No yesterday data found for {symbol}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to check data for {symbol}", error=str(e))
                        continue
                
                # 배치 간 잠시 대기 (DB 부하 방지)
                await asyncio.sleep(0.1)
            
            return valid_stocks
            
        except Exception as e:
            logger.error("Failed to filter stocks with yesterday data", error=str(e))
            return []
    
    async def manual_update(self, symbols: List[str] = None) -> Dict[str, Any]:
        """수동으로 임베딩을 업데이트합니다."""
        try:
            if symbols is None:
                # 모든 활성 종목 대상
                symbols = await self._get_active_stocks()
                symbols = await self._filter_stocks_with_yesterday_data(symbols)
            
            if not symbols:
                return {
                    "success": False,
                    "message": "No valid stocks found for update",
                    "updated_count": 0
                }
            
            logger.info(f"Manual update started for {len(symbols)} stocks")
            
            result = await self.embedding_service.batch_update_embeddings(
                symbols=symbols,
                sequence_length=60,
                embedding_dim=64,
                model_type="autoencoder",
                include_features=True
            )
            
            return {
                "success": result.get("success", False),
                "message": "Manual update completed",
                "total_symbols": len(symbols),
                "indexed": result.get("indexed", 0),
                "failed": result.get("failed", 0)
            }
            
        except Exception as e:
            logger.error("Failed to perform manual update", error=str(e))
            return {
                "success": False,
                "message": f"Manual update failed: {str(e)}",
                "updated_count": 0
            }
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """스케줄러 상태를 조회합니다."""
        try:
            jobs = []
            for job in self.scheduler.get_jobs():
                jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger)
                })
            
            return {
                "is_running": self.is_running,
                "jobs": jobs,
                "job_count": len(jobs)
            }
            
        except Exception as e:
            logger.error("Failed to get scheduler status", error=str(e))
            return {
                "is_running": False,
                "jobs": [],
                "job_count": 0,
                "error": str(e)
            }


# 전역 스케줄러 인스턴스
embedding_scheduler = EmbeddingSchedulerService()
