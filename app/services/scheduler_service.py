"""스케줄러 서비스

매일 새벽 2시 KST에 네이버 금융에서 주식 데이터를 크롤링하는 스케줄러
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import List

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.database import AsyncSessionLocal
from app.models.stock import Stock, StockPrice
from app.services.naver_crawling_service import naver_crawling_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SchedulerService:
    """스케줄러 서비스"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
    
    def start(self):
        """스케줄러 시작"""
        if self.is_running:
            logger.warning("스케줄러가 이미 실행 중입니다.")
            return
        
        # 매일 새벽 2시 KST에 실행
        self.scheduler.add_job(
            func=self._crawl_daily_stock_data,
            trigger=CronTrigger(
                hour=2,  # 새벽 2시
                minute=0,  # 0분
                timezone='Asia/Seoul'  # KST
            ),
            id='daily_stock_crawling',
            name='Daily Stock Data Crawling',
            replace_existing=True
        )
        
        self.scheduler.start()
        self.is_running = True
        logger.info("스케줄러가 시작되었습니다. 매일 새벽 2시 KST에 실행됩니다.")
    
    def stop(self):
        """스케줄러 중지"""
        if not self.is_running:
            logger.warning("스케줄러가 실행 중이 아닙니다.")
            return
        
        self.scheduler.shutdown()
        self.is_running = False
        logger.info("스케줄러가 중지되었습니다.")
    
    async def _crawl_daily_stock_data(self):
        """일일 주식 데이터 크롤링 작업"""
        logger.info("일일 주식 데이터 크롤링 작업을 시작합니다.")
        
        try:
            # 데이터베이스 세션 생성
            async with AsyncSessionLocal() as db:
                # stocks 테이블에서 모든 활성화된 종목의 ticker 조회
                result = await db.execute(
                    select(Stock.ticker).where(Stock.is_active == 'Y')
                )
                stock_codes = [row[0] for row in result.fetchall()]
                
                if not stock_codes:
                    logger.warning("활성화된 종목이 없습니다.")
                    return
                
                logger.info(f"총 {len(stock_codes)}개 종목의 데이터를 크롤링합니다.")
                
                # 여러 종목 크롤링 (서비스 기본 규칙: 목표일 기본 어제)
                # 동시 50개로 처리하여 전체 시간을 단축
                stock_prices = await naver_crawling_service.crawl_multiple_stocks_concurrent(
                    stock_codes=stock_codes,
                    concurrency=50
                )
                
                if not stock_prices:
                    logger.warning("크롤링된 데이터가 없습니다.")
                    return
                
                # 데이터베이스에 저장
                saved_count = await self._save_stock_prices(db, stock_prices)
                
                logger.info(f"일일 주식 데이터 크롤링 완료. {saved_count}개 데이터가 저장되었습니다.")
                
        except Exception as e:
            logger.error(f"일일 주식 데이터 크롤링 오류: {e}")
    
    async def _save_stock_prices(self, db: AsyncSession, stock_prices: List[StockPrice]) -> int:
        """주식 가격 데이터를 배치로 저장. 실패 시 개별 저장으로 재시도."""
        if not stock_prices:
            return 0
        
        # 1) 배치 저장 시도
        try:
            db.add_all(stock_prices)
            await db.commit()
            logger.info(f"배치 저장 완료: {len(stock_prices)}개")
            return len(stock_prices)
        except Exception as e:
            await db.rollback()
            logger.warning(f"배치 저장 실패, 개별 저장으로 재시도: {e}")
        
        # 2) 개별 저장 재시도 (충돌 항목 건너뜀)
        saved_count = 0
        for sp in stock_prices:
            try:
                db.add(sp)
                await db.commit()
                saved_count += 1
            except Exception as e:
                await db.rollback()
                logger.debug(f"건너뜀: {sp.stock_code} ({sp.datetime}) - {e}")
        
        logger.info(f"개별 저장 완료: {saved_count}/{len(stock_prices)}개")
        return saved_count
    
    async def run_manual_crawling(self, stock_codes: List[str] = None) -> dict:
        """수동 크롤링 실행 (테스트용)"""
        logger.info("수동 크롤링을 시작합니다.")
        
        try:
            async with AsyncSessionLocal() as db:
                if stock_codes is None:
                    # stocks 테이블에서 모든 활성화된 종목의 ticker 조회
                    result = await db.execute(
                        select(Stock.ticker).where(Stock.is_active == 'Y')
                    )
                    stock_codes = [row[0] for row in result.fetchall()]
                
                if not stock_codes:
                    return {"success": False, "message": "활성화된 종목이 없습니다."}
                
                # 크롤링 실행 (동시 50개 처리)
                stock_prices = await naver_crawling_service.crawl_multiple_stocks_concurrent(
                    stock_codes=stock_codes,
                    concurrency=50
                )
                
                if not stock_prices:
                    return {"success": False, "message": "크롤링된 데이터가 없습니다."}
                
                # 데이터베이스에 저장
                saved_count = await self._save_stock_prices(db, stock_prices)
                
                return {
                    "success": True,
                    "message": f"{saved_count}개 데이터가 저장되었습니다.",
                    "crawled_count": len(stock_prices),
                    "saved_count": saved_count
                }
                
        except Exception as e:
            logger.error(f"수동 크롤링 오류: {e}")
            return {"success": False, "message": f"크롤링 오류: {str(e)}"}
    
    def get_scheduler_status(self) -> dict:
        """스케줄러 상태 조회"""
        if not self.is_running:
            return {"running": False, "jobs": []}
        
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
        
        return {
            "running": self.is_running,
            "jobs": jobs
        }


# 전역 스케줄러 서비스 인스턴스
scheduler_service = SchedulerService()
