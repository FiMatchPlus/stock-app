"""네이버 금융 크롤링 서비스

네이버 금융에서 일별 시세 데이터를 크롤링하는 서비스
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from decimal import Decimal

import httpx
from bs4 import BeautifulSoup
from app.models.stock import StockPrice
from app.utils.logger import get_logger

logger = get_logger(__name__)


class NaverCrawlingService:
    """네이버 금융 크롤링 서비스"""
    
    def __init__(self):
        self.base_url = "https://finance.naver.com/item/sise_day.naver"
        self.timeout = 30.0
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # HTTP 헤더 설정
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    async def _make_request(self, url: str) -> str:
        """HTTP 요청 실행"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.get(url, headers=self.headers)
                    response.raise_for_status()
                    # 한글 인코딩 설정
                    response.encoding = 'euc-kr'
                    return response.text
                
                except httpx.HTTPStatusError as e:
                    logger.error(
                        "HTTP request failed",
                        url=url,
                        status_code=e.response.status_code,
                        attempt=attempt + 1
                    )
                    
                    if attempt == self.max_retries - 1:
                        raise
                    
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                
                except httpx.RequestError as e:
                    logger.error(
                        "HTTP request error",
                        url=url,
                        error=str(e),
                        attempt=attempt + 1
                    )
                    
                    if attempt == self.max_retries - 1:
                        raise
                    
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
    
    def _parse_stock_data(self, html: str, stock_code: str, target_date: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """HTML에서 주식 데이터 파싱

        테이블을 위에서부터 내려가며 날짜를 비교한다.
        - 행 날짜 > target_date: 미래 데이터이므로 다음 행으로 진행
        - 행 날짜 == target_date: 해당 행의 데이터를 반환
        - 행 날짜 < target_date: 더 이상 내려갈 필요 없음(실패로 간주) -> None 반환
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # 일별시세 테이블 찾기 (첫 번째 테이블)
            tables = soup.find_all('table')
            if not tables:
                logger.warning(f"테이블을 찾을 수 없습니다. stock_code: {stock_code}")
                return None
            
            # 첫 번째 테이블 사용 (일별시세 테이블)
            table = tables[0]
            
            # 테이블이 올바른 테이블인지 확인 (헤더에 '날짜'가 포함되어 있는지)
            header_row = table.find('tr')
            if not header_row:
                logger.warning(f"헤더 행을 찾을 수 없습니다. stock_code: {stock_code}")
                return None
            
            # th와 td 모두 검색
            header_cells = header_row.find_all(['th', 'td'])
            if len(header_cells) < 7:
                logger.warning(f"헤더 셀을 찾을 수 없습니다. stock_code: {stock_code}")
                return None
            
            # 첫 번째 셀이 '날짜'를 포함하는지 확인
            first_header = header_cells[0].get_text(strip=True)
            if '날짜' not in first_header:
                logger.warning(f"올바른 테이블이 아닙니다. 헤더: {first_header}, stock_code: {stock_code}")
                return None
            
            # target_date 기본값: 어제(KST 기준)
            if target_date is None:
                kst_now = datetime.now()
                target_date = (kst_now - timedelta(days=1)).date()
            else:
                # date 혹은 datetime이 들어오면 date로 통일
                target_date = target_date.date() if isinstance(target_date, datetime) else target_date

            # 테이블 행들 찾기 (헤더 제외)
            rows = table.find_all('tr')[1:]
            if not rows:
                logger.warning(f"데이터 행을 찾을 수 없습니다. stock_code: {stock_code}")
                return None

            for row in rows:
                cells = row.find_all(['th', 'td'])
                if len(cells) < 7:
                    continue

                date_text = cells[0].get_text(strip=True)
                if not date_text or not re.match(r'\d{4}\.\d{2}\.\d{2}', date_text):
                    continue

                try:
                    row_date = datetime.strptime(date_text, '%Y.%m.%d').date()
                except Exception:
                    continue

                # 비교 로직
                if row_date > target_date:
                    # 미래 데이터 → 아래로 계속 진행
                    continue
                if row_date < target_date:
                    # 목표일보다 과거가 나오면 실패 처리
                    logger.warning(f"목표 날짜({target_date}) 이전 데이터만 남았습니다. stock_code: {stock_code}")
                    return None

                # row_date == target_date → 데이터 추출 및 반환
                close_price = cells[1].get_text(strip=True).replace(',', '')
                change_info = cells[2].get_text(strip=True)
                open_price = cells[3].get_text(strip=True).replace(',', '')
                high_price = cells[4].get_text(strip=True).replace(',', '')
                low_price = cells[5].get_text(strip=True).replace(',', '')
                volume = cells[6].get_text(strip=True).replace(',', '')

                change_direction = ''
                change_amount = '0'
                if '상승' in change_info:
                    change_direction = '상승'
                    change_amount = change_info.replace('상승', '').strip()
                elif '하락' in change_info:
                    change_direction = '하락'
                    change_amount = change_info.replace('하락', '').strip()
                elif '보합' in change_info:
                    change_direction = '보합'
                    change_amount = '0'

                # 쉼표 제거
                change_amount = change_amount.replace(',', '') if change_amount else '0'

                return {
                    'stock_code': stock_code,
                    'date': date_text,
                    'close_price': close_price,
                    'change_direction': change_direction,
                    'change_amount': change_amount,
                    'open_price': open_price,
                    'high_price': high_price,
                    'low_price': low_price,
                    'volume': volume
                }

            # 여기까지 오면 매칭 실패
            logger.warning(f"목표 날짜({target_date}) 데이터가 없습니다. stock_code: {stock_code}")
            return None
            
        except Exception as e:
            logger.error(f"HTML 파싱 오류: {e}, stock_code: {stock_code}")
            return None
    
    def _validate_date(self, date_text: str, target_date: Optional[datetime] = None) -> bool:
        """날짜 유효성 검증: target_date와 동일한지 확인 (기본은 어제)"""
        try:
            if target_date is None:
                kst_now = datetime.now()
                target = (kst_now - timedelta(days=1)).date()
            else:
                target = target_date.date() if isinstance(target_date, datetime) else target_date

            parsed_date = datetime.strptime(date_text, '%Y.%m.%d').date()
            return parsed_date == target
        except Exception as e:
            logger.error(f"날짜 검증 오류: {e}, date_text: {date_text}")
            return False
    
    def _convert_to_stock_price(self, data: Dict[str, Any]) -> Optional[StockPrice]:
        """크롤링 데이터를 StockPrice 모델로 변환"""
        try:
            # 날짜를 datetime으로 변환
            date_obj = datetime.strptime(data['date'], '%Y.%m.%d')
            
            # 전일비 금액 계산 (상승/하락에 따라 부호 결정)
            change_amount = Decimal(data['change_amount'])
            if data['change_direction'] == '하락':
                change_amount = -change_amount
            
            # StockPrice 객체 생성
            stock_price = StockPrice(
                stock_code=data['stock_code'],
                datetime=date_obj,
                interval_unit='1d',
                open_price=Decimal(data['open_price']),
                high_price=Decimal(data['high_price']),
                low_price=Decimal(data['low_price']),
                close_price=Decimal(data['close_price']),
                volume=int(data['volume']),
                change_amount=change_amount
                # change_rate는 데이터베이스에서 자동 생성되므로 제외
            )
            
            return stock_price
            
        except Exception as e:
            logger.error(f"StockPrice 변환 오류: {e}, data: {data}")
            return None
    
    async def crawl_stock_data(self, stock_code: str, target_date: Optional[datetime] = None) -> Optional[StockPrice]:
        """특정 종목의 목표일 데이터 크롤링"""
        try:
            start_ts = datetime.now()
            # URL 구성
            url = f"{self.base_url}?code={stock_code}"
            
            logger.info(f"주식 데이터 크롤링 시작: {stock_code}")
            
            # HTTP 요청
            html = await self._make_request(url)
            
            # HTML 파싱 (목표일 검색)
            data = self._parse_stock_data(html, stock_code, target_date)
            if not data:
                duration_ms = int((datetime.now() - start_ts).total_seconds() * 1000)
                logger.warning(
                    "Crawl failed - parse returned no data",
                    stock_code=stock_code,
                    target_date=(target_date.date() if isinstance(target_date, datetime) else target_date) if target_date else "yesterday",
                    duration_ms=duration_ms,
                )
                return None
            
            # StockPrice 객체로 변환
            stock_price = self._convert_to_stock_price(data)
            if not stock_price:
                duration_ms = int((datetime.now() - start_ts).total_seconds() * 1000)
                logger.warning(
                    "Crawl failed - conversion to StockPrice returned None",
                    stock_code=stock_code,
                    duration_ms=duration_ms,
                )
                return None
            
            duration_ms = int((datetime.now() - start_ts).total_seconds() * 1000)
            logger.info(
                "Crawl succeeded",
                stock_code=stock_code,
                interval_unit=stock_price.interval_unit,
                date=str(stock_price.datetime.date()),
                duration_ms=duration_ms,
            )
            return stock_price
            
        except Exception as e:
            logger.error(
                "Crawl exception",
                error=str(e),
                stock_code=stock_code,
            )
            return None
    
    

    async def crawl_multiple_stocks_concurrent(
        self,
        stock_codes: List[str],
        concurrency: int = 50,
        target_date: Optional[datetime] = None
    ) -> List[StockPrice]:
        """여러 종목을 동시 처리(동시 최대 concurrency)로 크롤링"""
        if not stock_codes:
            return []

        sem = asyncio.Semaphore(concurrency)
        results: List[StockPrice] = []

        async def worker(code: str) -> None:
            async with sem:
                res = await self.crawl_stock_data(code, target_date)
                if isinstance(res, StockPrice):
                    results.append(res)

        logger.info(
            "Concurrent crawl started",
            total_symbols=len(stock_codes),
            concurrency=concurrency,
        )

        tasks = [asyncio.create_task(worker(code)) for code in stock_codes]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(
            "Concurrent crawl completed",
            total_symbols=len(stock_codes),
            succeeded=len(results),
            failed=len(stock_codes) - len(results),
        )
        return results
    


# 전역 서비스 인스턴스
naver_crawling_service = NaverCrawlingService()
