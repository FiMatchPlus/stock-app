"""KIS API 서비스

KIS API 호출 시 필수 헤더:
- Content-Type: "application/json; charset=utf-8" (고정)
- Authorization: "Bearer {access_token}" (Redis에서 자동 조회)
- appkey: 환경변수 KIS_APP_KEY에서 가져옴
- appsecret: 환경변수 KIS_APP_SECRET에서 가져옴
- custtype: "P" (개인 고객으로 고정)
- tr_id: API별로 다름 (각 메서드에 주석으로 표시)

tr_id 예시:
- 종목 현재가 조회: HHKST01010100
- 일봉 차트 조회: HHKST01010400
- 지수 차트 조회: HHKST01010600
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import httpx
from app.config import settings
from app.utils.logger import get_logger
from app.services.token_service import korea_investment_token_service

logger = get_logger(__name__)


class KISAPIService:
    """KIS API 서비스"""
    
    def __init__(self):
        self.timeout = 30.0
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # KIS API 설정
        self.kis_base_url = settings.kis_base_url
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        tr_id: Optional[str] = None  # 요청마다 달라야 하는 tr_id
    ) -> Dict[str, Any]:
        """KIS API 요청 실행"""
        
        if not tr_id:
            raise ValueError("tr_id는 필수입니다. API 문서에서 해당 요청의 tr_id를 확인하세요.")
        
        url = f"{self.kis_base_url}{endpoint}"
        
        # KIS API 토큰 자동 관리
        access_token = await korea_investment_token_service.get_valid_token()
        
        # KIS API 필수 헤더 설정 (외부에서 수정 불가)
        headers = {
            "Content-Type": "application/json; charset=utf-8", 
            "Authorization": f"Bearer {access_token}",
            "appkey": settings.kis_app_key,
            "appsecret": settings.kis_app_secret,
            "custtype": "P",
            "tr_id": tr_id
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.request(
                        method=method,
                        url=url,
                        params=params,
                        headers=headers
                    )
                    response.raise_for_status()
                    return response.json()
                
                except httpx.HTTPStatusError as e:
                    logger.error(
                        "API request failed",
                        status_code=e.response.status_code,
                        response_text=e.response.text,
                        attempt=attempt + 1
                    )
                    
                    if attempt == self.max_retries - 1:
                        raise
                    
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                
                except httpx.RequestError as e:
                    logger.error(
                        "API request error",
                        error=str(e),
                        attempt=attempt + 1
                    )
                    
                    if attempt == self.max_retries - 1:
                        raise
                    
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
    
    
    # KIS API 메서드들
    async def get_kis_stock_info(self, symbol: str) -> Dict[str, Any]:
        """KIS API로 종목 정보 조회"""
        endpoint = f"/uapi/domestic-stock/v1/quotations/inquire-price"
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": symbol
        }
        # tr_id: HHKST01010100 (종목 현재가 조회)
        return await self._make_request("GET", endpoint, params=params, tr_id="HHKST01010100")
    
    async def get_kis_stock_prices(
        self, 
        symbol: str, 
        period: str = "D",
        count: int = 100
    ) -> Dict[str, Any]:
        """KIS API로 주가 데이터 조회"""
        endpoint = f"/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": symbol,
            "fid_input_date_1": "",
            "fid_input_date_2": "",
            "fid_period_div_code": period,
            "fid_org_adj_prc": "1"
        }
        # tr_id: HHKST01010400 (일봉 차트 조회)
        return await self._make_request("GET", endpoint, params=params, tr_id="HHKST01010400")
    
    async def get_kis_market_summary(self) -> Dict[str, Any]:
        """KIS API로 시장 요약 정보 조회"""
        endpoint = f"/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice"
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_cond_std_div_code": "01",
            "fid_input_iscd": "0001",
            "fid_input_date_1": "",
            "fid_input_date_2": "",
            "fid_period_div_code": "D",
            "fid_org_adj_prc": "1"
        }
        # tr_id: HHKST01010600 (지수 차트 조회)
        return await self._make_request("GET", endpoint, params=params, tr_id="HHKST01010600")
    
    async def search_kis_stocks(self, query: str) -> Dict[str, Any]:
        """KIS API로 종목 검색 (실제로는 종목 정보 조회)"""
        # KIS API에는 별도의 종목 검색 엔드포인트가 없으므로
        # 종목 코드로 직접 조회하는 방식 사용
        return await self.get_kis_stock_info(query)

