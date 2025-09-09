#!/usr/bin/env python3
"""KIS API 테스트 스크립트"""

import asyncio
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.external_api_service import KISAPIService
from app.services.token_service import korea_investment_token_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def test_token_service():
    """KIS 토큰 서비스 테스트"""
    print("=== KIS 토큰 서비스 테스트 ===")
    
    try:
        # 토큰 정보 조회
        token_info = await korea_investment_token_service.get_token_info()
        print(f"토큰 정보: {token_info}")
        
        # 유효한 토큰 가져오기
        token = await korea_investment_token_service.get_valid_token()
        print(f"액세스 토큰 (처음 20자): {token[:20]}...")
        
    except Exception as e:
        print(f"KIS 토큰 서비스 테스트 실패: {e}")


async def test_kis_stock_api():
    """KIS 주식 API 테스트"""
    print("\n=== KIS 주식 API 테스트 ===")
    
    api_service = KISAPIService()
    
    try:
        # 삼성전자 종목 정보 조회
        print("삼성전자 종목 정보 조회 중...")
        stock_info = await api_service.get_kis_stock_info("005930")
        print(f"종목 정보: {stock_info}")
        
    except Exception as e:
        print(f"KIS 주식 API 테스트 실패: {e}")


async def main():
    """메인 함수"""
    print("KIS API 테스트 시작")
    
    # 토큰 서비스 테스트
    await test_token_service()
    
    # 주식 API 테스트
    await test_kis_stock_api()
    
    print("\n테스트 완료")


if __name__ == "__main__":
    asyncio.run(main())
