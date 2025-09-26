"""벤치마크 지수 자동 결정 및 수익률 조회 서비스"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, asc, desc

from app.models.stock import BenchmarkPrice, Stock
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkService:
    """벤치마크 지수 자동 결정 및 수익률 조회 서비스"""
    
    def __init__(self):
        # KOSPI/KOSDAQ 종목 분류 기준
        self.kospi_indicators = ['KOSPI', 'KQ']
        self.kosdaq_indicators = ['KOSDAQ', 'KQ']
        
        # 기본 벤치마크 매핑
        self.default_benchmarks = {
            'KOSPI': 'KOSPI',
            'KOSDAQ': 'KOSDAQ'
        }
    
    async def determine_benchmark(
        self,
        holdings: List[Dict[str, any]],
        session: AsyncSession
    ) -> str:
        """
        포트폴리오 구성 종목을 분석하여 적절한 벤치마크 지수 결정
        
        Args:
            holdings: 보유 종목 리스트 [{"code": "005930", "quantity": 100}, ...]
            session: 데이터베이스 세션
            
        Returns:
            str: 결정된 벤치마크 지수 코드 (KOSPI, KOSDAQ 등)
        """
        if not holdings:
            return "KOSPI"  # 기본값
        
        stock_codes = [holding['code'] for holding in holdings]
        
        try:
            # 종목별 시장 구분 조회
            market_analysis = await self._analyze_portfolio_markets(
                stock_codes, session
            )
            
            # 벤치마크 결정 로직
            benchmark = await self._select_benchmark(market_analysis)
            
            logger.info(
                "Benchmark determined",
                portfolio_size=len(holdings),
                market_distribution=market_analysis,
                selected_benchmark=benchmark
            )
            
            return benchmark
            
        except Exception as e:
            logger.error(f"Failed to determine benchmark: {str(e)}")
            return "KOSPI"  # 기본값
    
    async def _analyze_portfolio_markets(
        self,
        stock_codes: List[str],
        session: AsyncSession
    ) -> Dict[str, int]:
        """포트폴리오 내 시장별 종목 분포 분석"""
        
        # 종목 정보 조회
        query = select(Stock).where(Stock.ticker.in_(stock_codes))
        result = await session.execute(query)
        stocks = result.scalars().all()
        
        # 시장별 분류
        market_counts = {
            'KOSPI': 0,
            'KOSDAQ': 0,
            'OTHER': 0
        }
        
        for stock in stocks:
            market = stock.market.upper() if stock.market else 'OTHER'
            
            if market in ['KOSPI', 'KQ']:
                market_counts['KOSPI'] += 1
            elif market in ['KOSDAQ', 'KQ']:
                market_counts['KOSDAQ'] += 1
            else:
                market_counts['OTHER'] += 1
        
        return market_counts
    
    async def _select_benchmark(self, market_analysis: Dict[str, int]) -> str:
        """시장 분석 결과를 바탕으로 벤치마크 선택"""
        
        kospi_count = market_analysis.get('KOSPI', 0)
        kosdaq_count = market_analysis.get('KOSDAQ', 0)
        other_count = market_analysis.get('OTHER', 0)
        total_count = sum(market_analysis.values())
        
        if total_count == 0:
            return "KOSPI"
        
        # 비중 계산
        kospi_ratio = kospi_count / total_count
        kosdaq_ratio = kosdaq_count / total_count
        
        # 결정 로직
        if kospi_ratio >= 0.6:  # KOSPI 종목이 60% 이상
            return "KOSPI"
        elif kosdaq_ratio >= 0.6:  # KOSDAQ 종목이 60% 이상
            return "KOSDAQ"
        elif kospi_ratio > kosdaq_ratio:  # KOSPI가 더 많음
            return "KOSPI"
        elif kosdaq_ratio > kospi_ratio:  # KOSDAQ이 더 많음
            return "KOSDAQ"
        else:
            # 비슷하거나 기타 종목이 많은 경우 KOSPI 기본값
            return "KOSPI"
    
    async def get_benchmark_returns(
        self,
        benchmark_code: str,
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession
    ) -> pd.Series:
        """
        벤치마크 지수의 기간별 수익률 조회
        
        Args:
            benchmark_code: 벤치마크 지수 코드 (KOSPI, KOSDAQ 등)
            start_date: 시작일
            end_date: 종료일
            session: 데이터베이스 세션
            
        Returns:
            pd.Series: 일별 수익률 시계열 (날짜가 인덱스)
        """
        try:
            # 벤치마크 가격 데이터 조회
            query = (
                select(BenchmarkPrice)
                .where(
                    and_(
                        BenchmarkPrice.index_code == benchmark_code,
                        BenchmarkPrice.datetime >= start_date,
                        BenchmarkPrice.datetime <= end_date
                    )
                )
                .order_by(asc(BenchmarkPrice.datetime))
            )
            
            result = await session.execute(query)
            benchmark_prices = result.scalars().all()
            
            if not benchmark_prices:
                logger.warning(
                    f"No benchmark data found",
                    benchmark_code=benchmark_code,
                    start_date=start_date,
                    end_date=end_date
                )
                return pd.Series(dtype=float)
            
            # DataFrame으로 변환
            data = []
            for price in benchmark_prices:
                data.append({
                    'datetime': price.datetime,
                    'close_price': float(price.close_price)
                })
            
            df = pd.DataFrame(data)
            df = df.set_index('datetime')
            df = df.sort_index()
            
            # 수익률 계산
            returns = df['close_price'].pct_change().fillna(0)
            
            logger.info(
                f"Benchmark returns retrieved",
                benchmark_code=benchmark_code,
                data_points=len(returns),
                start_date=returns.index[0] if not returns.empty else None,
                end_date=returns.index[-1] if not returns.empty else None
            )
            
            return returns
            
        except Exception as e:
            logger.error(
                f"Failed to get benchmark returns",
                benchmark_code=benchmark_code,
                error=str(e)
            )
            # 트랜잭션 실패로 인한 에러는 더 명확한 메시지와 함께 전파
            if "InFailedSQLTransaction" in str(e) or "transaction is aborted" in str(e):
                raise Exception(f"Database transaction failed during benchmark retrieval: {str(e)}")
            return pd.Series(dtype=float)
    
    async def get_available_benchmarks(self, session: AsyncSession) -> List[str]:
        """사용 가능한 벤치마크 지수 목록 조회"""
        try:
            query = select(BenchmarkPrice.index_code).distinct()
            result = await session.execute(query)
            benchmarks = [row[0] for row in result.fetchall()]
            
            return sorted(benchmarks)
            
        except Exception as e:
            logger.error(f"Failed to get available benchmarks: {str(e)}")
            return ["KOSPI", "KOSDAQ"]  # 기본값
    
    async def validate_benchmark(
        self,
        benchmark_code: str,
        session: AsyncSession
    ) -> bool:
        """벤치마크 지수 코드 유효성 검증"""
        try:
            available_benchmarks = await self.get_available_benchmarks(session)
            return benchmark_code in available_benchmarks
            
        except Exception as e:
            logger.error(f"Failed to validate benchmark: {str(e)}")
            return False
    
    async def get_benchmark_info(
        self,
        benchmark_code: str,
        session: AsyncSession
    ) -> Optional[Dict[str, any]]:
        """벤치마크 지수 정보 조회"""
        try:
            # 가장 최근 데이터 조회
            query = (
                select(BenchmarkPrice)
                .where(BenchmarkPrice.index_code == benchmark_code)
                .order_by(BenchmarkPrice.datetime.desc())
                .limit(1)
            )
            
            result = await session.execute(query)
            latest_price = result.scalar_one_or_none()
            
            if not latest_price:
                return None
            
            # 데이터 범위 조회
            min_max_query = (
                select(
                    BenchmarkPrice.datetime.label('min_date'),
                    BenchmarkPrice.datetime.label('max_date')
                )
                .where(BenchmarkPrice.index_code == benchmark_code)
                .order_by(asc(BenchmarkPrice.datetime))
                .limit(1)
            )
            
            result = await session.execute(min_max_query)
            min_data = result.fetchone()
            
            max_query = (
                select(BenchmarkPrice.datetime.label('max_date'))
                .where(BenchmarkPrice.index_code == benchmark_code)
                .order_by(BenchmarkPrice.datetime.desc())
                .limit(1)
            )
            
            result = await session.execute(max_query)
            max_data = result.fetchone()
            
            return {
                'benchmark_code': benchmark_code,
                'latest_price': float(latest_price.close_price),
                'latest_date': latest_price.datetime,
                'data_range': {
                    'start_date': min_data[0] if min_data else None,
                    'end_date': max_data[0] if max_data else None
                },
                'latest_change_rate': float(latest_price.change_rate)
            }
            
        except Exception as e:
            logger.error(f"Failed to get benchmark info: {str(e)}")
            return None
