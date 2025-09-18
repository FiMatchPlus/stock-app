"""백테스트 서비스 - 최적화된 포트폴리오 백테스트 엔진"""

import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from concurrent.futures import ThreadPoolExecutor

from app.models.stock import StockPrice
from app.models.schemas import (
    BacktestRequest, BacktestResponse, BacktestMetrics,
    PortfolioSnapshotResponse, HoldingSnapshotResponse
)
from app.utils.logger import get_logger
# from app.services.cache_service import cache_service

logger = get_logger(__name__)


class BacktestService:
    """최적화된 백테스트 서비스"""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.cache = None  # cache_service 임시 비활성화
        
    async def run_backtest(
        self, 
        request: BacktestRequest, 
        session: Optional[AsyncSession] = None,
        portfolio_id: Optional[int] = None
    ) -> BacktestResponse:
        """백테스트 실행"""
        start_time = time.time()
        
        try:
            # 1. 입력 검증 및 데이터 준비
            logger.info(f"Starting backtest for {len(request.holdings)} holdings", 
                       start=request.start, end=request.end)
            
            # 2. 주가 데이터 조회 (캐싱 활용)
            if session is None:
                raise ValueError("Session is required for stock price data retrieval")
            stock_prices = await self._get_stock_prices_optimized(
                request, session
            )
            
            if stock_prices is None or stock_prices.empty:
                stock_codes = [holding.code for holding in request.holdings]
                raise ValueError(
                    f"주가 데이터를 찾을 수 없습니다. "
                    f"종목 코드({', '.join(stock_codes[:3])}{'...' if len(stock_codes) > 3 else ''})와 "
                    f"기간({request.start.strftime('%Y-%m-%d')} ~ {request.end.strftime('%Y-%m-%d')})을 확인해주세요."
                )
            
            # 3. 백테스트 실행
            portfolio_data, result_summary = await self._execute_backtest(
                request, stock_prices
            )
            
            # 4. 성과 지표 계산
            metrics = await self._calculate_metrics(result_summary)
            
            # 5. 포트폴리오 스냅샷 생성 (메모리만)
            portfolio_snapshot = self._create_portfolio_snapshot_response(
                request, portfolio_data, portfolio_id
            )
            
            execution_time = time.time() - start_time
            portfolio_snapshot.execution_time = execution_time
            
            logger.info(f"Backtest completed successfully", 
                       execution_time=f"{execution_time:.3f}s",
                       portfolio_id=portfolio_snapshot.id)
            
            return BacktestResponse(
                portfolio_snapshot=portfolio_snapshot,
                metrics=metrics,
                result_summary=result_summary,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Backtest failed", error=str(e))
            raise
    
    async def _get_stock_prices_optimized(
        self, 
        request: BacktestRequest, 
        session: AsyncSession
    ) -> pd.DataFrame:
        """최적화된 주가 데이터 조회"""
        # 캐시 키 생성
        cache_key = self._generate_cache_key(request)
        
        # 캐시에서 조회 시도 (임시 비활성화)
        # if self.cache:
        #     cached_data = await self.cache.get(cache_key)
        #     if cached_data:
        #         logger.info("Using cached stock price data")
        #         return pd.read_json(cached_data)
        
        # 종목 코드 추출
        stock_codes = [holding.code for holding in request.holdings]
        
        # 데이터베이스에서 조회
        query = select(StockPrice).where(
            and_(
                StockPrice.stock_code.in_(stock_codes),
                StockPrice.datetime >= request.start,
                StockPrice.datetime <= request.end,
                StockPrice.interval_unit == "1d"
            )
        ).order_by(StockPrice.datetime)
        
        result = await session.execute(query)
        stock_prices = result.scalars().all()
        
        if stock_prices is None or len(stock_prices) == 0:
            return pd.DataFrame()
        
        # DataFrame으로 변환
        data = []
        for price in stock_prices:
            data.append({
                'stock_code': price.stock_code,
                'datetime': price.datetime,
                'close_price': price.close_price,
                'open_price': price.open_price,
                'high_price': price.high_price,
                'low_price': price.low_price,
                'volume': price.volume
            })
        
        df = pd.DataFrame(data)
        
        # 데이터 전처리 및 최적화
        df = await self._preprocess_stock_data(df)
        
        # 캐시에 저장 (임시 비활성화)
        # if self.cache:
        #     await self.cache.set(cache_key, df.to_json(), ttl=3600)
        
        return df
    
    async def _preprocess_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """주가 데이터 전처리"""
        if df is None or df.empty:
            return df
        
        # 날짜별로 정렬
        df = df.sort_values(['datetime', 'stock_code'])
        
        # 결측값 처리 (전진 채우기)
        df['close_price'] = df.groupby('stock_code')['close_price'].fillna(method='ffill')
        
        # 수익률 계산 (벡터화 연산)
        df['returns'] = df.groupby('stock_code')['close_price'].pct_change()
        
        # 첫 번째 날의 수익률은 0으로 설정
        df['returns'] = df['returns'].fillna(0)
        
        return df
    
    async def _execute_backtest(
        self, 
        request: BacktestRequest, 
        stock_prices: pd.DataFrame
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """백테스트 실행 (수량 기반 벡터화 연산)"""
        
        # 포트폴리오 보유 수량 딕셔너리 생성
        quantities = {holding.code: holding.quantity for holding in request.holdings}
        
        # 피벗 테이블로 변환 (날짜 x 종목)
        price_pivot = stock_prices.pivot_table(
            index='datetime', 
            columns='stock_code', 
            values='close_price',
            aggfunc='first'
        )
        
        # 수익률 피벗 테이블
        returns_pivot = stock_prices.pivot_table(
            index='datetime', 
            columns='stock_code', 
            values='returns',
            aggfunc='first'
        )
        
        # 공통 종목 필터링 (여기서 정의)
        common_stocks = price_pivot.columns.intersection(quantities.keys())
        price_data = price_pivot[common_stocks]
        
        # 포트폴리오 가치 계산 (수량 기반)
        portfolio_values, portfolio_returns = self._calculate_portfolio_values_by_quantity(
            price_pivot, quantities
        )
        
        # 일별 종목별 데이터 생성
        portfolio_data = []
        result_summary = []
        
        for i, (date, value) in enumerate(portfolio_values.items()):
            # 포트폴리오 데이터
            portfolio_data.append({
                'datetime': date,
                'portfolio_value': float(value),
                'daily_return': float(portfolio_returns.iloc[i]) if i < len(portfolio_returns) else 0.0
            })
            
            # 종목별 일별 데이터 생성
            daily_stocks = []
            for stock_code in common_stocks:
                if pd.notna(price_data.loc[date, stock_code]):
                    stock_price = price_data.loc[date, stock_code]
                    stock_return = returns_pivot.loc[date, stock_code] if pd.notna(returns_pivot.loc[date, stock_code]) else 0.0
                    stock_quantity = quantities[stock_code]
                    stock_value = stock_price * stock_quantity
                    stock_weight = stock_value / value if value > 0 else 0.0
                    
                    # 포트폴리오 수익률 기여도 계산
                    portfolio_contribution = stock_return * stock_weight if stock_weight > 0 else 0.0
                    
                    daily_stocks.append({
                        'stock_code': stock_code,
                        'date': date.isoformat(),
                        'close_price': float(stock_price),
                        'daily_return': float(stock_return),
                        'portfolio_weight': float(stock_weight),
                        'portfolio_contribution': float(portfolio_contribution),
                        'value': float(stock_value)
                    })
            
            # 결과 요약 데이터 (종목별 데이터 포함)
            summary_item = {
                'date': date.isoformat(),
                'stocks': daily_stocks
            }
            
            result_summary.append(summary_item)
        
        return portfolio_data, result_summary
    
    def _calculate_portfolio_returns(
        self, 
        returns_pivot: pd.DataFrame, 
        weights: Dict[str, float]
    ) -> pd.Series:
        """포트폴리오 수익률 계산 (벡터화)"""
        # 가중치 벡터 생성
        weight_vector = pd.Series(weights)
        
        # 수익률과 가중치 정렬
        common_stocks = returns_pivot.columns.intersection(weight_vector.index)
        returns_aligned = returns_pivot[common_stocks]
        weights_aligned = weight_vector[common_stocks]
        
        # 정규화 (가중치 합이 1이 되도록)
        weights_aligned = weights_aligned / weights_aligned.sum()
        
        # 포트폴리오 수익률 계산 (행렬 곱셈)
        portfolio_returns = returns_aligned.dot(weights_aligned)
        
        return portfolio_returns
    
    def _calculate_portfolio_values(
        self, 
        portfolio_returns: pd.Series, 
        initial_capital: Decimal
    ) -> pd.Series:
        """포트폴리오 가치 계산"""
        # 누적 수익률 계산
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # 포트폴리오 가치 계산
        portfolio_values = cumulative_returns * float(initial_capital)
        
        return portfolio_values
    
    def _calculate_portfolio_values_by_quantity(
        self, 
        price_pivot: pd.DataFrame, 
        quantities: Dict[str, int]
    ) -> Tuple[pd.Series, pd.Series]:
        """수량 기반 포트폴리오 가치 및 수익률 계산"""
        
        # 공통 종목만 필터링 (이미 상위에서 정의됨)
        common_stocks = price_pivot.columns.intersection(quantities.keys())
        price_data = price_pivot[common_stocks]
        
        # 각 종목별 가치 계산 (가격 × 수량)
        portfolio_values_daily = pd.Series(index=price_data.index, dtype=float)
        
        for date in price_data.index:
            daily_value = 0
            for stock in common_stocks:
                if pd.notna(price_data.loc[date, stock]):
                    stock_value = price_data.loc[date, stock] * quantities[stock]
                    daily_value += stock_value
            portfolio_values_daily[date] = daily_value
        
        # 일별 수익률 계산
        portfolio_returns = portfolio_values_daily.pct_change().fillna(0)
        
        return portfolio_values_daily, portfolio_returns
    
    async def _calculate_metrics(self, result_summary: List[Dict[str, Any]]) -> BacktestMetrics:
        """성과 지표 계산 (최적화된 버전)"""
        if not result_summary:
            raise ValueError("백테스트 결과 데이터가 없어 성과 지표를 계산할 수 없습니다.")
        
        # 포트폴리오 수익률 배열 추출 (종목별 기여도 합계)
        returns = []
        for rs in result_summary:
            # 각 날짜의 모든 종목 기여도 합계
            daily_return = sum(stock['portfolio_contribution'] for stock in rs['stocks'])
            returns.append(daily_return)
        
        returns = np.array(returns)
        
        # 기본 통계량 계산
        total_return = (1 + returns).prod() - 1
        
        # 연환산 수익률
        days = len(returns)
        annualized_return = (1 + total_return) ** (252 / days) - 1
        
        # 변동성 (연환산)
        volatility = returns.std() * np.sqrt(252)
        
        # 샤프 비율 (무위험 수익률 0% 가정)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # 최대 낙폭 계산
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # VaR/CVaR 계산 (병렬 처리)
        var_95, var_99, cvar_95, cvar_99 = await self._calculate_var_cvar(returns)
        
        # 승률 및 손익비
        win_rate, profit_loss_ratio = self._calculate_win_loss_metrics(returns)
        
        return BacktestMetrics(
            total_return=Decimal(str(total_return)),
            annualized_return=Decimal(str(annualized_return)),
            volatility=Decimal(str(volatility)),
            sharpe_ratio=Decimal(str(sharpe_ratio)),
            max_drawdown=Decimal(str(max_drawdown)),
            var_95=Decimal(str(var_95)),
            var_99=Decimal(str(var_99)),
            cvar_95=Decimal(str(cvar_95)),
            cvar_99=Decimal(str(cvar_99)),
            win_rate=Decimal(str(win_rate)),
            profit_loss_ratio=Decimal(str(profit_loss_ratio))
        )
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """최대 낙폭 계산"""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    async def _calculate_var_cvar(
        self, 
        returns: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """VaR/CVaR 계산 (병렬 처리)"""
        loop = asyncio.get_event_loop()
        
        # 정렬된 수익률 배열 생성 (공통 사용)
        sorted_returns = np.sort(returns)
        
        # VaR 계산 (병렬)
        var_95_task = loop.run_in_executor(
            self.thread_pool, 
            self._calculate_var, 
            sorted_returns, 0.05
        )
        var_99_task = loop.run_in_executor(
            self.thread_pool, 
            self._calculate_var, 
            sorted_returns, 0.01
        )
        
        # CVaR 계산 (병렬)
        cvar_95_task = loop.run_in_executor(
            self.thread_pool, 
            self._calculate_cvar, 
            sorted_returns, 0.05
        )
        cvar_99_task = loop.run_in_executor(
            self.thread_pool, 
            self._calculate_cvar, 
            sorted_returns, 0.01
        )
        
        # 결과 대기
        var_95, var_99, cvar_95, cvar_99 = await asyncio.gather(
            var_95_task, var_99_task, cvar_95_task, cvar_99_task
        )
        
        return var_95, var_99, cvar_95, cvar_99
    
    def _calculate_var(self, sorted_returns: np.ndarray, confidence_level: float) -> float:
        """VaR 계산"""
        index = int(confidence_level * len(sorted_returns))
        return sorted_returns[index]
    
    def _calculate_cvar(self, sorted_returns: np.ndarray, confidence_level: float) -> float:
        """CVaR 계산"""
        index = int(confidence_level * len(sorted_returns))
        if index == 0:
            return 0.0
        return sorted_returns[:index].mean() if not np.isnan(sorted_returns[:index].mean()) else 0.0
    
    def _calculate_win_loss_metrics(self, returns: np.ndarray) -> Tuple[float, float]:
        """승률 및 손익비 계산"""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0
        
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        return win_rate, profit_loss_ratio
    
    def _create_portfolio_snapshot_response(
        self, 
        request: BacktestRequest, 
        portfolio_data: List[Dict[str, Any]], 
        portfolio_id: Optional[int] = None
    ) -> PortfolioSnapshotResponse:
        """포트폴리오 스냅샷 응답 생성 (메모리만)"""
        if not portfolio_data:
            raise ValueError("포트폴리오 데이터가 없습니다.")
        
        # 포트폴리오 ID 생성 (없는 경우)
        if portfolio_id is None:
            portfolio_id = int(time.time())
        
        # 최종 포트폴리오 값
        final_data = portfolio_data[-1]
        initial_data = portfolio_data[0]
        base_value = Decimal(str(initial_data['portfolio_value']))
        current_value = Decimal(str(final_data['portfolio_value']))
        
        # 보유 종목 정보 생성
        holdings = []
        for i, holding in enumerate(request.holdings):
            holdings.append(HoldingSnapshotResponse(
                id=i + 1,
                stock_id=holding.code,
                quantity=holding.quantity
            ))
        
        # 포트폴리오 스냅샷 응답 생성
        return PortfolioSnapshotResponse(
            id=12345,  # 고정 ID (실제 DB 저장하지 않음)
            portfolio_id=portfolio_id,
            base_value=base_value,
            current_value=current_value,
            start_at=request.start,
            end_at=request.end,
            created_at=datetime.utcnow(),
            execution_time=0.0,  # 나중에 설정됨
            holdings=holdings
        )

