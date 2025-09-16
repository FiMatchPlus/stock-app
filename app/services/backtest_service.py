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
from sqlalchemy.orm import selectinload
import functools
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

from app.models.stock import StockPrice, PortfolioSnapshot, HoldingSnapshot
from app.models.schemas import (
    BacktestRequest, BacktestResponse, BacktestMetrics,
    PortfolioSnapshotResponse, HoldingSnapshotResponse
)
from app.utils.logger import get_logger
from app.services.metrics_service import MetricsService
from app.utils.mongodb_client import MongoDBClient
# from app.services.cache_service import cache_service

logger = get_logger(__name__)


class BacktestService:
    """최적화된 백테스트 서비스"""
    
    def __init__(self, mongodb_client: MongoDBClient = None):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.cache = None  # cache_service 임시 비활성화
        self.metrics_service = MetricsService(mongodb_client) if mongodb_client else None
        
    async def run_backtest(
        self, 
        request: BacktestRequest, 
        session: AsyncSession,
        portfolio_id: Optional[int] = None,
        portfolio_name: Optional[str] = None
    ) -> BacktestResponse:
        """백테스트 실행"""
        start_time = time.time()
        
        try:
            # 1. 입력 검증 및 데이터 준비
            logger.info(f"Starting backtest for {len(request.holdings)} holdings", 
                       start=request.start, end=request.end)
            
            # 2. 주가 데이터 조회 (캐싱 활용)
            stock_prices = await self._get_stock_prices_optimized(
                request, session
            )
            
            if stock_prices is None or stock_prices.empty:
                raise ValueError("No stock price data available for the given period")
            
            # 3. 백테스트 실행
            portfolio_data, result_summary = await self._execute_backtest(
                request, stock_prices
            )
            
            # 4. 성과 지표 계산
            metrics = await self._calculate_metrics(result_summary)
            
            # 5. 데이터베이스 저장
            portfolio_snapshot = await self._save_portfolio_snapshot(
                request, portfolio_data, session, portfolio_id, portfolio_name
            )
            
            # 6. MongoDB에 metrics 저장
            metric_id = None
            if self.metrics_service:
                try:
                    metric_id = await self.metrics_service.save_metrics(
                        metrics, portfolio_snapshot.id
                    )
                    # PortfolioSnapshot에 metric_id 업데이트
                    await self._update_portfolio_snapshot_metric_id(
                        session, portfolio_snapshot.id, metric_id
                    )
                except Exception as e:
                    logger.error(f"Failed to save metrics to MongoDB: {str(e)}")
                    # MongoDB 저장 실패해도 백테스트는 계속 진행
            
            execution_time = time.time() - start_time
            
            # execution_time을 데이터베이스에 업데이트
            await self._update_portfolio_snapshot_execution_time(
                session, portfolio_snapshot.id, execution_time
            )
            
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
        """백테스트 실행 (벡터화 연산 활용)"""
        
        # 포트폴리오 가중치 딕셔너리 생성
        weights = {holding.code: holding.weight for holding in request.holdings}
        
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
        
        # 포트폴리오 수익률 계산 (벡터화)
        portfolio_returns = self._calculate_portfolio_returns(
            returns_pivot, weights
        )
        
        # 포트폴리오 가치 계산
        portfolio_values = self._calculate_portfolio_values(
            portfolio_returns, request.initial_capital
        )
        
        # 일별 데이터 생성
        portfolio_data = []
        result_summary = []
        
        for i, (date, value) in enumerate(portfolio_values.items()):
            # 포트폴리오 데이터
            portfolio_data.append({
                'datetime': date,
                'portfolio_value': float(value),
                'daily_return': float(portfolio_returns.iloc[i]) if i < len(portfolio_returns) else 0.0
            })
            
            # 결과 요약 데이터 (간소화)
            summary_item = {
                'date': date.isoformat(),
                'portfolio_return': float(portfolio_returns.iloc[i]) if i < len(portfolio_returns) else 0.0,
                'portfolio_value': float(value),
                'sharpe_ratio': None  # 나중에 계산됨
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
    
    async def _calculate_metrics(self, result_summary: List[Dict[str, Any]]) -> BacktestMetrics:
        """성과 지표 계산 (최적화된 버전)"""
        if not result_summary:
            raise ValueError("No result summary data available")
        
        # 수익률 배열 추출
        returns = np.array([rs['portfolio_return'] for rs in result_summary])
        
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
    
    async def _save_portfolio_snapshot(
        self, 
        request: BacktestRequest, 
        portfolio_data: List[Dict[str, Any]], 
        session: AsyncSession,
        portfolio_id: Optional[int] = None,
        portfolio_name: Optional[str] = None,
        execution_time: Optional[float] = None
    ) -> PortfolioSnapshotResponse:
        """포트폴리오 스냅샷 저장"""
        if not portfolio_data:
            raise ValueError("No portfolio data to save")
        
        # 포트폴리오 ID 생성 (없는 경우)
        if portfolio_id is None:
            portfolio_id = int(time.time())
        
        # 최종 포트폴리오 값
        final_data = portfolio_data[-1]
        base_value = Decimal(str(request.initial_capital))
        current_value = Decimal(str(final_data['portfolio_value']))
        
        # 포트폴리오 스냅샷 생성
        portfolio_snapshot = PortfolioSnapshot(
            portfolio_id=portfolio_id,
            # portfolio_name=portfolio_name,  # DB에 컬럼이 없음
            base_value=base_value,
            current_value=current_value,
            start_at=request.start,
            end_at=request.end,
            execution_time=execution_time
        )
        
        session.add(portfolio_snapshot)
        await session.flush()  # ID 생성
        
        # 보유 종목 스냅샷 생성
        for holding in request.holdings:
            # 해당 종목의 최종 가격 찾기
            stock_price = await self._get_final_stock_price(
                holding.code, request.end, session
            )
            
            if stock_price:
                # 보유 수량 계산
                quantity = int((current_value * Decimal(str(holding.weight))) / Decimal(str(stock_price.close_price)))
                
                # 보유 가치 계산
                holding_value = Decimal(str(stock_price.close_price)) * quantity
                
                holding_snapshot = HoldingSnapshot(
                    stock_id=holding.code,
                    portfolio_snapshot_id=portfolio_snapshot.id,
                    weight=Decimal(str(holding.weight)),
                    price=Decimal(str(stock_price.close_price)),
                    quantity=quantity,
                    value=holding_value,
                    recorded_at=request.end
                )
                
                session.add(holding_snapshot)
        
        await session.commit()
        
        # 응답 객체 생성
        return PortfolioSnapshotResponse(
            id=portfolio_snapshot.id,
            portfolio_id=portfolio_snapshot.portfolio_id,
            portfolio_name=portfolio_snapshot.portfolio_name,
            base_value=portfolio_snapshot.base_value,
            current_value=portfolio_snapshot.current_value,
            start_at=portfolio_snapshot.start_at,
            end_at=portfolio_snapshot.end_at,
            created_at=portfolio_snapshot.created_at,
            metric_id=portfolio_snapshot.metric_id,
            holdings=[]
        )
    
    async def _update_portfolio_snapshot_metric_id(
        self, 
        session: AsyncSession, 
        portfolio_snapshot_id: int, 
        metric_id: str
    ):
        """포트폴리오 스냅샷에 metric_id 업데이트"""
        try:
            from sqlalchemy import update
            
            stmt = update(PortfolioSnapshot).where(
                PortfolioSnapshot.id == portfolio_snapshot_id
            ).values(metric_id=metric_id)
            
            await session.execute(stmt)
            await session.commit()
            
            logger.info(
                f"Updated portfolio snapshot with metric_id",
                portfolio_snapshot_id=portfolio_snapshot_id,
                metric_id=metric_id
            )
            
        except Exception as e:
            logger.error(f"Failed to update portfolio snapshot metric_id: {str(e)}")
            await session.rollback()
            raise
    
    async def _get_final_stock_price(
        self, 
        stock_code: str, 
        date: datetime, 
        session: AsyncSession
    ) -> Optional[StockPrice]:
        """최종 주가 조회"""
        query = select(StockPrice).where(
            and_(
                StockPrice.stock_code == stock_code,
                StockPrice.datetime <= date,
                StockPrice.interval_unit == "1d"
            )
        ).order_by(desc(StockPrice.datetime)).limit(1)
        
        result = await session.execute(query)
        return result.scalar_one_or_none()
    
    def _generate_cache_key(self, request: BacktestRequest) -> str:
        """캐시 키 생성"""
        cache_data = {
            'start': request.start.isoformat(),
            'end': request.end.isoformat(),
            'holdings': [(h.code, h.weight) for h in request.holdings],
            'initial_capital': str(request.initial_capital)
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"backtest:{hashlib.md5(cache_string.encode()).hexdigest()}"
    
    async def get_backtest_history(
        self, 
        session: AsyncSession,
        portfolio_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[PortfolioSnapshotResponse]:
        """백테스트 히스토리 조회"""
        query = select(PortfolioSnapshot).options(
            selectinload(PortfolioSnapshot.holdings)
        )
        
        # 필터 조건 추가
        conditions = []
        if portfolio_id:
            conditions.append(PortfolioSnapshot.portfolio_id == portfolio_id)
        if start_date:
            conditions.append(PortfolioSnapshot.created_at >= start_date)
        if end_date:
            conditions.append(PortfolioSnapshot.created_at <= end_date)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(desc(PortfolioSnapshot.created_at))
        query = query.offset(offset).limit(limit)
        
        result = await session.execute(query)
        snapshots = result.scalars().all()
        
        # 응답 객체로 변환
        response_snapshots = []
        for snapshot in snapshots:
            holdings = [
                HoldingSnapshotResponse(
                    id=h.id,
                    stock_id=h.stock_id,
                    weight=h.weight,
                    price=h.price,
                    quantity=h.quantity,
                    value=h.value,
                    recorded_at=h.recorded_at
                )
                for h in snapshot.holdings
            ]
            
            response_snapshots.append(PortfolioSnapshotResponse(
                id=snapshot.id,
                portfolio_id=snapshot.portfolio_id,
                # portfolio_name=snapshot.portfolio_name,  # DB에 컬럼이 없음
                base_value=snapshot.base_value,
                current_value=snapshot.current_value,
                start_at=snapshot.start_at,
                end_at=snapshot.end_at,
                created_at=snapshot.created_at,
                metric_id=snapshot.metric_id,
                execution_time=float(snapshot.execution_time) if snapshot.execution_time else None,
                holdings=holdings
            ))
        
        return response_snapshots

    async def _update_portfolio_snapshot_execution_time(
        self, 
        session: AsyncSession, 
        portfolio_snapshot_id: int, 
        execution_time: float
    ) -> None:
        """포트폴리오 스냅샷의 실행 시간 업데이트"""
        try:
            from sqlalchemy import update
            from app.models.stock import PortfolioSnapshot
            
            stmt = update(PortfolioSnapshot).where(
                PortfolioSnapshot.id == portfolio_snapshot_id
            ).values(execution_time=execution_time)
            
            await session.execute(stmt)
            await session.commit()
            
            logger.debug(f"Updated execution_time for portfolio_snapshot {portfolio_snapshot_id}: {execution_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Failed to update execution_time for portfolio_snapshot {portfolio_snapshot_id}: {str(e)}")
            await session.rollback()
            raise


# 백테스트 서비스는 의존성 주입으로 사용
