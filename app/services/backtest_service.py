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
    BacktestRequest, BacktestResponse, BacktestMetrics, BenchmarkMetrics,
    PortfolioSnapshotResponse, HoldingSnapshotResponse
)
from app.services.analysis_service import AnalysisService
from app.services.trading_rules_service import TradingRulesService
from app.services.benchmark_service import BenchmarkService
from app.services.risk_free_rate_service import RiskFreeRateService
from app.exceptions import MissingStockPriceDataException, InsufficientDataException
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestService:
    """최적화된 백테스트 서비스"""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    async def _retry_with_new_session_if_transaction_failed(
        self, 
        operation_func, 
        session: AsyncSession,
        service_name: str,
        **kwargs
    ):
        """
        트랜잭션 에러 감지시 새 독립 세션으로 재시도
        
        Args:
            operation_func: 실행할 함수 (session을 첫 번째 인자로 받음)
            session: 원본 세션 
            service_name: 로깅용 서비스명
            **kwargs: operation_func에 전달할 추가 인자들
        """
        try:
            return await operation_func(session, **kwargs)
        except Exception as e:
            if "InFailedSQLTransaction" in str(e) or "transaction is aborted" in str(e):
                logger.warning(f"Primary session failed for {service_name}, retrying with fresh session", error=str(e))
                # 새 독립 세션으로 재시도
                from app.models.database import AsyncSessionLocal
                async with AsyncSessionLocal() as new_session:
                    return await operation_func(new_session, **kwargs)
            else:
                raise
        
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
                       start=request.start.isoformat(), end=request.end.isoformat())
            
            # 2. 주가 데이터 조회 (캐싱 활용)
            if session is None:
                raise ValueError("Session is required for stock price data retrieval")
            stock_prices = await self._get_stock_prices_optimized(
                request, session
            )
            
            if stock_prices is None or stock_prices.empty:
                # 모든 종목의 데이터가 없는 경우
                missing_stocks = []
                for holding in request.holdings:
                    missing_stocks.append({
                        'stock_code': holding.code,
                        'start_date': request.start.strftime('%Y-%m-%d'),
                        'end_date': request.end.strftime('%Y-%m-%d'),
                        'available_date_range': None
                    })
                
                requested_period = f"{request.start.strftime('%Y-%m-%d')} ~ {request.end.strftime('%Y-%m-%d')}"
                raise MissingStockPriceDataException(
                    missing_stocks=missing_stocks,
                    requested_period=requested_period,
                    total_stocks=len(request.holdings)
                )
            
            # 3. 백테스트 실행
            portfolio_data, result_summary, execution_logs, final_status, benchmark_info = await self._execute_backtest(
                request, stock_prices, session
            )
            
            # 4. 성과 지표 계산
            metrics = await self._calculate_metrics(result_summary)
            
            # 5. 벤치마크 지표 계산
            benchmark_metrics = await self._calculate_benchmark_metrics(
                request, result_summary, session
            )
            
            # 6. 무위험수익률 정보 초기화
            risk_free_rate_info = None
            
            # 7. 포트폴리오 스냅샷 생성 (메모리만)
            portfolio_snapshot = self._create_portfolio_snapshot_response(
                request, portfolio_data, portfolio_id
            )
            
            execution_time = time.time() - start_time
            portfolio_snapshot.execution_time = execution_time
            
            logger.info(f"Backtest completed successfully", 
                       execution_time=f"{execution_time:.3f}s",
                       portfolio_id=portfolio_snapshot.id,
                       final_status=final_status,
                       execution_logs_count=len(execution_logs))
            
            return BacktestResponse(
                portfolio_snapshot=portfolio_snapshot,
                metrics=metrics,
                benchmark_metrics=benchmark_metrics,
                result_summary=result_summary,
                execution_time=execution_time,
                request_id=f"req_{int(time.time() * 1000)}",  # 타임스탬프 기반 요청 ID
                execution_logs=execution_logs,
                result_status=final_status,
                benchmark_info=benchmark_info,
                risk_free_rate_info=risk_free_rate_info
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
        
        # 각 종목별로 데이터 존재 여부 확인
        if not df.empty:
            available_stocks = set(df['stock_code'].unique())
            missing_stocks = []
            
            for stock_code in stock_codes:
                if stock_code not in available_stocks:
                    # 해당 종목의 사용 가능한 날짜 범위 확인
                    available_range_query = select(StockPrice).where(
                        and_(
                            StockPrice.stock_code == stock_code,
                            StockPrice.interval_unit == "1d"
                        )
                    ).order_by(StockPrice.datetime)
                    
                    range_result = await session.execute(available_range_query)
                    available_prices = range_result.scalars().all()
                    
                    available_range = None
                    if available_prices:
                        start_date = min(price.datetime for price in available_prices).strftime('%Y-%m-%d')
                        end_date = max(price.datetime for price in available_prices).strftime('%Y-%m-%d')
                        available_range = f"{start_date} ~ {end_date}"
                    
                    missing_stocks.append({
                        'stock_code': stock_code,
                        'start_date': request.start.strftime('%Y-%m-%d'),
                        'end_date': request.end.strftime('%Y-%m-%d'),
                        'available_date_range': available_range
                    })
            
            # 일부 종목의 데이터가 없는 경우 예외 발생
            if missing_stocks:
                requested_period = f"{request.start.strftime('%Y-%m-%d')} ~ {request.end.strftime('%Y-%m-%d')}"
                raise MissingStockPriceDataException(
                    missing_stocks=missing_stocks,
                    requested_period=requested_period,
                    total_stocks=len(stock_codes)
                )
        
        # 데이터 전처리 및 최적화
        df = await self._preprocess_stock_data(df)
        
        return df
    
    async def _preprocess_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """주가 데이터 전처리"""
        if df is None or df.empty:
            return df
        
        # 날짜별로 정렬
        df = df.sort_values(['datetime', 'stock_code'])
        
        # 결측값 처리 (전진 채우기)
        df['close_price'] = df.groupby('stock_code')['close_price'].ffill()
        
        # 수익률 계산 (벡터화 연산)
        df['returns'] = df.groupby('stock_code')['close_price'].pct_change()
        
        # 첫 번째 날의 수익률은 0으로 설정
        df['returns'] = df['returns'].fillna(0)
        
        return df
    
    async def _execute_backtest(
        self, 
        request: BacktestRequest, 
        stock_prices: pd.DataFrame,
        session: AsyncSession
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str, Optional[Dict[str, Any]]]:
        """백테스트 실행 (수량 기반 벡터화 연산 + 손절/익절 로직)"""
        
        # 포트폴리오 보유 수량 딕셔너리 생성
        quantities = {holding.code: holding.quantity for holding in request.holdings}
        
        # 평균 매수가 저장 (개별 수익률 계산용)
        avg_prices = {}
        for holding in request.holdings:
            if holding.avg_price:
                avg_prices[holding.code] = float(holding.avg_price)
            else:
                # 평균 매수가가 없으면 첫날 가격으로 설정
                first_price = stock_prices[
                    (stock_prices['stock_code'] == holding.code) & 
                    (stock_prices['datetime'] == stock_prices['datetime'].min())
                ]['close_price'].iloc[0]
                avg_prices[holding.code] = float(first_price)
        
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
        
        # 공통 종목 필터링
        common_stocks = price_pivot.columns.intersection(quantities.keys())
        price_data = price_pivot[common_stocks]
        
        # 벤치마크 수익률 조회 (베타 계산용)
        benchmark_returns, benchmark_info = await self._get_benchmark_returns_for_period(
            request, session
        )
        
        # 무위험 수익률 조회
        risk_free_rates, risk_free_rate_info = await self._get_risk_free_rate_for_period(
            request, session
        )
        
        # 중요 데이터 조회 실패 시 에러 처리
        critical_data_failure = []
        if benchmark_returns is None or benchmark_returns.empty:
            critical_data_failure.append("benchmark returns")
        if risk_free_rates is None or risk_free_rates.empty:
            critical_data_failure.append("risk-free rates")
            
        if critical_data_failure:
            logger.error(f"Critical data retrieval failure: {', '.join(critical_data_failure)}")
            raise Exception(f"Failed to retrieve critical data: {', '.join(critical_data_failure)}. This may be due to database transaction errors or missing data.")
        
        # 손절/익절 서비스 초기화
        trading_rules_service = TradingRulesService() if request.rules else None
        
        # 일별 백테스트 실행
        portfolio_data = []
        result_summary = []
        execution_logs = []
        final_status = "COMPLETED"
        
        for i, date in enumerate(price_pivot.index):
            # 현재 포트폴리오 가치 계산
            current_portfolio_value = 0
            individual_values = {}
            individual_prices = {}
            individual_returns = {}
            
            for stock_code in common_stocks:
                if pd.notna(price_data.loc[date, stock_code]):
                    stock_price = price_data.loc[date, stock_code]
                    stock_quantity = quantities[stock_code]
                    stock_value = stock_price * stock_quantity
                    
                    current_portfolio_value += stock_value
                    individual_values[stock_code] = stock_value
                    individual_prices[stock_code] = stock_price
                    
                    # 개별 종목 수익률 계산 (평균 매수가 대비)
                    if stock_code in avg_prices:
                        stock_return = (stock_price - avg_prices[stock_code]) / avg_prices[stock_code]
                        individual_returns[stock_code] = stock_return
            
            # 일별 수익률 계산
            prev_value = portfolio_data[-1]['portfolio_value'] if portfolio_data else current_portfolio_value
            daily_return = (current_portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
            
            # 포트폴리오 데이터 추가
            portfolio_data.append({
                'datetime': date,
                'portfolio_value': current_portfolio_value,
                'daily_return': daily_return,
                'quantities': quantities.copy()
            })
            
            # 손절/익절 규칙 체크
            if trading_rules_service:
                should_execute, daily_logs, status = await trading_rules_service.check_trading_rules(
                    date=date,
                    portfolio_data=portfolio_data,
                    individual_values=individual_values,
                    individual_prices=individual_prices,
                    individual_returns=individual_returns,
                    quantities=quantities,
                    benchmark_returns=benchmark_returns,
                    rules=request.rules
                )
                
                if should_execute:
                    execution_logs.extend(daily_logs)
                    final_status = status
                    
                    # 포트폴리오 청산 (모든 수량을 0으로 설정)
                    quantities = {code: 0 for code in quantities.keys()}
                    
                    # 청산된 상태로 마지막 데이터 업데이트
                    portfolio_data[-1]['status'] = 'LIQUIDATED'
                    portfolio_data[-1]['liquidation_reason'] = 'TRADING_RULES'
                    
                    break
            
            # 종목별 일별 데이터 생성
            daily_stocks = []
            for stock_code in common_stocks:
                if pd.notna(price_data.loc[date, stock_code]):
                    stock_price = price_data.loc[date, stock_code]
                    stock_return = returns_pivot.loc[date, stock_code] if pd.notna(returns_pivot.loc[date, stock_code]) else 0.0
                    stock_quantity = quantities[stock_code]
                    stock_value = stock_price * stock_quantity
                    stock_weight = stock_value / current_portfolio_value if current_portfolio_value > 0 else 0.0
                    
                    # 포트폴리오 수익률 기여도 계산
                    portfolio_contribution = stock_return * stock_weight if stock_weight > 0 else 0.0
                    
                    daily_stocks.append({
                        'stock_code': stock_code,
                        'date': date.isoformat(),
                        'close_price': float(stock_price),
                        'daily_return': float(stock_return),
                        'portfolio_weight': float(stock_weight),
                        'portfolio_contribution': float(portfolio_contribution),
                        'quantity': stock_quantity,
                        'avg_price': avg_prices.get(stock_code, 0.0)
                    })
            
            # 결과 요약 데이터
            summary_item = {
                'date': date.isoformat(),
                'stocks': daily_stocks,
                'portfolio_value': current_portfolio_value,
                'quantities': quantities.copy()
            }
            
            result_summary.append(summary_item)
        
        return portfolio_data, result_summary, execution_logs, final_status, benchmark_info
    
    async def _execute_benchmark_operations(
        self,
        session: AsyncSession,
        benchmark_service: BenchmarkService,
        benchmark_code: str,
        start_date: datetime,
        end_date: datetime
    ):
        """벤치마크 관련 연산 수행"""
        # 벤치마크 수익률 조회
        benchmark_returns = await benchmark_service.get_benchmark_returns(
            benchmark_code=benchmark_code,
            start_date=start_date,
            end_date=end_date,
            session=session
        )
        
        # 벤치마크 정보 조회
        benchmark_info = await benchmark_service.get_benchmark_info(
            benchmark_code=benchmark_code,
            session=session
        )
        
        return benchmark_returns, benchmark_info

    async def _get_benchmark_returns_for_period(
        self,
        request: BacktestRequest,
        session: AsyncSession
    ) -> Tuple[Optional[pd.Series], Optional[Dict[str, Any]]]:
        """벤치마크 수익률 조회 - 트랜잭션 에러 시 독립 세션으로 재시도"""
        try:
            benchmark_service = BenchmarkService()
            
            # 사용자가 지정한 벤치마크 코드 사용
            benchmark_code = request.benchmark_code or "KOSPI"
            
            # 트랜잭션 에러 감지 시 새 독립 세션으로 재시도
            benchmark_returns, benchmark_info = await self._retry_with_new_session_if_transaction_failed(
                operation_func=self._execute_benchmark_operations,
                session=session,
                service_name="benchmark",
                benchmark_service=benchmark_service,
                benchmark_code=benchmark_code,
                start_date=request.start,
                end_date=request.end
            )
            
            logger.info(
                "Benchmark returns retrieved for backtest",
                benchmark_code=benchmark_code,
                data_points=len(benchmark_returns),
                portfolio_size=len(request.holdings)
            )
            
            return benchmark_returns, benchmark_info
            
        except Exception as e:
            logger.error(f"Failed to get benchmark returns: {str(e)}")
            # 데이터베이스 에러를 상위로 전파하여 백테스트 실패 처리
            raise Exception(f"Database error during benchmark retrieval: {str(e)}")
    
    async def _get_risk_free_rate_for_period(
        self,
        request: BacktestRequest,
        session: AsyncSession
    ) -> Tuple[Optional[pd.Series], Optional[Dict[str, Any]]]:
        """무위험 수익률 조회 - 트랜잭션 에러 시 독립 세션으로 재시도"""
        try:
            risk_free_rate_service = RiskFreeRateService()
            
            # 사용자가 직접 지정한 경우
            if request.risk_free_rate is not None:
                # 고정 금리를 일별 시계열로 변환
                daily_rate = request.risk_free_rate / 365.0 / 100.0
                dates = pd.date_range(start=request.start, end=request.end, freq='D')
                risk_free_rates = pd.Series([daily_rate] * len(dates), index=dates)
                
                risk_free_rate_info = {
                    'rate_type': 'USER_PROVIDED',
                    'annual_rate': request.risk_free_rate,
                    'daily_rate': daily_rate,
                    'selection_reason': 'user_specified'
                }
                
                logger.info(
                    "Using user-provided risk-free rate",
                    annual_rate=request.risk_free_rate
                )
                
                return risk_free_rates, risk_free_rate_info
            
            # 트랜잭션 에러 감지 시 새 독립 세션으로 전체 재실행
            try:
                # 자동 결정
                rate_type, decision_info = await risk_free_rate_service.determine_risk_free_rate_type(
                    start_date=request.start,
                    end_date=request.end,
                    session=session
                )
                
                # 무위험 수익률 조회
                risk_free_rates = await risk_free_rate_service.get_risk_free_rate(
                    rate_type=rate_type,
                    start_date=request.start,
                    end_date=request.end,
                    session=session
                )
                
                # 무위험 수익률 정보 조회
                rate_info = await risk_free_rate_service.get_rate_info(
                    rate_type=rate_type,
                    session=session
                )
                
            except Exception as e:
                if "InFailedSQLTransaction" in str(e) or "transaction is aborted" in str(e):
                    logger.warning("Primary session failed, retrying with fresh session", error=str(e))
                    # 새 독립 세션으로 전체 재시도
                    from app.models.database import AsyncSessionLocal
                    async with AsyncSessionLocal() as new_session:
                        rate_type, decision_info = await risk_free_rate_service.determine_risk_free_rate_type(
                            start_date=request.start,
                            end_date=request.end,
                            session=new_session
                        )
                        
                        risk_free_rates = await risk_free_rate_service.get_risk_free_rate(
                            rate_type=rate_type,
                            start_date=request.start,
                            end_date=request.end,
                            session=new_session
                        )
                        
                        rate_info = await risk_free_rate_service.get_rate_info(
                            rate_type=rate_type,
                            session=new_session
                        )
                else:
                    raise
            
            # 정보 통합
            risk_free_rate_info = {
                'rate_type': rate_type,
                'decision_info': decision_info,
                'rate_info': rate_info,
                'data_points': len(risk_free_rates),
                'avg_annual_rate': float(risk_free_rates.mean() * 365 * 100) if not risk_free_rates.empty else 0.0
            }
            
            logger.info(
                "Risk-free rate retrieved for backtest",
                rate_type=rate_type,
                data_points=len(risk_free_rates),
                avg_annual_rate=risk_free_rate_info['avg_annual_rate']
            )
            
            return risk_free_rates, risk_free_rate_info
            
        except Exception as e:
            logger.error(f"Failed to get risk-free rate: {str(e)}")
            # 데이터베이스 에러를 상위로 전파하여 백테스트 실패 처리
            raise Exception(f"Database error during risk-free rate retrieval: {str(e)}")
    
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
        
        # 샤프 비율 (무위험 수익률 고려)
        # TODO: 무위험 수익률을 매개변수로 받아서 계산하도록 수정 필요
        risk_free_rate = 0.0  # 임시로 0% 사용
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
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
        """포트폴리오 스냅샷 응답 생성"""
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

    async def analyze_backtest_performance(
        self,
        portfolio_returns: pd.Series,
        benchmark_code: str,
        session: AsyncSession,
        risk_free_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """백테스트 결과에 대한 고도화된 성과 분석"""
        
        from app.repositories.benchmark_repository import BenchmarkRepository
        from app.repositories.risk_free_rate_repository import RiskFreeRateRepository
        
        # Repository 초기화
        benchmark_repo = BenchmarkRepository(session)
        risk_free_repo = RiskFreeRateRepository(session)
        
        # 분석 기간
        start_date = portfolio_returns.index[0]
        end_date = portfolio_returns.index[-1]
        
        # 벤치마크 수익률 조회
        benchmark_returns = await benchmark_repo.get_benchmark_returns_series(
            benchmark_code, start_date, end_date
        )
        
        # 무위험수익률 조회
        if risk_free_rate is None:
            risk_free_rate = await risk_free_repo.get_risk_free_rate("CD91", end_date) or 0.0
        
        # 시계열 동기화
        if not benchmark_returns.empty:
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_synced = portfolio_returns.loc[common_dates]
            benchmark_synced = benchmark_returns.loc[common_dates]
        else:
            portfolio_synced = portfolio_returns
            benchmark_synced = pd.Series(dtype=float)
        
        # AnalysisService의 고급 계산 메서드 활용
        analysis_service = AnalysisService()
        
        # 기본 통계
        annual_return = portfolio_synced.mean() * 252
        annual_volatility = portfolio_synced.std() * np.sqrt(252)
        
        # 고급 지표들
        sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0
        
        max_dd = analysis_service._calculate_max_drawdown(portfolio_synced)
        downside_dev = analysis_service._calculate_downside_deviation(portfolio_synced)
        sortino = (annual_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0
        
        metrics = {
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_volatility),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'calmar_ratio': float(calmar),
            'max_drawdown': float(max_dd),
            'downside_deviation': float(downside_dev),
            'risk_free_rate_used': float(risk_free_rate)
        }
        
        # 벤치마크 비교 (있는 경우)
        if not benchmark_synced.empty:
            beta, alpha, correlation = analysis_service._calculate_beta_alpha(
                portfolio_synced, benchmark_synced, risk_free_rate
            )
            tracking_error = analysis_service._calculate_tracking_error(portfolio_synced, benchmark_synced)
            
            benchmark_annual_return = benchmark_synced.mean() * 252
            excess_return = annual_return - benchmark_annual_return
            
            metrics.update({
                'beta': float(beta),
                'alpha': float(alpha),
                'jensen_alpha': float(alpha),  # 이 경우 alpha와 jensen_alpha가 동일
                'tracking_error': float(tracking_error),
                'correlation_with_benchmark': float(correlation),
                'benchmark_annual_return': float(benchmark_annual_return),
                'excess_return': float(excess_return)
            })
        
        return metrics
    
    async def _calculate_benchmark_metrics(
        self,
        request: BacktestRequest,
        result_summary: List[Dict[str, Any]],
        session: AsyncSession
    ) -> Optional[BenchmarkMetrics]:
        """벤치마크 성과 지표 계산"""
        try:
            benchmark_service = BenchmarkService()
            
            # 벤치마크 코드 가져오기
            benchmark_code = request.benchmark_code or "KOSPI"
            
            # 벤치마크 수익률 조회
            benchmark_returns = await benchmark_service.get_benchmark_returns(
                benchmark_code=benchmark_code,
                start_date=request.start,
                end_date=request.end,
                session=session
            )
            
            if benchmark_returns is None or benchmark_returns.empty:
                return None
            
            # 벤치마크 기본 지표 계산
            benchmark_total_return = float((benchmark_returns.cumprod()[-1] - 1.0) * 100)
            benchmark_volatility = float(benchmark_returns.std() * np.sqrt(252.0) * 100)
            
            # 벤치마크 최고가/최저가 조회
            benchmark_prices = await self._get_benchmark_prices_for_period(
                benchmark_code, request.start, request.end, session
            )
            
            benchmark_max_price = 0.0
            benchmark_min_price = 0.0
            if benchmark_prices is not None and not benchmark_prices.empty:
                benchmark_max_price = float(benchmark_prices.max())
                benchmark_min_price = float(benchmark_prices.min())
            
            # 포트폴리오 수익률 계산
            portfolio_returns = []
            for rs in result_summary:
                daily_return = sum(stock['portfolio_contribution'] for stock in rs['stocks'])
                portfolio_returns.append(daily_return)
            
            portfolio_returns = np.array(portfolio_returns)
            
            # 알파 계산 (포트폴리오 수익률 - 벤치마크 수익률)
            if len(portfolio_returns) == len(benchmark_returns):
                alpha = float((portfolio_returns.mean() - benchmark_returns.mean()) * 252.0 * 100)
            else:
                alpha = 0.0
            
            # 벤치마크 일일 평균 수익률
            benchmark_daily_average = float(benchmark_returns.mean() * 100)
            
            return BenchmarkMetrics(
                benchmark_total_return=benchmark_total_return,
                benchmark_volatility=benchmark_volatility,
                benchmark_max_price=benchmark_max_price,
                benchmark_min_price=benchmark_min_price,
                alpha=alpha,
                benchmark_daily_average=benchmark_daily_average
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate benchmark metrics: {str(e)}")
            return None
    
    async def _get_benchmark_prices_for_period(
        self,
        benchmark_code: str,
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession
    ) -> Optional[pd.Series]:
        """벤치마크 지수 가격 조회"""
        try:
            from app.models.stock import BenchmarkPrice
            from sqlalchemy import select
            
            query = select(BenchmarkPrice).where(
                and_(
                    BenchmarkPrice.index_code == benchmark_code,
                    BenchmarkPrice.datetime >= start_date,
                    BenchmarkPrice.datetime <= end_date
                )
            ).order_by(BenchmarkPrice.datetime)
            
            result = await session.execute(query)
            benchmark_prices = result.scalars().all()
            
            if not benchmark_prices:
                return None
            
            dates = []
            prices = []
            for price in benchmark_prices:
                dates.append(price.datetime)
                prices.append(float(price.close_price))
            
            return pd.Series(prices, index=dates)
            
        except Exception as e:
            logger.error(f"Failed to get benchmark prices: {str(e)}")
            return None

