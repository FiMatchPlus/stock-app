"""백테스트 실행 로직 서비스"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import BacktestRequest
from app.services.trading_rules_service import TradingRulesService
from app.exceptions import MissingStockPriceDataException
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestExecutionService:
    """백테스트 실행 로직 서비스"""
    
    async def _execute_backtest(
        self, 
        request: BacktestRequest, 
        stock_prices: pd.DataFrame,
        session: AsyncSession
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str, Optional[Dict[str, Any]], datetime, datetime]:
        """백테스트 실행 (수량 기반 벡터화 연산 + 손절/익절 로직)"""
        
        # 포트폴리오 보유 수량 딕셔너리 생성
        quantities = {holding.code: holding.quantity for holding in request.holdings}
        
        # 평균 매수가 저장 (개별 수익률 계산용)
        avg_prices = {}
        for holding in request.holdings:
            if holding.avg_price:
                avg_prices[holding.code] = float(holding.avg_price)
            else:
                # 평균 매수가가 없으면 첫 번째 거래일 가격으로 설정
                stock_data = stock_prices[stock_prices['stock_code'] == holding.code]
                if not stock_data.empty:
                    # 해당 종목의 첫 번째 거래일 가격 (datetime 기준 정렬 후 첫 번째)
                    first_price_data = stock_data.sort_values('datetime').iloc[0]
                    avg_prices[holding.code] = float(first_price_data['close_price'])
                    logger.info(f"Using first trading day price for {holding.code}: {avg_prices[holding.code]:.2f} "
                              f"on {first_price_data['datetime'].strftime('%Y-%m-%d')}")
                else:
                    # 해당 종목의 데이터가 없는 경우 에러 발생
                    logger.error(f"No stock price data found for {holding.code}")
                    raise MissingStockPriceDataException(
                        missing_stocks=[{
                            'stock_code': holding.code,
                            'start_date': request.start.strftime('%Y-%m-%d'),
                            'end_date': request.end.strftime('%Y-%m-%d'),
                            'available_date_range': None
                        }],
                        requested_period=f"{request.start.strftime('%Y-%m-%d')} ~ {request.end.strftime('%Y-%m-%d')}",
                        total_stocks=1
                    )
        
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
        
        # 피벗 테이블이 비어있는지 확인
        if price_pivot.empty or returns_pivot.empty:
            logger.error("Empty pivot tables created from stock prices data")
            raise MissingStockPriceDataException(
                missing_stocks=[{
                    'stock_code': 'ALL',
                    'start_date': request.start.strftime('%Y-%m-%d'),
                    'end_date': request.end.strftime('%Y-%m-%d'),
                    'available_date_range': None
                }],
                requested_period=f"{request.start.strftime('%Y-%m-%d')} ~ {request.end.strftime('%Y-%m-%d')}",
                total_stocks=len(request.holdings)
            )
        
        # 공통 종목 필터링
        common_stocks = price_pivot.columns.intersection(quantities.keys())
        
        # 공통 종목이 없는 경우 에러
        if len(common_stocks) == 0:
            logger.error("No common stocks found between portfolio and price data")
            raise MissingStockPriceDataException(
                missing_stocks=[{
                    'stock_code': 'ALL',
                    'start_date': request.start.strftime('%Y-%m-%d'),
                    'end_date': request.end.strftime('%Y-%m-%d'),
                    'available_date_range': None
                }],
                requested_period=f"{request.start.strftime('%Y-%m-%d')} ~ {request.end.strftime('%Y-%m-%d')}",
                total_stocks=len(request.holdings)
            )
        
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
        
        # 실제 거래일 범위 추출
        if portfolio_data:
            actual_start = portfolio_data[0]['datetime']
            actual_end = portfolio_data[-1]['datetime']
        else:
            actual_start = request.start
            actual_end = request.end
        
        return portfolio_data, result_summary, execution_logs, final_status, benchmark_info, actual_start, actual_end

