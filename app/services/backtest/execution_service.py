"""백테스트 실행 로직 서비스"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import BacktestRequest
from app.services.backtest.trading_rules_service import TradingRulesService
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
        
        quantities = {holding.code: holding.quantity for holding in request.holdings}
        
        avg_prices = {}
        for holding in request.holdings:
            if holding.avg_price:
                avg_prices[holding.code] = float(holding.avg_price)
            else:
                stock_data = stock_prices[stock_prices['stock_code'] == holding.code]
                if not stock_data.empty:
                    first_price_data = stock_data.sort_values('datetime').iloc[0]
                    avg_prices[holding.code] = float(first_price_data['close_price'])
                    logger.info(f"Using first trading day price for {holding.code}: {avg_prices[holding.code]:.2f} "
                              f"on {first_price_data['datetime'].strftime('%Y-%m-%d')}")
                else:
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
        
        price_pivot = stock_prices.pivot_table(
            index='datetime', 
            columns='stock_code', 
            values='close_price',
            aggfunc='first'
        )
        
        if price_pivot.empty:
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
        
        common_stocks = price_pivot.columns.intersection(quantities.keys())
        
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
        
        benchmark_returns, benchmark_info = await self._get_benchmark_returns_for_period(
            request, session
        )
        
        risk_free_rates, risk_free_rate_info = await self._get_risk_free_rate_for_period(
            request, session
        )
        
        critical_data_failure = []
        if benchmark_returns is None or benchmark_returns.empty:
            critical_data_failure.append("benchmark returns")
        if risk_free_rates is None or risk_free_rates.empty:
            critical_data_failure.append("risk-free rates")
            
        if critical_data_failure:
            logger.error(f"Critical data retrieval failure: {', '.join(critical_data_failure)}")
            raise Exception(f"Failed to retrieve critical data: {', '.join(critical_data_failure)}. This may be due to database transaction errors or missing data.")
        
        trading_rules_service = TradingRulesService() if request.rules else None
        
        portfolio_data = []
        result_summary = []
        execution_logs = []
        final_status = "COMPLETED"
        
        for i, date in enumerate(price_pivot.index):
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
                    
                    if stock_code in avg_prices:
                        stock_return = (stock_price - avg_prices[stock_code]) / avg_prices[stock_code]
                        individual_returns[stock_code] = stock_return
            
            prev_value = portfolio_data[-1]['portfolio_value'] if portfolio_data else current_portfolio_value
            daily_return = (current_portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
            
            portfolio_data.append({
                'datetime': date,
                'portfolio_value': current_portfolio_value,
                'daily_return': daily_return,
                'quantities': quantities.copy()
            })
            
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
                    
                    quantities = {code: 0 for code in quantities.keys()}
                    
                    portfolio_data[-1]['status'] = 'LIQUIDATED'
                    portfolio_data[-1]['liquidation_reason'] = 'TRADING_RULES'
                    
                    break
            
            daily_stocks = []
            for stock_code in common_stocks:
                if pd.notna(price_data.loc[date, stock_code]):
                    stock_price = price_data.loc[date, stock_code]
                    stock_quantity = quantities[stock_code]
                    stock_value = stock_price * stock_quantity
                    stock_weight = stock_value / current_portfolio_value if current_portfolio_value > 0 else 0.0
                    
                    if stock_code in avg_prices:
                        stock_return = (stock_price - avg_prices[stock_code]) / avg_prices[stock_code]
                    else:
                        stock_return = 0.0
                    
                    portfolio_contribution = daily_return * stock_weight if stock_weight > 0 else 0.0
                    
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
            
            summary_item = {
                'date': date.isoformat(),
                'stocks': daily_stocks,
                'portfolio_value': current_portfolio_value,
                'quantities': quantities.copy()
            }
            
            result_summary.append(summary_item)
        
        if portfolio_data:
            actual_start = portfolio_data[0]['datetime']
            actual_end = portfolio_data[-1]['datetime']
        else:
            actual_start = request.start
            actual_end = request.end
        
        return portfolio_data, result_summary, execution_logs, final_status, benchmark_info, actual_start, actual_end

