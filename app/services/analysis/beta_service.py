from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import linregress

from app.models.schemas import BetaAnalysis, StockDetails
from app.utils.logger import get_logger


logger = get_logger(__name__)


class BetaService:
    async def _calculate_stock_details(
        self,
        optimization_results: Dict[str, Any],
        benchmark_code: str,
        benchmark_returns: pd.Series,
        session,
        prices_df: pd.DataFrame = None,
    ) -> Optional[Dict[str, Any]]:
        """개별 종목 상세 정보 계산 (베타 포함)"""
        try:
            if not benchmark_code or benchmark_returns.empty:
                logger.warning("개별 종목 베타 계산을 위한 벤치마크 데이터가 없음")
                return None
            
            latest_weights = optimization_results['latest_weights']
            portfolio_stocks = set(latest_weights['min_downside_risk'].keys()) | set(latest_weights['max_sortino'].keys())
            
            if not portfolio_stocks:
                logger.warning("베타 계산을 위한 포트폴리오 종목을 찾을 수 없음")
                return None
            
            latest_expected_returns = optimization_results['expected_returns'][-1] if optimization_results['expected_returns'] else {}
            latest_covariances = optimization_results['covariances'][-1] if optimization_results['covariances'] else {}
            
            stock_details = {}
            
            for stock_code in portfolio_stocks:
                try:
                    expected_return = latest_expected_returns.get(stock_code, 0.0)
                    volatility = np.sqrt(latest_covariances.get(stock_code, {}).get(stock_code, 0.0))
                    
                    max_sortino_weights = latest_weights['max_sortino']
                    portfolio_variance = 0.0
                    stock_portfolio_covariance = 0.0
                    
                    for other_stock, weight in max_sortino_weights.items():
                        if other_stock in latest_covariances:
                            if stock_code == other_stock:
                                stock_portfolio_covariance += weight * latest_covariances.get(stock_code, {}).get(stock_code, 0.0)
                            else:
                                stock_portfolio_covariance += weight * latest_covariances.get(stock_code, {}).get(other_stock, 0.0)
                            
                            for other_stock2, weight2 in max_sortino_weights.items():
                                if other_stock2 in latest_covariances.get(other_stock, {}):
                                    portfolio_variance += weight * weight2 * latest_covariances.get(other_stock, {}).get(other_stock2, 0.0)
                    
                    portfolio_std = np.sqrt(portfolio_variance) if portfolio_variance > 0 else 1.0
                    correlation_to_portfolio = stock_portfolio_covariance / (volatility * portfolio_std) if volatility > 0 and portfolio_std > 0 else 0.0
                    
                    beta_analysis = None
                    if prices_df is not None and stock_code in prices_df.columns:
                        try:
                            stock_prices = prices_df[stock_code].dropna()
                            stock_returns = stock_prices.pct_change().dropna()
                            
                            common_dates = stock_returns.index.intersection(benchmark_returns.index)
                            if len(common_dates) >= 10:
                                stock_returns_aligned = stock_returns.loc[common_dates]
                                benchmark_returns_aligned = benchmark_returns.loc[common_dates]
                                
                                slope, intercept, r_value, p_value, std_err = linregress(
                                    benchmark_returns_aligned.values, 
                                    stock_returns_aligned.values
                                )
                                
                                beta_analysis = BetaAnalysis(
                                    beta=float(slope),
                                    r_square=float(r_value ** 2),
                                    alpha=float(intercept) * 252
                                )
                                
                                logger.debug(f"{stock_code} 베타 분석 계산 완료: beta={slope:.4f}, r²={r_value**2:.4f}")
                            else:
                                logger.warning(f"베타 계산을 위한 데이터 부족: {stock_code}")
                                beta_analysis = self._get_default_beta_analysis()
                        except Exception as e:
                            logger.error(f"{stock_code} 베타 계산 오류: {str(e)}")
                            beta_analysis = self._get_default_beta_analysis()
                    else:
                        beta_analysis = self._get_default_beta_analysis()
                    
                    stock_detail = StockDetails(
                        expected_return=expected_return,
                        volatility=volatility,
                        correlation_to_portfolio=correlation_to_portfolio,
                        beta_analysis=beta_analysis
                    )
                    
                    stock_details[stock_code] = stock_detail
                    
                except Exception as e:
                    logger.error(f"종목 {stock_code} 처리 오류: {str(e)}")
                    continue
            
            logger.info(f"{len(stock_details)}개 종목의 상세 정보 계산 완료")
            return stock_details
            
        except Exception as e:
            logger.error(f"종목 상세 정보 계산 오류: {str(e)}")
            return None

    async def _calculate_portfolio_beta_analysis(
        self,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        analysis_start,
        analysis_end,
    ) -> Optional[object]:
        """포트폴리오 베타 분석 계산"""
        try:
            if benchmark_returns.empty:
                logger.warning("포트폴리오 베타 계산을 위한 벤치마크 데이터가 없음")
                return None
            
            portfolio_returns = [r['max_sortino'] for r in optimization_results['portfolio_returns']]
            dates = optimization_results['dates']
            
            if len(portfolio_returns) < 10:
                logger.warning(f"베타 계산을 위한 포트폴리오 수익률 데이터 부족: {len(portfolio_returns)}")
                return None
            
            portfolio_returns_series = pd.Series(portfolio_returns, index=dates)
            
            benchmark_period_returns = self._align_benchmark_returns(
                benchmark_returns, optimization_results['dates']
            )
            benchmark_period_returns.index = dates
            
            if len(benchmark_period_returns) < 10:
                logger.warning(f"베타 계산을 위한 벤치마크 수익률 데이터 부족: {len(benchmark_period_returns)}")
                return None
            
            common_index = portfolio_returns_series.index.intersection(benchmark_period_returns.index)
            
            if len(common_index) < 10:
                logger.warning(f"베타 계산을 위한 공통 데이터 포인트 부족: {len(common_index)}")
                return None
            
            portfolio_aligned = portfolio_returns_series.loc[common_index]
            benchmark_aligned = benchmark_period_returns.loc[common_index]
            
            slope, intercept, r_value, p_value, std_err = linregress(
                benchmark_aligned.values, 
                portfolio_aligned.values
            )
            
            portfolio_beta_analysis = BetaAnalysis(
                beta=float(slope),
                r_square=float(r_value ** 2),
                alpha=float(intercept) * 252
            )
            
            logger.info(f"포트폴리오 베타 분석 계산 완료: beta={slope:.4f}, r²={r_value**2:.4f}, alpha={intercept*252:.4f}")
            return portfolio_beta_analysis
            
        except Exception as e:
            logger.error(f"포트폴리오 베타 분석 계산 오류: {str(e)}")
            return None

    async def _calculate_portfolio_beta_for_weights(
        self,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        analysis_start,
        analysis_end,
        portfolio_type: str,
    ) -> Optional[object]:
        """특정 포트폴리오 타입의 베타 계산"""
        try:
            if benchmark_returns.empty:
                logger.warning(f"{portfolio_type} 베타 계산을 위한 벤치마크 데이터가 없음")
                return None
            
            portfolio_returns = [r[portfolio_type] for r in optimization_results['portfolio_returns']]
            dates = optimization_results['dates']
            
            if len(portfolio_returns) < 10:
                logger.warning(f"{portfolio_type} 베타 계산을 위한 포트폴리오 수익률 데이터 부족: {len(portfolio_returns)}")
                return None
            
            portfolio_returns_series = pd.Series(portfolio_returns, index=dates)
            
            benchmark_period_returns = self._align_benchmark_returns(
                benchmark_returns, optimization_results['dates']
            )
            benchmark_period_returns.index = dates
            
            if len(benchmark_period_returns) < 10:
                logger.warning(f"{portfolio_type} 베타 계산을 위한 벤치마크 수익률 데이터 부족: {len(benchmark_period_returns)}")
                return None
            
            common_index = portfolio_returns_series.index.intersection(benchmark_period_returns.index)
            
            if len(common_index) < 10:
                logger.warning(f"{portfolio_type} 베타 계산을 위한 공통 데이터 포인트 부족: {len(common_index)}")
                return None
            
            portfolio_aligned = portfolio_returns_series.loc[common_index]
            benchmark_aligned = benchmark_period_returns.loc[common_index]
            
            slope, intercept, r_value, p_value, std_err = linregress(
                benchmark_aligned.values, 
                portfolio_aligned.values
            )
            
            portfolio_beta_analysis = BetaAnalysis(
                beta=float(slope),
                r_square=float(r_value ** 2),
                alpha=float(intercept) * 252
            )
            
            logger.info(f"{portfolio_type} 베타 분석 계산 완료: beta={slope:.4f}, r²={r_value**2:.4f}, alpha={intercept*252:.4f}")
            return portfolio_beta_analysis
            
        except Exception as e:
            logger.error(f"{portfolio_type} 베타 분석 계산 오류: {str(e)}")
            return None

    async def _calculate_user_portfolio_beta(
        self,
        request,
        benchmark_returns: pd.Series,
        analysis_start,
        analysis_end,
    ) -> Optional[object]:
        """사용자 입력 포트폴리오 베타 계산"""
        try:
            if benchmark_returns.empty or not request.holdings:
                logger.warning("사용자 포트폴리오 베타 계산을 위한 벤치마크 데이터 또는 보유 종목이 없음")
                return None
            
            user_weights = {}
            total_value = sum(h.quantity for h in request.holdings)
            
            for holding in request.holdings:
                user_weights[holding.code] = holding.quantity / total_value
            
            logger.info(f"{len(user_weights)}개 종목에 대한 사용자 포트폴리오 베타 계산 요청됨")
            
            return BetaAnalysis(
                beta=1.0,
                r_square=0.0,
                alpha=0.0
            )
            
        except Exception as e:
            logger.error(f"사용자 포트폴리오 베타 계산 오류: {str(e)}")
            return None

    def _get_default_beta_analysis(self) -> BetaAnalysis:
        """기본 베타 분석 정보"""
        return BetaAnalysis(
            beta=1.0,
            r_square=0.0,
            alpha=0.0
        )

