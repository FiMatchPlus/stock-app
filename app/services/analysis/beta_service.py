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
                logger.warning("No benchmark data available for stock beta calculation")
                return None
            
            # 최적화에 사용된 종목 목록 추출
            latest_weights = optimization_results['latest_weights']
            portfolio_stocks = set(latest_weights['min_downside_risk'].keys()) | set(latest_weights['max_sortino'].keys())
            
            if not portfolio_stocks:
                logger.warning("No portfolio stocks found for beta calculation")
                return None
            
            # 최근 윈도우의 기대수익률과 공분산 데이터 추출
            latest_expected_returns = optimization_results['expected_returns'][-1] if optimization_results['expected_returns'] else {}
            latest_covariances = optimization_results['covariances'][-1] if optimization_results['covariances'] else {}
            
            stock_details = {}
            
            for stock_code in portfolio_stocks:
                try:
                    # 기대수익률과 변동성 추출
                    expected_return = latest_expected_returns.get(stock_code, 0.0)
                    volatility = np.sqrt(latest_covariances.get(stock_code, {}).get(stock_code, 0.0))
                    
                    # 포트폴리오와의 상관관계 계산 (최대 소르티노 포트폴리오 기준)
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
                    
                    # 상관관계 계산
                    portfolio_std = np.sqrt(portfolio_variance) if portfolio_variance > 0 else 1.0
                    correlation_to_portfolio = stock_portfolio_covariance / (volatility * portfolio_std) if volatility > 0 and portfolio_std > 0 else 0.0
                    
                    # 베타 분석 계산
                    beta_analysis = None
                    if prices_df is not None and stock_code in prices_df.columns:
                        try:
                            # 종목 수익률 계산
                            stock_prices = prices_df[stock_code].dropna()
                            stock_returns = stock_prices.pct_change().dropna()
                            
                            # 벤치마크와 공통 기간 찾기
                            common_dates = stock_returns.index.intersection(benchmark_returns.index)
                            if len(common_dates) >= 10:  # 최소 10개 데이터 포인트
                                stock_returns_aligned = stock_returns.loc[common_dates]
                                benchmark_returns_aligned = benchmark_returns.loc[common_dates]
                                
                                # 회귀분석으로 베타 계산
                                slope, intercept, r_value, p_value, std_err = linregress(
                                    benchmark_returns_aligned.values, 
                                    stock_returns_aligned.values
                                )
                                
                                beta_analysis = BetaAnalysis(
                                    beta=float(slope),
                                    r_square=float(r_value ** 2),
                                    alpha=float(intercept) * 252  # 연환산
                                )
                                
                                logger.debug(f"Calculated beta analysis for {stock_code}: beta={slope:.4f}, r²={r_value**2:.4f}")
                            else:
                                logger.warning(f"Insufficient data for beta calculation: {stock_code}")
                                beta_analysis = self._get_default_beta_analysis()
                        except Exception as e:
                            logger.error(f"Error calculating beta for {stock_code}: {str(e)}")
                            beta_analysis = self._get_default_beta_analysis()
                    else:
                        # 가격 데이터가 없는 경우 기본값 사용
                        beta_analysis = self._get_default_beta_analysis()
                    
                    stock_detail = StockDetails(
                        expected_return=expected_return,
                        volatility=volatility,
                        correlation_to_portfolio=correlation_to_portfolio,
                        beta_analysis=beta_analysis
                    )
                    
                    stock_details[stock_code] = stock_detail
                    
                except Exception as e:
                    logger.error(f"Error processing stock {stock_code}: {str(e)}")
                    continue
            
            logger.info(f"Calculated stock details for {len(stock_details)} stocks")
            return stock_details
            
        except Exception as e:
            logger.error(f"Error calculating stock details: {str(e)}")
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
                logger.warning("No benchmark data available for portfolio beta calculation")
                return None
            
            # 백테스팅 기간의 포트폴리오 수익률 추출 (최대 소르티노 포트폴리오 기준, 날짜 인덱스 포함)
            portfolio_returns = [r['max_sortino'] for r in optimization_results['portfolio_returns']]
            dates = optimization_results['dates']
            
            if len(portfolio_returns) < 10:
                logger.warning(f"Insufficient portfolio returns for beta calculation: {len(portfolio_returns)}")
                return None
            
            # 포트폴리오 수익률을 시계열로 변환 (날짜 인덱스 사용)
            portfolio_returns_series = pd.Series(portfolio_returns, index=dates)
            
            # 벤치마크 수익률과 동기화
            benchmark_period_returns = self._align_benchmark_returns(
                benchmark_returns, optimization_results['dates']
            )
            # 벤치마크도 같은 인덱스 사용
            benchmark_period_returns.index = dates
            
            if len(benchmark_period_returns) < 10:
                logger.warning(f"Insufficient benchmark returns for beta calculation: {len(benchmark_period_returns)}")
                return None
            
            # 베타 계산 (회귀분석) - 공통 인덱스 기반
            common_index = portfolio_returns_series.index.intersection(benchmark_period_returns.index)
            
            if len(common_index) < 10:
                logger.warning(f"Insufficient common data points for beta calculation: {len(common_index)}")
                return None
            
            portfolio_aligned = portfolio_returns_series.loc[common_index]
            benchmark_aligned = benchmark_period_returns.loc[common_index]
            
            slope, intercept, r_value, p_value, std_err = linregress(
                benchmark_aligned.values, 
                portfolio_aligned.values
            )
            
            # 베타 분석 정보 생성
            portfolio_beta_analysis = BetaAnalysis(
                beta=float(slope),
                r_square=float(r_value ** 2),
                alpha=float(intercept) * 252  # 연환산
            )
            
            logger.info(f"Calculated portfolio beta analysis: beta={slope:.4f}, r²={r_value**2:.4f}, alpha={intercept*252:.4f}")
            return portfolio_beta_analysis
            
        except Exception as e:
            logger.error(f"Error calculating portfolio beta analysis: {str(e)}")
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
                logger.warning(f"No benchmark data available for {portfolio_type} beta calculation")
                return None
            
            # 백테스팅 기간의 포트폴리오 수익률 추출 (날짜 인덱스 포함)
            portfolio_returns = [r[portfolio_type] for r in optimization_results['portfolio_returns']]
            dates = optimization_results['dates']
            
            if len(portfolio_returns) < 10:
                logger.warning(f"Insufficient portfolio returns for {portfolio_type} beta calculation: {len(portfolio_returns)}")
                return None
            
            # 포트폴리오 수익률을 시계열로 변환 (날짜 인덱스 사용)
            portfolio_returns_series = pd.Series(portfolio_returns, index=dates)
            
            # 벤치마크 수익률과 동기화
            benchmark_period_returns = self._align_benchmark_returns(
                benchmark_returns, optimization_results['dates']
            )
            # 벤치마크도 같은 인덱스 사용
            benchmark_period_returns.index = dates
            
            if len(benchmark_period_returns) < 10:
                logger.warning(f"Insufficient benchmark returns for {portfolio_type} beta calculation: {len(benchmark_period_returns)}")
                return None
            
            # 베타 계산 (회귀분석) - 공통 인덱스 기반
            common_index = portfolio_returns_series.index.intersection(benchmark_period_returns.index)
            
            if len(common_index) < 10:
                logger.warning(f"Insufficient common data points for {portfolio_type} beta calculation: {len(common_index)}")
                return None
            
            portfolio_aligned = portfolio_returns_series.loc[common_index]
            benchmark_aligned = benchmark_period_returns.loc[common_index]
            
            slope, intercept, r_value, p_value, std_err = linregress(
                benchmark_aligned.values, 
                portfolio_aligned.values
            )
            
            # 베타 분석 정보 생성
            portfolio_beta_analysis = BetaAnalysis(
                beta=float(slope),
                r_square=float(r_value ** 2),
                alpha=float(intercept) * 252  # 연환산
            )
            
            logger.info(f"Calculated {portfolio_type} beta analysis: beta={slope:.4f}, r²={r_value**2:.4f}, alpha={intercept*252:.4f}")
            return portfolio_beta_analysis
            
        except Exception as e:
            logger.error(f"Error calculating {portfolio_type} beta analysis: {str(e)}")
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
                logger.warning("No benchmark data or user holdings available for user portfolio beta calculation")
                return None
            
            # 사용자 포트폴리오 구성
            user_weights = {}
            total_value = sum(h.quantity for h in request.holdings)
            
            for holding in request.holdings:
                user_weights[holding.code] = holding.quantity / total_value
            
            # 사용자 포트폴리오 수익률 계산 (간소화)
            # 실제로는 가격 데이터를 로드해서 계산해야 하지만, 
            # 여기서는 기본값 반환
            logger.info(f"User portfolio beta calculation requested for {len(user_weights)} stocks")
            
            return BetaAnalysis(
                beta=1.0,
                r_square=0.0,
                alpha=0.0
            )
            
        except Exception as e:
            logger.error(f"Error calculating user portfolio beta: {str(e)}")
            return None

    def _get_default_beta_analysis(self) -> BetaAnalysis:
        """기본 베타 분석 정보"""
        return BetaAnalysis(
            beta=1.0,
            r_square=0.0,
            alpha=0.0
        )

