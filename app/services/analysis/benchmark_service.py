from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd

from app.models.schemas import BenchmarkComparison
from app.utils.logger import get_logger


logger = get_logger(__name__)


class BenchmarkService:
    def _calculate_return_attribution(
        self,
        optimization_results: Dict[str, Any],
        benchmark_return: float,
        portfolio_type: str = 'max_sortino'
    ) -> tuple:
        """수익률 기여도 분석
        
        각 종목의 수익률 기여도를 계산:
        종목 i의 기여도 = w_i × r_i
        
        Args:
            optimization_results: 최적화 결과
            benchmark_return: 벤치마크 연환산 수익률
            portfolio_type: 'max_sortino' 또는 'min_downside_risk'
            
        Returns:
            (security_selection, timing_effect) 튜플
        """
        try:
            # 최신 윈도우의 비중과 기대수익률
            if not optimization_results.get('expected_returns'):
                logger.warning("No expected returns data for attribution")
                return 0.0, 0.0
                
            latest_expected_returns = optimization_results['expected_returns'][-1]
            latest_weights = optimization_results['latest_weights'][portfolio_type]
            
            # 각 종목의 수익률 기여도 계산
            total_contribution = 0.0
            stock_contributions = {}
            
            for stock_code, weight in latest_weights.items():
                expected_return = latest_expected_returns.get(stock_code, 0.0)
                contribution = weight * expected_return
                stock_contributions[stock_code] = contribution
                total_contribution += contribution
            
            # 종목 선택 효과: 각 종목의 초과수익률 기여도
            # 종목 i의 초과기여도 = w_i × (r_i - r_benchmark)
            security_selection = 0.0
            for stock_code, weight in latest_weights.items():
                expected_return = latest_expected_returns.get(stock_code, 0.0)
                excess_contribution = weight * (expected_return - benchmark_return)
                security_selection += excess_contribution
            
            # 타이밍 효과: 나머지 초과수익률
            # (실제로는 시간에 따른 비중 변화 효과지만, 현재는 단순화)
            portfolio_return = total_contribution
            excess_return = portfolio_return - benchmark_return
            timing_effect = excess_return - security_selection
            
            logger.debug(
                f"Return attribution for {portfolio_type}: "
                f"security_selection={security_selection:.4f}, timing_effect={timing_effect:.4f}, "
                f"total_excess={excess_return:.4f}"
            )
            
            return float(security_selection), float(timing_effect)
            
        except Exception as e:
            logger.error(f"Error calculating return attribution: {str(e)}")
            # Fallback to simple split
            return 0.0, 0.0
    
    async def _determine_benchmark(self, request, benchmark_repo) -> str:
        """적절한 벤치마크 결정"""
        if request.benchmark:
            # 사용자가 지정한 벤치마크가 있는지 확인
            available_benchmarks = await benchmark_repo.get_available_benchmarks()
            if request.benchmark in available_benchmarks:
                return request.benchmark
            else:
                logger.warning(f"Requested benchmark {request.benchmark} not available, falling back to KOSPI")
        
        # 기본적으로 KOSPI 사용
        return "KOSPI"

    async def _calculate_benchmark_comparison(
        self,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        benchmark_code: str,
    ) -> Optional[object]:
        """벤치마크 비교 분석"""
        try:
            if benchmark_returns.empty:
                return None
            
            # 백테스팅 기간의 포트폴리오 수익률
            mv_returns = [r['min_downside_risk'] for r in optimization_results['portfolio_returns']]
            ms_returns = [r['max_sortino'] for r in optimization_results['portfolio_returns']]
            
            # 최대 소르티노 포트폴리오를 기준으로 비교 분석
            portfolio_returns = pd.Series(ms_returns)
            
            # 벤치마크 수익률 조정
            benchmark_period_returns = self._align_benchmark_returns(benchmark_returns, optimization_results['dates'])
            
            # 벤치마크 통계
            benchmark_annual_return = benchmark_period_returns.mean() * 252.0
            benchmark_volatility = benchmark_period_returns.std() * np.sqrt(252)
            
            # 포트폴리오 통계
            portfolio_annual_return = portfolio_returns.mean() * 252.0
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            
            # 초과 성과
            excess_return = portfolio_annual_return - benchmark_annual_return
            relative_volatility = portfolio_volatility / benchmark_volatility if benchmark_volatility > 0 else 1.0
            
            # 수익률 기여도 분석 (최신 윈도우 기준)
            security_selection, timing_effect = self._calculate_return_attribution(
                optimization_results, 
                benchmark_annual_return,
                portfolio_type='max_sortino'
            )
            
            return BenchmarkComparison(
                benchmark_code=benchmark_code,
                benchmark_return=float(benchmark_annual_return),
                benchmark_volatility=float(benchmark_volatility),
                excess_return=float(excess_return),
                relative_volatility=float(relative_volatility),
                security_selection=float(security_selection),
                timing_effect=float(timing_effect)
            )
            
        except Exception as e:
            logger.error(f"Error calculating benchmark comparison: {str(e)}")
            return None

    async def _calculate_user_benchmark_comparison(
        self,
        request,
        benchmark_returns: pd.Series,
        analysis_start: datetime,
        analysis_end: datetime,
        prices_df: pd.DataFrame,
        benchmark_code: Optional[str],
    ) -> Optional[object]:
        """사용자 포트폴리오 벤치마크 비교"""
        try:
            if prices_df is None or prices_df.empty or benchmark_returns.empty:
                return None
            symbols = [h.code for h in request.holdings]
            available = [s for s in symbols if s in prices_df.columns]
            if not available:
                return None
            total_qty = sum(h.quantity for h in request.holdings if h.code in available)
            if total_qty <= 0:
                return None
            weights = {h.code: h.quantity / total_qty for h in request.holdings if h.code in available}
            returns_df = prices_df[available].pct_change().dropna()
            if returns_df.empty:
                return None
            user_returns = returns_df.dot(pd.Series(weights))
            common_idx = user_returns.index.intersection(benchmark_returns.index)
            if len(common_idx) == 0:
                return None
            port_aligned = user_returns.loc[common_idx]
            bench_aligned = benchmark_returns.loc[common_idx]
            benchmark_annual_return = bench_aligned.mean() * 252.0
            benchmark_volatility = bench_aligned.std() * np.sqrt(252)
            portfolio_annual_return = port_aligned.mean() * 252.0
            portfolio_volatility = port_aligned.std() * np.sqrt(252)
            excess_return = portfolio_annual_return - benchmark_annual_return
            relative_volatility = (
                portfolio_volatility / benchmark_volatility if benchmark_volatility > 0 else 1.0
            )
            
            # 사용자 포트폴리오의 수익률 기여도 계산
            # 각 종목의 기여도 = w_i × r_i
            security_selection = 0.0
            for h in request.holdings:
                if h.code in available:
                    stock_weight = weights[h.code]
                    stock_returns = returns_df[h.code]
                    stock_aligned = stock_returns.loc[common_idx]
                    stock_annual_return = stock_aligned.mean() * 252.0
                    # 종목의 초과기여도 = w_i × (r_i - r_benchmark)
                    excess_contribution = stock_weight * (stock_annual_return - benchmark_annual_return)
                    security_selection += excess_contribution
            
            # 타이밍 효과는 나머지
            timing_effect = excess_return - security_selection
            
            return BenchmarkComparison(
                benchmark_code=benchmark_code or "KOSPI",
                benchmark_return=float(benchmark_annual_return),
                benchmark_volatility=float(benchmark_volatility),
                excess_return=float(excess_return),
                relative_volatility=float(relative_volatility),
                security_selection=float(security_selection),
                timing_effect=float(timing_effect),
            )
        except Exception as e:
            logger.error(f"Error calculating user benchmark comparison: {str(e)}")
            return None

    async def _calculate_portfolio_benchmark_comparison(
        self,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        portfolio_type: str,
        benchmark_code: Optional[str] = None
    ) -> Optional[object]:
        """특정 포트폴리오의 벤치마크 비교"""
        try:
            if benchmark_returns.empty:
                return None
            
            # 백테스팅 기간의 포트폴리오 수익률 추출
            portfolio_returns = [r[portfolio_type] for r in optimization_results['portfolio_returns']]
            portfolio_returns_series = pd.Series(portfolio_returns)
            
            # 벤치마크 수익률과 동기화
            benchmark_period_returns = self._align_benchmark_returns(
                benchmark_returns, optimization_results['dates']
            )
            
            # 벤치마크 통계
            benchmark_annual_return = benchmark_period_returns.mean() * 252.0
            benchmark_volatility = benchmark_period_returns.std() * np.sqrt(252)
            
            # 포트폴리오 통계
            portfolio_annual_return = portfolio_returns_series.mean() * 252.0
            portfolio_volatility = portfolio_returns_series.std() * np.sqrt(252)
            
            # 초과 성과
            excess_return = portfolio_annual_return - benchmark_annual_return
            relative_volatility = portfolio_volatility / benchmark_volatility if benchmark_volatility > 0 else 1.0
            
            # 수익률 기여도 분석
            security_selection, timing_effect = self._calculate_return_attribution(
                optimization_results,
                benchmark_annual_return,
                portfolio_type=portfolio_type
            )
            
            return BenchmarkComparison(
                benchmark_code=benchmark_code or "KOSPI",
                benchmark_return=float(benchmark_annual_return),
                benchmark_volatility=float(benchmark_volatility),
                excess_return=float(excess_return),
                relative_volatility=float(relative_volatility),
                security_selection=float(security_selection),
                timing_effect=float(timing_effect)
            )
            
        except Exception as e:
            logger.error(f"Error calculating {portfolio_type} benchmark comparison: {str(e)}")
            return None

