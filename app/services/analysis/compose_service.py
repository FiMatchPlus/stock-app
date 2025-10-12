from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd

from app.models.schemas import PortfolioData
from app.utils.logger import get_logger


logger = get_logger(__name__)


class ComposeService:
    async def _build_portfolio_data(
        self,
        request,
        optimization_results: Dict[str, Any],
        benchmark_returns: pd.Series,
        analysis_start,
        analysis_end,
        backtest_metrics,
        prices_df: pd.DataFrame,
        risk_free_rate: float,
        benchmark_code: Optional[str],
    ) -> List[object]:
        """포트폴리오 데이터 구성"""
        portfolios = []
        latest_weights = optimization_results['latest_weights']
        
        # DEBUG: 실제 전송될 weights 로깅
        logger.info(
            "Portfolio weights being prepared for callback",
            min_downside_risk_weights=latest_weights.get('min_downside_risk', {}),
            max_sortino_weights=latest_weights.get('max_sortino', {})
        )
        
        # 1. 사용자 포트폴리오
        user_weights = self._calculate_user_weights(request, prices_df)
        user_beta = await self._calculate_user_portfolio_beta(
            request, benchmark_returns, analysis_start, analysis_end
        )
        user_metrics = await self._calculate_user_portfolio_metrics(
            request,
            benchmark_returns,
            analysis_start,
            analysis_end,
            prices_df,
            risk_free_rate,
        )
        user_benchmark_comparison = await self._calculate_user_benchmark_comparison(
            request,
            benchmark_returns,
            analysis_start,
            analysis_end,
            prices_df,
            benchmark_code,
        )
        
        portfolios.append(PortfolioData(
            type="user",
            weights=user_weights,
            beta_analysis=user_beta,
            metrics=user_metrics,
            benchmark_comparison=user_benchmark_comparison
        ))
        
        # 2. 최소 하방위험 포트폴리오 (Min Downside Risk)
        min_var_beta = await self._calculate_portfolio_beta_for_weights(
            optimization_results, benchmark_returns, analysis_start, analysis_end, "min_downside_risk"
        )
        min_var_benchmark_comparison = await self._calculate_portfolio_benchmark_comparison(
            optimization_results, benchmark_returns, "min_downside_risk", benchmark_code
        )
        
        portfolios.append(PortfolioData(
            type="min_downside_risk",
            weights=latest_weights['min_downside_risk'],
            beta_analysis=min_var_beta,
            metrics=backtest_metrics.get('min_downside_risk'),
            benchmark_comparison=min_var_benchmark_comparison
        ))
        
        # 3. 최대소르티노 포트폴리오
        max_sortino_beta = await self._calculate_portfolio_beta_for_weights(
            optimization_results, benchmark_returns, analysis_start, analysis_end, "max_sortino"
        )
        max_sortino_benchmark_comparison = await self._calculate_portfolio_benchmark_comparison(
            optimization_results, benchmark_returns, "max_sortino", benchmark_code
        )
        
        portfolios.append(PortfolioData(
            type="max_sortino",
            weights=latest_weights['max_sortino'],
            beta_analysis=max_sortino_beta,
            metrics=backtest_metrics.get('max_sortino'),
            benchmark_comparison=max_sortino_benchmark_comparison
        ))
        
        return portfolios

    def _calculate_user_weights(self, request, prices_df: pd.DataFrame) -> Dict[str, float]:
        """사용자 포트폴리오 비중 계산 (평가액 기반)
        
        Args:
            request: 분석 요청 (holdings 포함)
            prices_df: 가격 데이터프레임
            
        Returns:
            종목코드별 비중 딕셔너리 (평가액 기준)
        """
        if prices_df is None or prices_df.empty:
            # 가격 데이터가 없으면 수량 기반으로 폴백
            logger.warning("No price data available for user weights calculation, using quantity-based weights")
            total_qty = sum(h.quantity for h in request.holdings)
            return {h.code: h.quantity / total_qty for h in request.holdings} if total_qty > 0 else {}
        
        # 최신 가격 사용 (마지막 거래일)
        latest_prices = prices_df.iloc[-1]
        
        # 각 종목의 평가액 계산
        holdings_value = {}
        total_value = 0.0
        
        for h in request.holdings:
            if h.code in latest_prices:
                price = latest_prices[h.code]
                value = h.quantity * price
                holdings_value[h.code] = value
                total_value += value
            else:
                logger.warning(f"No price data for stock {h.code}, excluding from weight calculation")
        
        # 비중 계산
        if total_value <= 0:
            logger.warning("Total portfolio value is zero or negative")
            return {}
        
        weights = {code: value / total_value for code, value in holdings_value.items()}
        
        # DEBUG: 사용자 비중 계산 로깅
        logger.info(
            "User portfolio weights calculated",
            holdings_value=holdings_value,
            total_value=total_value,
            weights=weights
        )
        
        return weights

    async def _calculate_user_portfolio_metrics(
        self,
        request,
        benchmark_returns: pd.Series,
        analysis_start: datetime,
        analysis_end: datetime,
        prices_df: pd.DataFrame,
        risk_free_rate: float,
    ) -> Optional[object]:
        """사용자 포트폴리오 성과 지표 계산"""
        try:
            if prices_df is None or prices_df.empty:
                return None
            symbols = [h.code for h in request.holdings]
            available = [s for s in symbols if s in prices_df.columns]
            if not available:
                return None
            
            # 최신 가격으로 평가액 기반 비중 계산
            latest_prices = prices_df[available].iloc[-1]
            holdings_value = {}
            total_value = 0.0
            
            for h in request.holdings:
                if h.code in available and h.code in latest_prices:
                    price = latest_prices[h.code]
                    value = h.quantity * price
                    holdings_value[h.code] = value
                    total_value += value
            
            if total_value <= 0:
                logger.warning("Total portfolio value is zero or negative in metrics calculation")
                return None
            
            # 평가액 기반 비중
            weights = {code: value / total_value for code, value in holdings_value.items()}
            
            returns_df = prices_df[available].pct_change().dropna()
            if returns_df.empty:
                return None
            user_returns = returns_df.dot(pd.Series(weights))
            if not benchmark_returns.empty:
                common_idx = user_returns.index.intersection(benchmark_returns.index)
                bench_aligned = benchmark_returns.loc[common_idx]
                port_aligned = user_returns.loc[common_idx]
            else:
                bench_aligned = pd.Series(dtype=float)
                port_aligned = user_returns
            return self._calculate_portfolio_metrics(
                port_aligned, bench_aligned, risk_free_rate, "User", None
            )
        except Exception as e:
            logger.error(f"Error calculating user portfolio metrics: {str(e)}")
            return None

