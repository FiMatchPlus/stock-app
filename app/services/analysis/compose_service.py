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
        
        logger.info(
            "콜백을 위한 포트폴리오 비중 준비 중",
            min_downside_risk_weights=latest_weights.get('min_downside_risk', {}),
            max_sortino_weights=latest_weights.get('max_sortino', {})
        )
        
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
            logger.warning("사용자 비중 계산을 위한 가격 데이터가 없음, 수량 기반 비중 사용")
            total_qty = sum(h.quantity for h in request.holdings)
            return {h.code: h.quantity / total_qty for h in request.holdings} if total_qty > 0 else {}
        
        latest_prices = prices_df.iloc[-1]
        
        holdings_value = {}
        total_value = 0.0
        
        for h in request.holdings:
            if h.code in latest_prices:
                price = latest_prices[h.code]
                if pd.isna(price):
                    logger.warning(f"종목 {h.code}의 가격이 NaN, 비중 계산에서 제외")
                    continue
                value = h.quantity * price
                holdings_value[h.code] = value
                total_value += value
            else:
                logger.warning(f"종목 {h.code}의 가격 데이터가 없음, 비중 계산에서 제외")
        
        if total_value <= 0 or pd.isna(total_value):
            logger.warning(f"총 포트폴리오 가치가 유효하지 않음: {total_value}")
            return {}
        
        weights = {code: value / total_value for code, value in holdings_value.items()}
        
        logger.info(
            "사용자 포트폴리오 비중 계산 완료",
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
            
            latest_prices = prices_df[available].iloc[-1]
            holdings_value = {}
            total_value = 0.0
            
            for h in request.holdings:
                if h.code in available and h.code in latest_prices:
                    price = latest_prices[h.code]
                    if pd.isna(price):
                        logger.warning(f"메트릭 계산에서 종목 {h.code}의 가격이 NaN, 제외")
                        continue
                    value = h.quantity * price
                    holdings_value[h.code] = value
                    total_value += value
            
            if total_value <= 0 or pd.isna(total_value):
                logger.warning(f"메트릭 계산에서 총 포트폴리오 가치가 유효하지 않음: {total_value}")
                return None
            
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
            logger.error(f"사용자 포트폴리오 메트릭 계산 오류: {str(e)}")
            return None

