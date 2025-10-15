from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app.utils.logger import get_logger


logger = get_logger(__name__)


class OptimizationService:
    async def _perform_rolling_optimization(
        self,
        prices_df: pd.DataFrame,
        benchmark_returns: pd.Series,
        risk_free_rate: float,
    ) -> Dict[str, Any]:
        """Moving Window 기반 포트폴리오 최적화 수행"""
        
        logger.info("Rolling Window 최적화 시작")
        
        # 윈도우 크기 (1년 = 252 거래일)
        window_days = int(252 * self.window_years)
        
        # 이동 간격 (2주 = 약 10.5 거래일)
        step_days = int(21 * self.step_months)
        
        # 백테스팅 기간 (3개월 = 63 거래일) - VaR/CVaR 계산용
        backtest_days = int(21 * self.backtest_months)
        
        # 전체 데이터 길이 확인
        total_days = len(prices_df)
        min_required_days = window_days + backtest_days
        if total_days < min_required_days:
            raise ValueError(f"Insufficient data: need at least {min_required_days} days, got {total_days}")
        
        # 최적화 결과 저장
        optimization_results = {
            'dates': [],
            'min_downside_risk_weights': [],
            'max_sortino_weights': [],
            'portfolio_returns': [],
            'window_var_cvar': [],  # 윈도우별 VaR/CVaR 저장
            'expected_returns': [],
            'covariances': []
        }
        
        for start_idx in range(0, total_days - window_days, step_days):
            end_idx = start_idx + window_days
            
            # 윈도우 데이터 추출
            window_prices = prices_df.iloc[start_idx:end_idx]
            window_date = window_prices.index[-1]  # 윈도우의 마지막 날짜
            
            logger.debug(f"윈도우 최적화 중: {window_prices.index[0]} ~ {window_prices.index[-1]}")
            
            # 수익률 계산
            window_returns = window_prices.pct_change().fillna(0.0)
            
            # 통계 계산
            expected_returns = window_returns.mean() * 252.0
            
            # Semi-covariance matrix 계산 (하방위험 전용 - 양쪽 전략 모두 사용)
            semi_cov_matrix = self._calculate_semi_covariance(window_returns, target_return=0.0)
            
            # 포트폴리오 최적화 (둘 다 하방위험 기준)
            min_var_weights = self._optimize_min_variance(semi_cov_matrix, expected_returns, risk_free_rate)
            max_sortino_weights = self._optimize_max_sortino(expected_returns, semi_cov_matrix, risk_free_rate)
            
            # 다음 기간의 실제 수익률 계산 (백테스팅용)
            # 백테스팅 기간(3개월)의 데이터로 성과 평가 및 VaR/CVaR 계산
            if end_idx < total_days - backtest_days:
                next_period_prices = prices_df.iloc[end_idx:end_idx + backtest_days]
                next_period_returns = next_period_prices.pct_change().dropna()  # NaN 제거
                
                # 최적화된 비중으로 포트폴리오 수익률 계산 (일별)
                mv_portfolio_returns = next_period_returns.dot(min_var_weights)
                ms_portfolio_returns = next_period_returns.dot(max_sortino_weights)
                
                # 윈도우별 VaR/CVaR 계산 (백테스팅 기간 전체 사용)
                mv_var, mv_cvar = self._calculate_var_cvar(mv_portfolio_returns)
                ms_var, ms_cvar = self._calculate_var_cvar(ms_portfolio_returns)
                
                # 다음 기간의 평균 수익률 (성과 평가용)
                mv_portfolio_return = float(mv_portfolio_returns.mean())
                ms_portfolio_return = float(ms_portfolio_returns.mean())
            else:
                mv_portfolio_return = 0.0
                ms_portfolio_return = 0.0
                mv_var = mv_cvar = ms_var = ms_cvar = 0.0
            
            # 결과 저장
            optimization_results['dates'].append(window_date)
            optimization_results['min_downside_risk_weights'].append(min_var_weights.to_dict())
            optimization_results['max_sortino_weights'].append(max_sortino_weights.to_dict())
            optimization_results['portfolio_returns'].append({
                'min_downside_risk': mv_portfolio_return,
                'max_sortino': ms_portfolio_return
            })
            optimization_results['window_var_cvar'].append({
                'Min Downside Risk': {'var': mv_var, 'cvar': mv_cvar},
                'Max Sortino': {'var': ms_var, 'cvar': ms_cvar}
            })
            optimization_results['expected_returns'].append(expected_returns.to_dict())
            optimization_results['covariances'].append(semi_cov_matrix.to_dict())
        
        # 최신 비중 저장 (마지막 윈도우의 비중)
        optimization_results['latest_weights'] = {
            'min_downside_risk': optimization_results['min_downside_risk_weights'][-1],
            'max_sortino': optimization_results['max_sortino_weights'][-1]
        }
        
        # 최신 비중 로깅
        logger.info(
            f"Rolling 최적화 완료: {len(optimization_results['dates'])}개 윈도우 처리됨",
            latest_min_downside_risk_weights=optimization_results['latest_weights']['min_downside_risk'],
            latest_max_sortino_weights=optimization_results['latest_weights']['max_sortino']
        )
        return optimization_results

    def _calculate_semi_covariance(self, returns: pd.DataFrame, target_return: float = 0.0) -> pd.DataFrame:
        """Semi-covariance matrix 계산 (하방위험)
        
        Args:
            returns: 수익률 데이터프레임 (일별 수익률)
            target_return: 목표 수익률 (일별 기준, 기본값 0)
        
        Returns:
            Semi-covariance matrix (연환산)
        """
        # 목표 수익률 이하의 수익률만 필터링
        downside_returns = returns.copy()
        
        # 각 컬럼(종목)별로 목표 수익률 이상인 값은 0으로 설정
        for col in downside_returns.columns:
            downside_returns.loc[downside_returns[col] >= target_return, col] = 0.0
        
        # Semi-covariance 계산
        n = len(downside_returns)
        if n <= 1:
            # 데이터가 부족하면 일반 공분산 반환
            return returns.cov() * 252.0
        
        # 중심화
        centered = downside_returns - target_return
        
        # Semi-covariance matrix 계산
        semi_cov = centered.T.dot(centered) / n
        
        # 연환산
        semi_cov_annual = semi_cov * 252.0
        
        # 대각 성분이 0에 가까운 경우 처리 (최소값 설정)
        min_variance = 1e-8
        for col in semi_cov_annual.columns:
            if semi_cov_annual.loc[col, col] < min_variance:
                semi_cov_annual.loc[col, col] = min_variance
        
        logger.debug(f"Semi-covariance 행렬 계산 완료 (target_return={target_return})")
        
        return semi_cov_annual

    def _optimize_min_variance(self, cov_matrix: pd.DataFrame, expected_returns: pd.Series, risk_free_rate: float) -> pd.Series:
        """최소 하방위험 포트폴리오 최적화 (Semi-covariance 기반, 최소 수익률 제약)
        
        하방표준편차(Downside Deviation)를 최소화하는 포트폴리오 비중 계산
        목표 수익률(0) 이하의 수익률만 고려하여 위험 측정
        
        제약조건:
        - 각 종목 최소 비중: 5% (0.05)
        - 각 종목 최대 비중: 100% (1.0)
        - 비중 합계: 100% (1.0)
        - 기대수익률 >= 무위험수익률 + 0.5%p (소르티노 비율 양수 보장)
        
        Args:
            cov_matrix: Semi-covariance 행렬 (하방위험)
            expected_returns: 기대수익률 벡터
            risk_free_rate: 무위험수익률
        """
        n_assets = len(cov_matrix)
        min_weight = 0.05  # 최소 비중 5%
        min_return_premium = 0.005  # 무위험수익률 대비 최소 프리미엄 0.5%p
        
        # 목적 함수: 포트폴리오 하방위험(semi-variance) 최소화
        # w^T * Σ_semi * w (Semi-covariance matrix 사용)
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix.values, weights))
        
        # 제약 조건: 최소 수익률 제약
        def return_constraint(weights):
            portfolio_return = np.dot(weights, expected_returns.values)
            min_required_return = risk_free_rate + min_return_premium
            return portfolio_return - min_required_return  # >= 0 이어야 함
        
        # 제약 조건: 비중의 합 = 1, 비중 >= 5%, 최소 수익률 만족
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': return_constraint}  # 부등식 제약
        ]
        bounds = tuple((min_weight, 1) for _ in range(n_assets))
        
        # 초기값 (균등 비중)
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                weights = pd.Series(result.x, index=cov_matrix.columns)
                weights = weights / weights.sum()  # 정규화
                
                # 제약 조건 만족 여부 확인
                portfolio_return = np.dot(weights.values, expected_returns.values)
                min_required = risk_free_rate + min_return_premium
                
                if portfolio_return >= min_required:
                    logger.info(f"최소 하방위험 최적화 성공 (수익률 제약 포함): "
                              f"expected_return={portfolio_return:.4f} >= {min_required:.4f}")
                    return weights
                else:
                    logger.warning(f"수익률 제약 미충족: {portfolio_return:.4f} < {min_required:.4f}")
                    # 제약 없이 재시도
                    return self._optimize_min_variance_without_return_constraint(cov_matrix)
            else:
                logger.warning("수익률 제약 포함 최소 하방위험 최적화 실패, 제약 없이 재시도")
                return self._optimize_min_variance_without_return_constraint(cov_matrix)
        except Exception as e:
            logger.error(f"최소 하방위험 최적화 오류: {e}, 제약 없는 버전으로 폴백")
            return self._optimize_min_variance_without_return_constraint(cov_matrix)
    
    def _optimize_min_variance_without_return_constraint(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """최소 하방위험 포트폴리오 최적화 (수익률 제약 없음 - fallback용)"""
        n_assets = len(cov_matrix)
        min_weight = 0.05
        
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix.values, weights))
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((min_weight, 1) for _ in range(n_assets))
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                weights = pd.Series(result.x, index=cov_matrix.columns)
                return weights / weights.sum()
            else:
                logger.warning("최소 하방위험 최적화 실패, inverse semi-variance 가중치 사용")
                return self._fallback_min_variance_weights(cov_matrix)
        except Exception as e:
            logger.error(f"최소 하방위험 최적화 오류: {e}")
            return self._fallback_min_variance_weights(cov_matrix)

    def _optimize_max_sortino(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float) -> pd.Series:
        """최대 소르티노 비율 포트폴리오 최적화
        
        하방표준편차(Downside Deviation)를 위험 척도로 사용하여 소르티노 비율 최대화
        Semi-covariance matrix를 사용하여 손실 상황의 위험만 고려
        
        Sortino Ratio = (R_p - R_f) / σ_downside
        
        제약조건:
        - 각 종목 최소 비중: 5% (0.05)
        - 각 종목 최대 비중: 100% (1.0)
        - 비중 합계: 100% (1.0)
        
        Args:
            expected_returns: 기대수익률 벡터
            cov_matrix: Semi-covariance 행렬 (하방위험)
            risk_free_rate: 무위험수익률
        """
        n_assets = len(expected_returns)
        min_weight = 0.05  # 최소 비중 5%
        
        # 목적 함수: 소르티노 비율 최대화 (하방위험 기준)
        # Sortino = (R_p - R_f) / σ_downside
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns.values)
            portfolio_downside_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
            portfolio_downside_std = np.sqrt(max(portfolio_downside_variance, 1e-8))
            sortino_ratio = (portfolio_return - risk_free_rate) / portfolio_downside_std
            return -sortino_ratio  # 음수를 취해서 최소화
        
        # 제약 조건: 비중의 합 = 1, 비중 >= 5%
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((min_weight, 1) for _ in range(n_assets))
        
        # 초기값 (균등 비중)
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                weights = pd.Series(result.x, index=expected_returns.index)
                return weights / weights.sum()  # 정규화
            else:
                logger.warning("최대 소르티노 최적화 실패, 대체 방법 사용")
                return self._fallback_max_sortino_weights(expected_returns, cov_matrix, risk_free_rate)
        except Exception as e:
            logger.error(f"최대 소르티노 최적화 오류: {e}")
            return self._fallback_max_sortino_weights(expected_returns, cov_matrix, risk_free_rate)

    def _fallback_min_variance_weights(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """최소 하방위험 최적화 실패 시 대체 방법 (inverse semi-variance)"""
        diag_var = np.diag(cov_matrix.values)
        inv_var = np.where(diag_var > 0, 1.0 / diag_var, 0.0)
        w = inv_var / inv_var.sum() if inv_var.sum() > 0 else np.ones_like(inv_var) / len(inv_var)
        return pd.Series(w, index=cov_matrix.columns)

    def _fallback_max_sortino_weights(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float) -> pd.Series:
        """최대 소르티노 최적화 실패 시 대체 방법 (하방위험 기준)"""
        diag_downside_var = np.diag(cov_matrix.values)  # Semi-covariance 대각 성분
        excess = (expected_returns.values - risk_free_rate)
        score = np.where(diag_downside_var > 0, excess / diag_downside_var, 0.0)
        score = np.maximum(score, 0.0)
        if score.sum() == 0:
            score = np.ones_like(score)
        w = score / score.sum()
        return pd.Series(w, index=expected_returns.index)

