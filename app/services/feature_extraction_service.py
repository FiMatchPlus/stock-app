"""금융 피처 추출 서비스"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.models.schemas import FinancialFeatures, TimeSeriesData
from app.utils.logger import get_logger
from app.repositories.stock_price_repository import StockPriceRepository
from app.repositories.stock_repository import StockRepository

logger = get_logger(__name__)


class FeatureExtractionService:
    """금융 피처 추출 서비스"""
    
    def __init__(self):
        self.stock_price_repo = StockPriceRepository()
        self.stock_repo = StockRepository()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def extract_financial_features(
        self, 
        symbol: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lookback_days: int = 252
    ) -> FinancialFeatures:
        """종목의 금융 피처를 추출합니다."""
        try:
            # 기본 날짜 설정
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=lookback_days)
            
            # 주가 데이터 조회
            stock_prices = await self.stock_price_repo.get_stock_prices(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            
            if not stock_prices or len(stock_prices) < 30:
                raise ValueError(f"Insufficient data for symbol {symbol}")
            
            # DataFrame으로 변환
            df = pd.DataFrame([
                {
                    'date': price.datetime,
                    'close': float(price.close_price),
                    'volume': price.volume,
                    'open': float(price.open_price),
                    'high': float(price.high_price),
                    'low': float(price.low_price)
                }
                for price in stock_prices
            ])
            df = df.sort_values('date').reset_index(drop=True)
            
            # 수익률 계산
            df['returns'] = df['close'].pct_change().fillna(0)
            
            # 변동성 계산 (연환산)
            volatility = df['returns'].std() * np.sqrt(252)
            
            # 베타 계산 (KOSPI 대비)
            beta = await self._calculate_beta(df, symbol, start_date, end_date)
            
            # 섹터 정보 조회
            sector = await self._get_sector_info(symbol)
            
            # 추가 피처들 계산
            pe_ratio, pb_ratio, dividend_yield = await self._calculate_ratios(symbol)
            market_cap = await self._calculate_market_cap(symbol, df['close'].iloc[-1])
            
            return FinancialFeatures(
                symbol=symbol,
                volatility=float(volatility),
                beta=float(beta),
                sector=sector,
                market_cap=market_cap,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                dividend_yield=dividend_yield
            )
            
        except Exception as e:
            logger.error(f"Failed to extract features for {symbol}", error=str(e))
            raise
    
    async def extract_time_series_data(
        self,
        symbol: str,
        sequence_length: int = 60,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> TimeSeriesData:
        """시계열 데이터를 추출합니다."""
        try:
            # 기본 날짜 설정
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=sequence_length + 30)
            
            # 주가 데이터 조회
            stock_prices = await self.stock_price_repo.get_stock_prices(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            
            if not stock_prices or len(stock_prices) < sequence_length:
                raise ValueError(f"Insufficient data for symbol {symbol}")
            
            # 최근 sequence_length개 데이터만 사용
            recent_prices = stock_prices[-sequence_length:]
            
            # 데이터 정렬
            recent_prices.sort(key=lambda x: x.datetime)
            
            # 시계열 데이터 구성
            timestamps = [price.datetime for price in recent_prices]
            prices = [float(price.close_price) for price in recent_prices]
            volumes = [price.volume for price in recent_prices]
            
            # 수익률 계산
            returns = []
            for i in range(1, len(prices)):
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
            returns.insert(0, 0.0)  # 첫 번째 값은 0
            
            return TimeSeriesData(
                symbol=symbol,
                timestamps=timestamps,
                prices=prices,
                volumes=volumes,
                returns=returns
            )
            
        except Exception as e:
            logger.error(f"Failed to extract time series data for {symbol}", error=str(e))
            raise
    
    async def extract_batch_features(
        self, 
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lookback_days: int = 252
    ) -> Dict[str, FinancialFeatures]:
        """여러 종목의 금융 피처를 배치로 추출합니다."""
        try:
            tasks = []
            for symbol in symbols:
                task = self.extract_financial_features(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    lookback_days=lookback_days
                )
                tasks.append((symbol, task))
            
            results = {}
            for symbol, task in tasks:
                try:
                    features = await task
                    results[symbol] = features
                except Exception as e:
                    logger.warning(f"Failed to extract features for {symbol}", error=str(e))
                    continue
            
            return results
            
        except Exception as e:
            logger.error("Failed to extract batch features", error=str(e))
            raise
    
    async def _calculate_beta(
        self, 
        stock_df: pd.DataFrame, 
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> float:
        """베타를 계산합니다 (KOSPI 대비)."""
        try:
            # KOSPI 데이터 조회 (간단한 구현)
            # 실제로는 KOSPI 지수 데이터를 조회해야 함
            kospi_prices = await self.stock_price_repo.get_stock_prices(
                symbol="KOSPI",  # KOSPI 지수 코드
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            
            if not kospi_prices or len(kospi_prices) < 30:
                return 1.0  # 기본값
            
            # KOSPI DataFrame 생성
            kospi_df = pd.DataFrame([
                {'date': price.datetime, 'close': float(price.close_price)}
                for price in kospi_prices
            ])
            kospi_df = kospi_df.sort_values('date').reset_index(drop=True)
            kospi_df['returns'] = kospi_df['close'].pct_change().fillna(0)
            
            # 날짜 기준으로 병합
            merged_df = pd.merge(
                stock_df[['date', 'returns']], 
                kospi_df[['date', 'returns']], 
                on='date', 
                suffixes=('_stock', '_kospi')
            )
            
            if len(merged_df) < 20:
                return 1.0
            
            # 베타 계산
            covariance = np.cov(merged_df['returns_stock'], merged_df['returns_kospi'])[0, 1]
            kospi_variance = np.var(merged_df['returns_kospi'])
            
            if kospi_variance == 0:
                return 1.0
            
            beta = covariance / kospi_variance
            return float(beta)
            
        except Exception as e:
            logger.warning(f"Failed to calculate beta for {symbol}", error=str(e))
            return 1.0
    
    async def _get_sector_info(self, symbol: str) -> str:
        """종목의 섹터 정보를 조회합니다."""
        try:
            stock = await self.stock_repo.get_stock_by_ticker(symbol)
            if stock and stock.industry_name:
                return stock.industry_name
            return "Unknown"
        except Exception as e:
            logger.warning(f"Failed to get sector info for {symbol}", error=str(e))
            return "Unknown"
    
    async def _calculate_ratios(self, symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """PER, PBR, 배당수익률을 계산합니다."""
        try:
            # 실제로는 재무제표 데이터를 조회해야 함
            # 여기서는 기본값 반환
            return None, None, None
        except Exception as e:
            logger.warning(f"Failed to calculate ratios for {symbol}", error=str(e))
            return None, None, None
    
    async def _calculate_market_cap(self, symbol: str, current_price: float) -> Optional[float]:
        """시가총액을 계산합니다."""
        try:
            # 실제로는 발행주식수를 조회해야 함
            # 여기서는 기본값 반환
            return None
        except Exception as e:
            logger.warning(f"Failed to calculate market cap for {symbol}", error=str(e))
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """기술적 지표를 계산합니다."""
        try:
            indicators = {}
            
            # RSI (14일)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = float(100 - (100 / (1 + rs.iloc[-1]))) if not pd.isna(rs.iloc[-1]) else 50.0
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            indicators['macd'] = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0
            indicators['macd_signal'] = float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else 0.0
            
            # 볼린저 밴드
            bb_period = 20
            bb_std = 2
            sma = df['close'].rolling(window=bb_period).mean()
            std = df['close'].rolling(window=bb_period).std()
            indicators['bb_upper'] = float(sma.iloc[-1] + (std.iloc[-1] * bb_std)) if not pd.isna(sma.iloc[-1]) else 0.0
            indicators['bb_lower'] = float(sma.iloc[-1] - (std.iloc[-1] * bb_std)) if not pd.isna(sma.iloc[-1]) else 0.0
            indicators['bb_middle'] = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else 0.0
            
            return indicators
            
        except Exception as e:
            logger.warning("Failed to calculate technical indicators", error=str(e))
            return {}
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
