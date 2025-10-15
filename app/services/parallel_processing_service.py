"""병렬처리 서비스"""

import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from decimal import Decimal
import numpy as np
import pandas as pd
from functools import partial

from app.config import settings
from app.models.schemas import AggregateResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ParallelProcessingService:
    """병렬처리 서비스"""
    
    def __init__(self):
        self.max_workers = settings.max_workers
        self.chunk_size = settings.chunk_size
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers * 2)
    
    async def calculate_stock_aggregates(
        self, 
        stock_data: List[Dict[str, Any]]
    ) -> List[AggregateResult]:
        """주가 집계 연산 (병렬처리)"""
        try:
            if not stock_data:
                return []
            
            chunks = self._split_data_into_chunks(stock_data, self.chunk_size)
            
            logger.info(
                "Starting parallel aggregation",
                total_records=len(stock_data),
                chunks=len(chunks),
                workers=self.max_workers
            )
            
            loop = asyncio.get_event_loop()
            tasks = []
            
            for i, chunk in enumerate(chunks):
                task = loop.run_in_executor(
                    self.process_pool,
                    self._calculate_chunk_aggregates,
                    chunk,
                    i
                )
                tasks.append(task)
            
            chunk_results = await asyncio.gather(*tasks)
            
            final_results = self._merge_aggregate_results(chunk_results)
            
            logger.info(
                "Parallel aggregation completed",
                total_results=len(final_results)
            )
            
            return final_results
        
        except Exception as e:
            logger.error("Failed to calculate stock aggregates", error=str(e))
            raise
    
    def _split_data_into_chunks(
        self, 
        data: List[Dict[str, Any]], 
        chunk_size: int
    ) -> List[List[Dict[str, Any]]]:
        """데이터를 청크로 분할"""
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunks.append(chunk)
        return chunks
    
    @staticmethod
    def _calculate_chunk_aggregates(
        chunk: List[Dict[str, Any]], 
        chunk_id: int
    ) -> List[Dict[str, Any]]:
        """청크 단위 집계 연산 (프로세스에서 실행)"""
        try:
            if not chunk:
                return []
            
            df = pd.DataFrame(chunk)
            
            results = []
            
            for symbol, group in df.groupby('stock_code'):
                prices = group['close_price'].astype(float)
                volumes = group['volume'].astype(int)
                
                result = {
                    'symbol': symbol,
                    'interval': group['interval_unit'].iloc[0],
                    'count': len(group),
                    'avg_price': float(prices.mean()),
                    'min_price': float(prices.min()),
                    'max_price': float(prices.max()),
                    'total_volume': int(volumes.sum()),
                    'volatility': float(prices.std()),
                    'chunk_id': chunk_id
                }
                
                if len(group) > 1:
                    result['price_correlation'] = float(prices.corr(prices.shift(1)))
                
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Chunk {chunk_id} processing failed", error=str(e))
            return []
    
    def _merge_aggregate_results(
        self, 
        chunk_results: List[List[Dict[str, Any]]]
    ) -> List[AggregateResult]:
        """집계 결과 병합"""
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)
        
        if not all_results:
            return []
        
        symbol_groups = {}
        for result in all_results:
            symbol = result['symbol']
            if symbol not in symbol_groups:
                symbol_groups[symbol] = []
            symbol_groups[symbol].append(result)
        
        final_results = []
        for symbol, results in symbol_groups.items():
            if not results:
                continue
            
            total_count = sum(r['count'] for r in results)
            if total_count == 0:
                continue
            
            weighted_avg_price = sum(
                r['avg_price'] * r['count'] for r in results
            ) / total_count
            
            min_price = min(r['min_price'] for r in results)
            max_price = max(r['max_price'] for r in results)
            
            total_volume = sum(r['total_volume'] for r in results)
            
            avg_volatility = np.mean([r['volatility'] for r in results if not np.isnan(r['volatility'])])
            
            correlations = [r.get('price_correlation', 0) for r in results if 'price_correlation' in r]
            avg_correlation = np.mean(correlations) if correlations else 0.0
            
            final_result = AggregateResult(
                symbol=symbol,
                interval=results[0]['interval'],
                count=total_count,
                avg_price=Decimal(str(round(weighted_avg_price, 2))),
                min_price=Decimal(str(round(min_price, 2))),
                max_price=Decimal(str(round(max_price, 2))),
                total_volume=total_volume,
                volatility=Decimal(str(round(avg_volatility, 6))),
                correlation={"price_autocorr": Decimal(str(round(avg_correlation, 6)))} if not np.isnan(avg_correlation) else None,
                calculated_at=datetime.utcnow()
            )
            
            final_results.append(final_result)
        
        return final_results
    
    async def process_large_dataset(
        self, 
        data: List[Dict[str, Any]], 
        processing_func: callable
    ) -> List[Any]:
        """대용량 데이터셋 병렬 처리"""
        try:
            chunks = self._split_data_into_chunks(data, self.chunk_size)
            
            logger.info(
                "Starting large dataset processing",
                total_records=len(data),
                chunks=len(chunks)
            )
            
            loop = asyncio.get_event_loop()
            tasks = []
            
            for i, chunk in enumerate(chunks):
                task = loop.run_in_executor(
                    self.thread_pool,
                    processing_func,
                    chunk,
                    i
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            final_results = []
            for result in results:
                if isinstance(result, list):
                    final_results.extend(result)
                else:
                    final_results.append(result)
            
            logger.info(
                "Large dataset processing completed",
                total_results=len(final_results)
            )
            
            return final_results
        
        except Exception as e:
            logger.error("Failed to process large dataset", error=str(e))
            raise
    
    async def calculate_correlation_matrix(
        self, 
        stock_data: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """종목 간 상관관계 행렬 계산"""
        try:
            if not stock_data:
                return {}
            
            df = pd.DataFrame(stock_data)
            
            pivot_df = df.pivot_table(
                index='timestamp',
                columns='stock_code',
                values='close_price',
                aggfunc='mean'
            )
            
            correlation_matrix = pivot_df.corr()
            
            correlation_matrix = correlation_matrix.fillna(0)
            
            result = {}
            for symbol1 in correlation_matrix.columns:
                result[symbol1] = {}
                for symbol2 in correlation_matrix.columns:
                    result[symbol1][symbol2] = float(correlation_matrix.loc[symbol1, symbol2])
            
            logger.info(
                "Correlation matrix calculated",
                symbols=len(correlation_matrix.columns)
            )
            
            return result
        
        except Exception as e:
            logger.error("Failed to calculate correlation matrix", error=str(e))
            raise
    
    async def calculate_technical_indicators(
        self, 
        stock_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """기술적 지표 계산 (병렬처리)"""
        try:
            if not stock_data:
                return []
            
            df = pd.DataFrame(stock_data)
            results = []
            
            for symbol, group in df.groupby('stock_code'):
                group = group.sort_values('timestamp')
                
                prices = group['close_price'].astype(float)
                volumes = group['volume'].astype(int)
                
                indicators = self._calculate_technical_indicators_for_symbol(
                    prices, volumes, symbol
                )
                
                results.extend(indicators)
            
            logger.info(
                "Technical indicators calculated",
                total_indicators=len(results)
            )
            
            return results
        
        except Exception as e:
            logger.error("Failed to calculate technical indicators", error=str(e))
            raise
    
    @staticmethod
    def _calculate_technical_indicators_for_symbol(
        prices: pd.Series, 
        volumes: pd.Series, 
        symbol: str
    ) -> List[Dict[str, Any]]:
        """개별 종목의 기술적 지표 계산"""
        indicators = []
        
        try:
            ma_5 = prices.rolling(window=5).mean()
            ma_20 = prices.rolling(window=20).mean()
            ma_50 = prices.rolling(window=50).mean()
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            bb_middle = prices.rolling(window=20).mean()
            bb_std = prices.rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            macd_histogram = macd_line - signal_line
            
            volume_ma = volumes.rolling(window=20).mean()
            
            for i in range(len(prices)):
                if i < 50:
                    continue
                
                indicator = {
                    'symbol': symbol,
                    'timestamp': prices.index[i],
                    'price': float(prices.iloc[i]),
                    'ma_5': float(ma_5.iloc[i]) if not pd.isna(ma_5.iloc[i]) else None,
                    'ma_20': float(ma_20.iloc[i]) if not pd.isna(ma_20.iloc[i]) else None,
                    'ma_50': float(ma_50.iloc[i]) if not pd.isna(ma_50.iloc[i]) else None,
                    'rsi': float(rsi.iloc[i]) if not pd.isna(rsi.iloc[i]) else None,
                    'bb_upper': float(bb_upper.iloc[i]) if not pd.isna(bb_upper.iloc[i]) else None,
                    'bb_middle': float(bb_middle.iloc[i]) if not pd.isna(bb_middle.iloc[i]) else None,
                    'bb_lower': float(bb_lower.iloc[i]) if not pd.isna(bb_lower.iloc[i]) else None,
                    'macd_line': float(macd_line.iloc[i]) if not pd.isna(macd_line.iloc[i]) else None,
                    'macd_signal': float(signal_line.iloc[i]) if not pd.isna(signal_line.iloc[i]) else None,
                    'macd_histogram': float(macd_histogram.iloc[i]) if not pd.isna(macd_histogram.iloc[i]) else None,
                    'volume_ma': float(volume_ma.iloc[i]) if not pd.isna(volume_ma.iloc[i]) else None,
                }
                
                indicators.append(indicator)
        
        except Exception as e:
            logger.error(f"Failed to calculate indicators for {symbol}", error=str(e))
        
        return indicators
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            self.process_pool.shutdown(wait=True)
            self.thread_pool.shutdown(wait=True)
            logger.info("Parallel processing service cleaned up")
        except Exception as e:
            logger.error("Failed to cleanup parallel processing service", error=str(e))

