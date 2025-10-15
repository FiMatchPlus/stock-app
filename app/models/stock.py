from sqlalchemy import Column, Integer, String, DateTime, Float, Computed, DECIMAL, BigInteger
from datetime import datetime, timezone, timedelta
from app.models.database import Base

KST = timezone(timedelta(hours=9))

def get_kst_now():
    return datetime.now(KST).replace(tzinfo=None)

class Stock(Base):
    __tablename__ = "stocks"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    market = Column(String(20), nullable=False)
    sector = Column(String(50), nullable=True)
    industry = Column(String(100), nullable=True)
    is_active = Column(String(1), nullable=False, default='Y', index=True)
    created_at = Column(DateTime, default=get_kst_now)

class StockPrice(Base):
    __tablename__ = "stock_prices"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_code = Column(String(10), nullable=False, index=True)
    datetime = Column(DateTime, nullable=False, index=True)
    interval_unit = Column(String(10), nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False, default=0)
    change_amount = Column(Float, nullable=False)
    change_rate = Column(Float, Computed('change_amount / (close_price - change_amount) * 100'))


class BenchmarkPrice(Base):
    """벤치마크 지수 가격 데이터 모델"""
    __tablename__ = "benchmark_prices"
    
    id = Column(BigInteger, primary_key=True, index=True)
    index_code = Column(String(20), nullable=False, index=True)
    datetime = Column(DateTime, nullable=False, index=True)
    open_price = Column(DECIMAL(15, 2), nullable=False)
    high_price = Column(DECIMAL(15, 2), nullable=False)
    low_price = Column(DECIMAL(15, 2), nullable=False)
    close_price = Column(DECIMAL(15, 2), nullable=False)
    change_amount = Column(DECIMAL(15, 2), nullable=False)
    change_rate = Column(DECIMAL(8, 2), nullable=False)
    volume = Column(BigInteger, default=0)
    trading_value = Column(DECIMAL(20, 2), default=0)
    market_cap = Column(DECIMAL(20, 2), nullable=True)
    created_at = Column(DateTime, default=get_kst_now)
    updated_at = Column(DateTime, default=get_kst_now, onupdate=get_kst_now)


class RiskFreeRate(Base):
    """무위험수익률 데이터 모델"""
    __tablename__ = "risk_free_rates"
    
    id = Column(BigInteger, primary_key=True, index=True)
    rate_type = Column(String(20), nullable=False, index=True)
    datetime = Column(DateTime, nullable=False, index=True)
    rate = Column(DECIMAL(12, 8), nullable=False)
    source = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=get_kst_now)
    updated_at = Column(DateTime, default=get_kst_now, onupdate=get_kst_now)

