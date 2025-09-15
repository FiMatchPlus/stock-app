from sqlalchemy import Column, Integer, String, DateTime, Float, Index, Computed
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone, timedelta

# 한국 시간대 설정
KST = timezone(timedelta(hours=9))

def get_kst_now():
    return datetime.now(KST)

Base = declarative_base()

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
    change_rate = Column(Float, Computed('change_amount / (close_price - change_amount) * 100'))  # 자동 계산 컬럼
