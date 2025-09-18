from sqlalchemy import Column, Integer, String, DateTime, Float, Index, Computed, BigInteger, Numeric, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone, timedelta
from app.models.database import Base

# 한국 시간대 설정
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
    change_rate = Column(Float, Computed('change_amount / (close_price - change_amount) * 100'))  # 자동 계산 컬럼


class PortfolioSnapshot(Base):
    """포트폴리오 스냅샷 모델"""
    __tablename__ = "portfolio_snapshots"
    
    id = Column(BigInteger, primary_key=True, index=True)
    portfolio_id = Column(BigInteger, nullable=False, index=True)  # 외래키 제약조건 제거
    # portfolio_name = Column(String(100), nullable=True)  # 포트폴리오 이름 - DB에 컬럼이 없음
    base_value = Column(Numeric(12, 2), nullable=False)
    current_value = Column(Numeric(12, 2), nullable=False)
    start_at = Column(DateTime, nullable=False, index=True)  # 백테스트 시작일
    end_at = Column(DateTime, nullable=False, index=True)    # 백테스트 종료일
    created_at = Column(DateTime, nullable=False, default=get_kst_now)
    metric_id = Column(String(24), nullable=True, index=True)  # MongoDB ObjectId
    execution_time = Column(Numeric(10, 3), nullable=True)  # 실행 시간 (초)
    
    # 관계 설정
    holdings = relationship("HoldingSnapshot", back_populates="portfolio_snapshot", cascade="all, delete-orphan")


class HoldingSnapshot(Base):
    """보유 종목 스냅샷 모델"""
    __tablename__ = "holding_snapshots"
    
    id = Column(BigInteger, primary_key=True, index=True)
    stock_id = Column(String(10), nullable=False, index=True)
    portfolio_snapshot_id = Column(BigInteger, ForeignKey("portfolio_snapshots.id"), nullable=False, index=True)
    weight = Column(Numeric(5, 4), nullable=False)
    price = Column(Numeric(12, 2), nullable=False)
    quantity = Column(BigInteger, nullable=False)
    value = Column(Numeric(20, 2), nullable=False)
    recorded_at = Column(DateTime, nullable=False, index=True)
    
    # 관계 설정
    portfolio_snapshot = relationship("PortfolioSnapshot", back_populates="holdings")
