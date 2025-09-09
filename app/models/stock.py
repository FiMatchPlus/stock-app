"""주가 관련 SQLAlchemy 모델"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy import (
    BigInteger,
    String,
    DateTime,
    Numeric,
    CheckConstraint,
    Index,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, foreign
from app.models.database import Base


class Stock(Base):
    """종목 정보 테이블"""
    __tablename__ = "stocks"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, comment="단축코드")
    name: Mapped[str] = mapped_column(String(100), nullable=False, comment="한글종목명")
    eng_name: Mapped[Optional[str]] = mapped_column(String(100), comment="영어이름")
    isin: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, comment="표준코드")
    region: Mapped[str] = mapped_column(String(20), nullable=False, default="KR", comment="지역")
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="KRW", comment="통화")
    major_code: Mapped[Optional[str]] = mapped_column(String(100), comment="지수업종대분류코드명")
    medium_code: Mapped[Optional[str]] = mapped_column(String(100), comment="지수업종중분류코드명")
    minor_code: Mapped[Optional[str]] = mapped_column(String(100), comment="지수업종소분류코드명")
    market: Mapped[str] = mapped_column(String(20), nullable=False, comment="시장구분")
    exchange: Mapped[Optional[str]] = mapped_column(String(20), comment="거래소구분")
    is_active: Mapped[str] = mapped_column(
        String(1), 
        nullable=False, 
        default="Y",
        comment="매매가능여부"
    )
    industry_code: Mapped[Optional[int]] = mapped_column(comment="표준산업분류코드")
    industry_name: Mapped[Optional[str]] = mapped_column(String(100), comment="표준산업분류코드명")
    type: Mapped[Optional[str]] = mapped_column(String(50), comment="상품종류")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow,
        comment="생성일시"
    )
    
    # 관계 설정
    prices: Mapped[list["StockPrice"]] = relationship(
        "StockPrice", 
        back_populates="stock",
        cascade="all, delete-orphan",
        primaryjoin="Stock.ticker == foreign(StockPrice.stock_code)"
    )
    
    # 제약 조건
    __table_args__ = (
        CheckConstraint("is_active IN ('Y', 'N')", name="ck_stocks_is_active"),
        Index("idx_stocks_ticker", "ticker"),
        Index("idx_stocks_market", "market"),
        Index("idx_stocks_industry", "industry_code"),
    )


class StockPrice(Base):
    """주가 데이터 테이블"""
    __tablename__ = "stock_prices"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stock_code: Mapped[str] = mapped_column(String(10), nullable=False, comment="종목코드")
    datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="시간")
    interval_unit: Mapped[str] = mapped_column(
        String(5), 
        nullable=False, 
        comment="시간간격"
    )
    
    # 가격 정보
    open_price: Mapped[Decimal] = mapped_column(
        Numeric(15, 2), 
        nullable=False, 
        comment="시가"
    )
    high_price: Mapped[Decimal] = mapped_column(
        Numeric(15, 2), 
        nullable=False, 
        comment="고가"
    )
    low_price: Mapped[Decimal] = mapped_column(
        Numeric(15, 2), 
        nullable=False, 
        comment="저가"
    )
    close_price: Mapped[Decimal] = mapped_column(
        Numeric(15, 2), 
        nullable=False, 
        comment="종가"
    )
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="거래량")
    
    # 전일대비 정보
    change_amount: Mapped[Decimal] = mapped_column(
        Numeric(15, 2), 
        nullable=False, 
        comment="전일대비"
    )
    
    # 전일대비 비율 (계산된 컬럼)
    change_rate: Mapped[Decimal] = mapped_column(
        Numeric(8, 6),
        comment="전일대비비율",
        server_default=text("""
            CASE
                WHEN change_amount IS NOT NULL
                     AND close_price > 0
                     AND (close_price - change_amount) > 0
                THEN ROUND(change_amount / (close_price - change_amount), 6)
                ELSE 0.0
            END
        """)
    )
    
    # 관계 설정 (외래키 제약 없이)
    stock: Mapped["Stock"] = relationship(
        "Stock", 
        back_populates="prices",
        primaryjoin="foreign(StockPrice.stock_code) == Stock.ticker"
    )
    
    # 제약 조건 및 인덱스
    __table_args__ = (
        CheckConstraint("interval_unit IN ('1m', '1d', '1W', '1Y')", name="ck_stock_prices_interval"),
        CheckConstraint("open_price >= 0", name="ck_stock_prices_open_price"),
        CheckConstraint("high_price >= 0", name="ck_stock_prices_high_price"),
        CheckConstraint("low_price >= 0", name="ck_stock_prices_low_price"),
        CheckConstraint("close_price >= 0", name="ck_stock_prices_close_price"),
        CheckConstraint("volume >= 0", name="ck_stock_prices_volume"),
        UniqueConstraint("stock_code", "interval_unit", "datetime", name="uq_stock_prices_unique"),
        Index("idx_stock_prices_stock_datetime", "stock_code", "datetime"),
        Index("idx_stock_prices_interval", "interval_unit"),
        Index("idx_stock_prices_datetime", "datetime"),
    )
