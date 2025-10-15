from typing import Dict

# Common
from app.models.schema_common import (
    KST,
    get_kst_now,
    ErrorResponse,
    MissingStockData,
    BacktestDataError,
)

# Stock
from app.models.schema_stock import (
    StockCreate,
    StockResponse,
    StockPriceCreate,
    StockPriceResponse,
    StockPriceCollectionRequest,
    StockPriceQueryRequest,
    AggregateResult,
    BenchmarkPriceResponse,
    RiskFreeRateResponse,
)

# Backtest
from app.models.schema_backtest import (
    Holding,
    TradingRule,
    TradingRules,
    BacktestRequest,
    BacktestMetrics,
    BenchmarkMetrics,
    HoldingSnapshotResponse,
    PortfolioSnapshotResponse,
    StockDailyData,
    ResultSummary,
    ExecutionLog,
    BacktestResponse,
    BacktestJobResponse,
    BacktestCallbackResponse,
)

# Analysis
from app.models.schema_analysis import (
    AnalysisHoldingInput,
    AnalysisRequest,
    BetaAnalysis,
    AnalysisMetrics,
    StockDetails,
    BenchmarkComparison,
    PortfolioData,
    AnalysisMetadata,
    BenchmarkInfo,
    PortfolioAnalysisResponse,
    AnalysisJobResponse,
    AnalysisCallbackMetadata,
    AnalysisCallbackResponse,
)

# Backward compatibility alias
PortfolioWeights = Dict[str, float]

__all__ = [
    # Common
    "KST",
    "get_kst_now",
    "ErrorResponse",
    "MissingStockData",
    "BacktestDataError",
    # Stock
    "StockCreate",
    "StockResponse",
    "StockPriceCreate",
    "StockPriceResponse",
    "StockPriceCollectionRequest",
    "StockPriceQueryRequest",
    "AggregateResult",
    "BenchmarkPriceResponse",
    "RiskFreeRateResponse",
    # Backtest
    "Holding",
    "TradingRule",
    "TradingRules",
    "BacktestRequest",
    "BacktestMetrics",
    "BenchmarkMetrics",
    "HoldingSnapshotResponse",
    "PortfolioSnapshotResponse",
    "StockDailyData",
    "ResultSummary",
    "ExecutionLog",
    "BacktestResponse",
    "BacktestJobResponse",
    "BacktestCallbackResponse",
    # Analysis
    "AnalysisHoldingInput",
    "AnalysisRequest",
    "BetaAnalysis",
    "AnalysisMetrics",
    "StockDetails",
    "BenchmarkComparison",
    "PortfolioData",
    "AnalysisMetadata",
    "BenchmarkInfo",
    "PortfolioAnalysisResponse",
    "AnalysisJobResponse",
    "AnalysisCallbackMetadata",
    "AnalysisCallbackResponse",
    # Backward compatibility
    "PortfolioWeights",
]
