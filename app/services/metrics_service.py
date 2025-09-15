"""MongoDB Metrics 서비스"""

from typing import Optional, Dict, Any
from bson import ObjectId
from datetime import datetime
from decimal import Decimal

from app.models.schemas import BacktestMetrics
from app.utils.mongodb_client import MongoDBClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsService:
    """MongoDB Metrics 관리 서비스"""
    
    def __init__(self, mongodb_client: MongoDBClient):
        self.mongodb_client = mongodb_client
        self.collection_name = "metrics"
    
    async def save_metrics(self, metrics: BacktestMetrics, portfolio_snapshot_id: int) -> str:
        """백테스트 지표를 MongoDB에 저장"""
        try:
            collection = self.mongodb_client.get_collection(self.collection_name)
            
            # Metrics 데이터를 MongoDB 문서 형태로 변환
            metrics_doc = {
                "portfolio_snapshot_id": portfolio_snapshot_id,
                "total_return": float(metrics.total_return),
                "annualized_return": float(metrics.annualized_return),
                "volatility": float(metrics.volatility),
                "sharpe_ratio": float(metrics.sharpe_ratio),
                "max_drawdown": float(metrics.max_drawdown),
                "var_95": float(metrics.var_95),
                "var_99": float(metrics.var_99),
                "cvar_95": float(metrics.cvar_95),
                "cvar_99": float(metrics.cvar_99),
                "win_rate": float(metrics.win_rate),
                "profit_loss_ratio": float(metrics.profit_loss_ratio),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # MongoDB에 문서 삽입
            result = await collection.insert_one(metrics_doc)
            metric_id = str(result.inserted_id)
            
            logger.info(
                f"Metrics saved to MongoDB",
                metric_id=metric_id,
                portfolio_snapshot_id=portfolio_snapshot_id
            )
            
            return metric_id
            
        except Exception as e:
            logger.error(f"Failed to save metrics to MongoDB: {str(e)}")
            raise
    
    async def get_metrics(self, metric_id: str) -> Optional[Dict[str, Any]]:
        """MongoDB에서 지표 조회"""
        try:
            collection = self.mongodb_client.get_collection(self.collection_name)
            
            # ObjectId로 검색
            if not ObjectId.is_valid(metric_id):
                logger.warning(f"Invalid ObjectId format: {metric_id}")
                return None
            
            doc = await collection.find_one({"_id": ObjectId(metric_id)})
            
            if doc:
                # ObjectId를 문자열로 변환
                doc["_id"] = str(doc["_id"])
                logger.info(f"Metrics retrieved from MongoDB: {metric_id}")
                return doc
            else:
                logger.warning(f"Metrics not found: {metric_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get metrics from MongoDB: {str(e)}")
            raise
    
    async def get_metrics_by_portfolio_snapshot(self, portfolio_snapshot_id: int) -> Optional[Dict[str, Any]]:
        """포트폴리오 스냅샷 ID로 지표 조회"""
        try:
            collection = self.mongodb_client.get_collection(self.collection_name)
            
            doc = await collection.find_one({"portfolio_snapshot_id": portfolio_snapshot_id})
            
            if doc:
                # ObjectId를 문자열로 변환
                doc["_id"] = str(doc["_id"])
                logger.info(f"Metrics retrieved by portfolio_snapshot_id: {portfolio_snapshot_id}")
                return doc
            else:
                logger.warning(f"Metrics not found for portfolio_snapshot_id: {portfolio_snapshot_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get metrics by portfolio_snapshot_id: {str(e)}")
            raise
    
    async def update_metrics(self, metric_id: str, metrics: BacktestMetrics) -> bool:
        """지표 업데이트"""
        try:
            collection = self.mongodb_client.get_collection(self.collection_name)
            
            if not ObjectId.is_valid(metric_id):
                logger.warning(f"Invalid ObjectId format: {metric_id}")
                return False
            
            update_doc = {
                "total_return": float(metrics.total_return),
                "annualized_return": float(metrics.annualized_return),
                "volatility": float(metrics.volatility),
                "sharpe_ratio": float(metrics.sharpe_ratio),
                "max_drawdown": float(metrics.max_drawdown),
                "var_95": float(metrics.var_95),
                "var_99": float(metrics.var_99),
                "cvar_95": float(metrics.cvar_95),
                "cvar_99": float(metrics.cvar_99),
                "win_rate": float(metrics.win_rate),
                "profit_loss_ratio": float(metrics.profit_loss_ratio),
                "updated_at": datetime.utcnow()
            }
            
            result = await collection.update_one(
                {"_id": ObjectId(metric_id)},
                {"$set": update_doc}
            )
            
            if result.modified_count > 0:
                logger.info(f"Metrics updated: {metric_id}")
                return True
            else:
                logger.warning(f"No metrics updated: {metric_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update metrics: {str(e)}")
            raise
    
    async def delete_metrics(self, metric_id: str) -> bool:
        """지표 삭제"""
        try:
            collection = self.mongodb_client.get_collection(self.collection_name)
            
            if not ObjectId.is_valid(metric_id):
                logger.warning(f"Invalid ObjectId format: {metric_id}")
                return False
            
            result = await collection.delete_one({"_id": ObjectId(metric_id)})
            
            if result.deleted_count > 0:
                logger.info(f"Metrics deleted: {metric_id}")
                return True
            else:
                logger.warning(f"No metrics deleted: {metric_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete metrics: {str(e)}")
            raise
    
    async def get_metrics_history(self, portfolio_id: int, limit: int = 100, offset: int = 0) -> list:
        """포트폴리오별 지표 히스토리 조회"""
        try:
            collection = self.mongodb_client.get_collection(self.collection_name)
            
            # portfolio_snapshot_id를 통해 portfolio_id를 찾기 위해 별도 쿼리 필요
            # 이 부분은 실제 구현에서 portfolio_snapshot_id와 portfolio_id의 관계를 고려해야 함
            cursor = collection.find({}).sort("created_at", -1).skip(offset).limit(limit)
            
            docs = []
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                docs.append(doc)
            
            logger.info(f"Retrieved {len(docs)} metrics from history")
            return docs
            
        except Exception as e:
            logger.error(f"Failed to get metrics history: {str(e)}")
            raise
