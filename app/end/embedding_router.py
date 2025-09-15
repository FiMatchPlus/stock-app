"""임베딩 모델 API 라우터"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
import asyncio

from app.models.schemas import (
    EmbeddingRequest, EmbeddingResponse, ModelTrainingRequest, ModelTrainingResponse,
    EmbeddingSearchRequest, EmbeddingSearchResponse, ElasticsearchDocument
)
from app.services.embedding_storage_service import EmbeddingStorageService
from app.services.embedding_model_service import EmbeddingModelService
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 라우터 생성
router = APIRouter(prefix="/embeddings", tags=["embeddings"])

# 서비스 인스턴스
embedding_storage_service = EmbeddingStorageService()
embedding_model_service = EmbeddingModelService()


@router.post("/generate", response_model=List[EmbeddingResponse])
async def generate_embeddings(request: EmbeddingRequest):
    """임베딩을 생성합니다."""
    try:
        logger.info(f"Generating embeddings for {len(request.symbols)} symbols")
        
        embeddings = await embedding_model_service.generate_embeddings(request)
        
        return embeddings
        
    except Exception as e:
        logger.error("Failed to generate embeddings", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-and-store")
async def generate_and_store_embeddings(
    symbols: List[str],
    sequence_length: int = 60,
    embedding_dim: int = 64,
    model_type: str = "autoencoder",
    include_features: bool = True
):
    """임베딩을 생성하고 Elasticsearch에 저장합니다."""
    try:
        logger.info(f"Generating and storing embeddings for {len(symbols)} symbols")
        
        result = await embedding_storage_service.generate_and_store_embeddings(
            symbols=symbols,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            model_type=model_type,
            include_features=include_features
        )
        
        return {
            "success": True,
            "message": "Embeddings generated and stored successfully",
            "data": result
        }
        
    except Exception as e:
        logger.error("Failed to generate and store embeddings", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=List[EmbeddingSearchResponse])
async def search_similar_embeddings(request: EmbeddingSearchRequest):
    """유사한 임베딩을 검색합니다."""
    try:
        logger.info(f"Searching for similar embeddings, top_k={request.top_k}")
        
        results = await embedding_storage_service.search_similar_embeddings(request)
        
        return results
        
    except Exception as e:
        logger.error("Failed to search similar embeddings", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbol/{symbol}", response_model=Optional[EmbeddingSearchResponse])
async def get_embedding_by_symbol(symbol: str):
    """특정 종목의 임베딩을 조회합니다."""
    try:
        logger.info(f"Getting embedding for symbol {symbol}")
        
        result = await embedding_storage_service.get_embedding_by_symbol(symbol)
        
        if result is None:
            raise HTTPException(status_code=404, detail=f"Embedding not found for symbol {symbol}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get embedding for symbol {symbol}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/symbol/{symbol}")
async def update_embedding(
    symbol: str,
    sequence_length: int = 60,
    embedding_dim: int = 64,
    model_type: str = "autoencoder",
    include_features: bool = True
):
    """특정 종목의 임베딩을 업데이트합니다."""
    try:
        logger.info(f"Updating embedding for symbol {symbol}")
        
        success = await embedding_storage_service.update_embedding(
            symbol=symbol,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            model_type=model_type,
            include_features=include_features
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update embedding")
        
        return {
            "success": True,
            "message": f"Embedding updated successfully for symbol {symbol}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update embedding for symbol {symbol}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/symbol/{symbol}")
async def delete_embedding(symbol: str):
    """특정 종목의 임베딩을 삭제합니다."""
    try:
        logger.info(f"Deleting embedding for symbol {symbol}")
        
        success = await embedding_storage_service.delete_embedding(symbol)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Embedding not found for symbol {symbol}")
        
        return {
            "success": True,
            "message": f"Embedding deleted successfully for symbol {symbol}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete embedding for symbol {symbol}", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-update")
async def batch_update_embeddings(
    symbols: List[str],
    sequence_length: int = 60,
    embedding_dim: int = 64,
    model_type: str = "autoencoder",
    include_features: bool = True
):
    """여러 종목의 임베딩을 배치로 업데이트합니다."""
    try:
        logger.info(f"Batch updating embeddings for {len(symbols)} symbols")
        
        result = await embedding_storage_service.batch_update_embeddings(
            symbols=symbols,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            model_type=model_type,
            include_features=include_features
        )
        
        return {
            "success": True,
            "message": "Batch update completed",
            "data": result
        }
        
    except Exception as e:
        logger.error("Failed to batch update embeddings", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-by-filters", response_model=List[EmbeddingSearchResponse])
async def search_embeddings_by_filters(
    sector: Optional[str] = None,
    volatility_min: Optional[float] = None,
    volatility_max: Optional[float] = None,
    beta_min: Optional[float] = None,
    beta_max: Optional[float] = None,
    model_type: Optional[str] = None,
    size: int = 100
):
    """필터 조건으로 임베딩을 검색합니다."""
    try:
        # 필터 구성
        filters = {}
        
        if sector:
            filters["sector"] = sector
        if volatility_min is not None or volatility_max is not None:
            filters["volatility_range"] = [
                volatility_min or 0.0,
                volatility_max or float('inf')
            ]
        if beta_min is not None or beta_max is not None:
            filters["beta_range"] = [
                beta_min or 0.0,
                beta_max or float('inf')
            ]
        if model_type:
            filters["model_type"] = model_type
        
        logger.info(f"Searching embeddings by filters: {filters}")
        
        results = await embedding_storage_service.get_embeddings_by_filters(filters, size)
        
        return results
        
    except Exception as e:
        logger.error("Failed to search embeddings by filters", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_embedding_statistics():
    """임베딩 통계를 조회합니다."""
    try:
        logger.info("Getting embedding statistics")
        
        statistics = await embedding_storage_service.get_embedding_statistics()
        
        return {
            "success": True,
            "data": statistics
        }
        
    except Exception as e:
        logger.error("Failed to get embedding statistics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=ModelTrainingResponse)
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """모델을 훈련합니다."""
    try:
        logger.info(f"Training model with {len(request.symbols)} symbols")
        
        # 백그라운드에서 훈련 실행
        result = await embedding_model_service.train_model(request)
        
        return result
        
    except Exception as e:
        logger.error("Failed to train model", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """임베딩 서비스 헬스 체크"""
    try:
        # Elasticsearch 연결 확인
        from app.utils.elasticsearch_client import elasticsearch_client
        
        if elasticsearch_client.client and elasticsearch_client.client.ping():
            return {
                "status": "healthy",
                "elasticsearch": "connected",
                "service": "embedding"
            }
        else:
            return {
                "status": "unhealthy",
                "elasticsearch": "disconnected",
                "service": "embedding"
            }
            
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "embedding"
        }
