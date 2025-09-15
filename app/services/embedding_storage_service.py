"""임베딩 저장 서비스"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from app.models.schemas import (
    EmbeddingResponse, ElasticsearchDocument, 
    EmbeddingSearchRequest, EmbeddingSearchResponse
)
from app.services.embedding_model_service import EmbeddingModelService
from app.services.feature_extraction_service import FeatureExtractionService
from app.utils.elasticsearch_client import elasticsearch_client
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingStorageService:
    """임베딩 저장 및 검색 서비스"""
    
    def __init__(self):
        self.embedding_service = EmbeddingModelService()
        self.feature_service = FeatureExtractionService()
    
    async def generate_and_store_embeddings(
        self, 
        symbols: List[str],
        sequence_length: int = 60,
        embedding_dim: int = 64,
        model_type: str = "autoencoder",
        include_features: bool = True
    ) -> Dict[str, Any]:
        """임베딩을 생성하고 Elasticsearch에 저장합니다."""
        try:
            logger.info(f"Generating embeddings for {len(symbols)} symbols")
            
            # 임베딩 생성 요청
            from app.models.schemas import EmbeddingRequest
            request = EmbeddingRequest(
                symbols=symbols,
                sequence_length=sequence_length,
                embedding_dim=embedding_dim,
                model_type=model_type,
                include_features=include_features
            )
            
            # 임베딩 생성
            embeddings = await self.embedding_service.generate_embeddings(request)
            
            if not embeddings:
                raise ValueError("No embeddings generated")
            
            # Elasticsearch 문서로 변환
            documents = []
            for embedding_response in embeddings:
                document = ElasticsearchDocument(
                    symbol=embedding_response.symbol,
                    embedding=embedding_response.embedding,
                    features=embedding_response.features,
                    model_info=embedding_response.model_info,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                documents.append(document.dict())
            
            # Elasticsearch에 저장
            result = await elasticsearch_client.bulk_index_documents(documents)
            
            return {
                "success": result["success"],
                "total_symbols": len(symbols),
                "indexed": result["indexed"],
                "failed": result["failed"],
                "errors": result.get("errors", []),
                "embeddings": [
                    {
                        "symbol": emb.symbol,
                        "embedding_dim": len(emb.embedding),
                        "has_features": emb.features is not None
                    }
                    for emb in embeddings
                ]
            }
            
        except Exception as e:
            logger.error("Failed to generate and store embeddings", error=str(e))
            raise
    
    async def search_similar_embeddings(
        self, 
        request: EmbeddingSearchRequest
    ) -> List[EmbeddingSearchResponse]:
        """유사한 임베딩을 검색합니다."""
        try:
            # Elasticsearch에서 검색
            results = await elasticsearch_client.search_similar_embeddings(
                query_embedding=request.query_embedding,
                top_k=request.top_k,
                filters=request.filters
            )
            
            # 응답 형식으로 변환
            responses = []
            for result in results:
                response = EmbeddingSearchResponse(
                    symbol=result["symbol"],
                    score=result["score"],
                    embedding=result["embedding"],
                    features=result["features"],
                    model_info=result["model_info"]
                )
                responses.append(response)
            
            logger.info(f"Found {len(responses)} similar embeddings")
            return responses
            
        except Exception as e:
            logger.error("Failed to search similar embeddings", error=str(e))
            raise
    
    async def get_embedding_by_symbol(self, symbol: str) -> Optional[EmbeddingSearchResponse]:
        """특정 종목의 임베딩을 조회합니다."""
        try:
            document = await elasticsearch_client.get_document(symbol)
            
            if document:
                response = EmbeddingSearchResponse(
                    symbol=document["symbol"],
                    score=1.0,  # 자기 자신이므로 완전 일치
                    embedding=document["embedding"],
                    features=document["features"],
                    model_info=document["model_info"]
                )
                return response
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get embedding for symbol {symbol}", error=str(e))
            return None
    
    async def update_embedding(
        self, 
        symbol: str,
        sequence_length: int = 60,
        embedding_dim: int = 64,
        model_type: str = "autoencoder",
        include_features: bool = True
    ) -> bool:
        """특정 종목의 임베딩을 업데이트합니다."""
        try:
            # 새로운 임베딩 생성
            from app.models.schemas import EmbeddingRequest
            request = EmbeddingRequest(
                symbols=[symbol],
                sequence_length=sequence_length,
                embedding_dim=embedding_dim,
                model_type=model_type,
                include_features=include_features
            )
            
            embeddings = await self.embedding_service.generate_embeddings(request)
            
            if not embeddings:
                raise ValueError(f"No embedding generated for {symbol}")
            
            embedding_response = embeddings[0]
            
            # 문서 업데이트
            document = ElasticsearchDocument(
                symbol=embedding_response.symbol,
                embedding=embedding_response.embedding,
                features=embedding_response.features,
                model_info=embedding_response.model_info,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            success = await elasticsearch_client.index_document(document.dict())
            
            if success:
                logger.info(f"Updated embedding for symbol {symbol}")
            else:
                logger.warning(f"Failed to update embedding for symbol {symbol}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update embedding for symbol {symbol}", error=str(e))
            return False
    
    async def delete_embedding(self, symbol: str) -> bool:
        """특정 종목의 임베딩을 삭제합니다."""
        try:
            success = await elasticsearch_client.delete_document(symbol)
            
            if success:
                logger.info(f"Deleted embedding for symbol {symbol}")
            else:
                logger.warning(f"Failed to delete embedding for symbol {symbol}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete embedding for symbol {symbol}", error=str(e))
            return False
    
    async def get_embeddings_by_filters(
        self, 
        filters: Dict[str, Any],
        size: int = 100
    ) -> List[EmbeddingSearchResponse]:
        """필터 조건으로 임베딩을 검색합니다."""
        try:
            results = await elasticsearch_client.search_by_filters(filters, size)
            
            responses = []
            for result in results:
                response = EmbeddingSearchResponse(
                    symbol=result["symbol"],
                    score=1.0,  # 필터 검색이므로 점수 없음
                    embedding=result["embedding"],
                    features=result["features"],
                    model_info=result["model_info"]
                )
                responses.append(response)
            
            logger.info(f"Found {len(responses)} embeddings matching filters")
            return responses
            
        except Exception as e:
            logger.error("Failed to get embeddings by filters", error=str(e))
            return []
    
    async def get_embedding_statistics(self) -> Dict[str, Any]:
        """임베딩 통계를 조회합니다."""
        try:
            # 전체 문서 수 조회
            all_embeddings = await elasticsearch_client.search_by_filters({}, size=10000)
            
            if not all_embeddings:
                return {
                    "total_count": 0,
                    "sectors": {},
                    "model_types": {},
                    "embedding_dims": {},
                    "avg_volatility": 0.0,
                    "avg_beta": 0.0
                }
            
            # 통계 계산
            sectors = {}
            model_types = {}
            embedding_dims = {}
            volatilities = []
            betas = []
            
            for doc in all_embeddings:
                # 섹터 통계
                sector = doc.get("features", {}).get("sector", "Unknown")
                sectors[sector] = sectors.get(sector, 0) + 1
                
                # 모델 타입 통계
                model_type = doc.get("model_info", {}).get("model_type", "unknown")
                model_types[model_type] = model_types.get(model_type, 0) + 1
                
                # 임베딩 차원 통계
                dim = len(doc.get("embedding", []))
                embedding_dims[dim] = embedding_dims.get(dim, 0) + 1
                
                # 변동성과 베타
                volatility = doc.get("features", {}).get("volatility")
                beta = doc.get("features", {}).get("beta")
                
                if volatility is not None:
                    volatilities.append(volatility)
                if beta is not None:
                    betas.append(beta)
            
            statistics = {
                "total_count": len(all_embeddings),
                "sectors": sectors,
                "model_types": model_types,
                "embedding_dims": embedding_dims,
                "avg_volatility": float(sum(volatilities) / len(volatilities)) if volatilities else 0.0,
                "avg_beta": float(sum(betas) / len(betas)) if betas else 0.0,
                "volatility_range": {
                    "min": float(min(volatilities)) if volatilities else 0.0,
                    "max": float(max(volatilities)) if volatilities else 0.0
                },
                "beta_range": {
                    "min": float(min(betas)) if betas else 0.0,
                    "max": float(max(betas)) if betas else 0.0
                }
            }
            
            return statistics
            
        except Exception as e:
            logger.error("Failed to get embedding statistics", error=str(e))
            return {}
    
    async def batch_update_embeddings(
        self, 
        symbols: List[str],
        sequence_length: int = 60,
        embedding_dim: int = 64,
        model_type: str = "autoencoder",
        include_features: bool = True
    ) -> Dict[str, Any]:
        """여러 종목의 임베딩을 배치로 업데이트합니다."""
        try:
            logger.info(f"Batch updating embeddings for {len(symbols)} symbols")
            
            # 기존 임베딩 삭제
            deleted_count = 0
            for symbol in symbols:
                if await self.delete_embedding(symbol):
                    deleted_count += 1
            
            # 새로운 임베딩 생성 및 저장
            result = await self.generate_and_store_embeddings(
                symbols=symbols,
                sequence_length=sequence_length,
                embedding_dim=embedding_dim,
                model_type=model_type,
                include_features=include_features
            )
            
            result["deleted_count"] = deleted_count
            return result
            
        except Exception as e:
            logger.error("Failed to batch update embeddings", error=str(e))
            raise
