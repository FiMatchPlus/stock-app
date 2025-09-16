"""Elasticsearch 클라이언트"""

from elasticsearch import Elasticsearch
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ElasticsearchClient:
    """Elasticsearch 클라이언트"""
    
    def __init__(self):
        self.client = None
        self.index_name = "stock_embeddings"
        self.connect()
    
    def connect(self):
        """Elasticsearch에 연결합니다."""
        try:
            # Elasticsearch 설정
            es_config = {
                'hosts': [settings.elasticsearch_url],
                'timeout': 30,
                'max_retries': 3,
                'retry_on_timeout': True
            }
            
            # 인증이 필요한 경우
            if hasattr(settings, 'elasticsearch_username') and settings.elasticsearch_username:
                es_config['basic_auth'] = (
                    settings.elasticsearch_username,
                    settings.elasticsearch_password
                )
            
            self.client = Elasticsearch(**es_config)
            
            # 연결 테스트
            if self.client.ping():
                logger.info("Elasticsearch connected successfully")
                self._create_index_if_not_exists()
            else:
                raise ConnectionError("Failed to connect to Elasticsearch")
                
        except Exception as e:
            logger.error("Failed to connect to Elasticsearch", error=str(e))
            raise
    
    def _create_index_if_not_exists(self):
        """인덱스가 없으면 생성합니다."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                mapping = {
                    "mappings": {
                        "properties": {
                            "symbol": {
                                "type": "keyword"
                            },
                            "embedding": {
                                "type": "dense_vector",
                                "dims": 64,  # 기본 차원, 실제로는 동적
                                "index": True,
                                "similarity": "cosine"
                            },
                            "features": {
                                "properties": {
                                    "symbol": {"type": "keyword"},
                                    "volatility": {"type": "float"},
                                    "beta": {"type": "float"},
                                    "sector": {"type": "keyword"},
                                    "market_cap": {"type": "float"},
                                    "pe_ratio": {"type": "float"},
                                    "pb_ratio": {"type": "float"},
                                    "dividend_yield": {"type": "float"}
                                }
                            },
                            "model_info": {
                                "properties": {
                                    "model_id": {"type": "keyword"},
                                    "embedding_dim": {"type": "integer"},
                                    "sequence_length": {"type": "integer"},
                                    "model_name": {"type": "keyword"}
                                }
                            },
                            "created_at": {
                                "type": "date"
                            },
                            "updated_at": {
                                "type": "date"
                            }
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "index": {
                            "knn": True,
                            "knn.algo_param.ef_search": 100
                        }
                    }
                }
                
                self.client.indices.create(index=self.index_name, body=mapping)
                logger.info(f"Created index: {self.index_name}")
            else:
                logger.info(f"Index {self.index_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to create index {self.index_name}", error=str(e))
            raise
    
    async def index_document(self, document: Dict[str, Any]) -> bool:
        """문서를 인덱싱합니다."""
        try:
            # 임베딩 차원에 따라 인덱스 매핑 업데이트
            embedding_dim = len(document.get('embedding', []))
            if embedding_dim > 0:
                await self._update_embedding_dimension(embedding_dim)
            
            # 문서 저장
            response = self.client.index(
                index=self.index_name,
                id=document['symbol'],
                body=document
            )
            
            if response['result'] in ['created', 'updated']:
                logger.info(f"Document indexed successfully", symbol=document['symbol'])
                return True
            else:
                logger.warning(f"Unexpected response from Elasticsearch", response=response)
                return False
                
        except Exception as e:
            logger.error(f"Failed to index document", error=str(e), symbol=document.get('symbol'))
            return False
    
    async def bulk_index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """여러 문서를 벌크로 인덱싱합니다."""
        try:
            if not documents:
                return {"success": True, "indexed": 0, "failed": 0}
            
            # 임베딩 차원 확인 및 업데이트
            embedding_dim = len(documents[0].get('embedding', []))
            if embedding_dim > 0:
                await self._update_embedding_dimension(embedding_dim)
            
            # 벌크 요청 구성
            bulk_body = []
            for doc in documents:
                bulk_body.append({
                    "index": {
                        "_index": self.index_name,
                        "_id": doc['symbol']
                    }
                })
                bulk_body.append(doc)
            
            # 벌크 실행
            response = self.client.bulk(body=bulk_body)
            
            # 결과 분석
            indexed = 0
            failed = 0
            errors = []
            
            for item in response['items']:
                if 'index' in item:
                    if item['index']['status'] in [200, 201]:
                        indexed += 1
                    else:
                        failed += 1
                        errors.append(item['index'].get('error', 'Unknown error'))
            
            result = {
                "success": failed == 0,
                "indexed": indexed,
                "failed": failed,
                "errors": errors
            }
            
            logger.info(f"Bulk indexing completed", indexed=indexed, failed=failed)
            return result
            
        except Exception as e:
            logger.error("Failed to bulk index documents", error=str(e))
            return {"success": False, "indexed": 0, "failed": len(documents), "errors": [str(e)]}
    
    async def search_similar_embeddings(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """유사한 임베딩을 검색합니다."""
        try:
            # 쿼리 구성
            query = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": top_k,
                    "num_candidates": top_k * 2
                }
            }
            
            # 필터 추가
            if filters:
                query = {
                    "bool": {
                        "must": [query],
                        "filter": self._build_filters(filters)
                    }
                }
            
            # 검색 실행
            response = self.client.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": top_k,
                    "_source": True
                }
            )
            
            # 결과 처리
            results = []
            for hit in response['hits']['hits']:
                result = {
                    "symbol": hit['_source']['symbol'],
                    "score": hit['_score'],
                    "embedding": hit['_source']['embedding'],
                    "features": hit['_source']['features'],
                    "model_info": hit['_source']['model_info']
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar embeddings")
            return results
            
        except Exception as e:
            logger.error("Failed to search similar embeddings", error=str(e))
            return []
    
    async def get_document(self, symbol: str) -> Optional[Dict[str, Any]]:
        """특정 문서를 조회합니다."""
        try:
            response = self.client.get(
                index=self.index_name,
                id=symbol
            )
            
            if response['found']:
                return response['_source']
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get document", error=str(e), symbol=symbol)
            return None
    
    async def delete_document(self, symbol: str) -> bool:
        """문서를 삭제합니다."""
        try:
            response = self.client.delete(
                index=self.index_name,
                id=symbol
            )
            
            if response['result'] == 'deleted':
                logger.info(f"Document deleted successfully", symbol=symbol)
                return True
            else:
                logger.warning(f"Unexpected response from Elasticsearch", response=response)
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete document", error=str(e), symbol=symbol)
            return False
    
    async def search_by_filters(self, filters: Dict[str, Any], size: int = 100) -> List[Dict[str, Any]]:
        """필터 조건으로 검색합니다."""
        try:
            query = {
                "bool": {
                    "filter": self._build_filters(filters)
                }
            }
            
            response = self.client.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": size,
                    "_source": True
                }
            )
            
            results = []
            for hit in response['hits']['hits']:
                results.append(hit['_source'])
            
            return results
            
        except Exception as e:
            logger.error("Failed to search by filters", error=str(e))
            return []
    
    def _build_filters(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """필터 조건을 Elasticsearch 쿼리로 변환합니다."""
        filter_queries = []
        
        for field, value in filters.items():
            if field == "sector":
                filter_queries.append({
                    "term": {
                        "features.sector": value
                    }
                })
            elif field == "volatility_range":
                min_vol, max_vol = value
                filter_queries.append({
                    "range": {
                        "features.volatility": {
                            "gte": min_vol,
                            "lte": max_vol
                        }
                    }
                })
            elif field == "beta_range":
                min_beta, max_beta = value
                filter_queries.append({
                    "range": {
                        "features.beta": {
                            "gte": min_beta,
                            "lte": max_beta
                        }
                    }
                })
            elif field == "model_name":
                filter_queries.append({
                    "term": {
                        "model_info.model_name": value
                    }
                })
        
        return filter_queries
    
    async def _update_embedding_dimension(self, embedding_dim: int):
        """임베딩 차원을 업데이트합니다."""
        try:
            # 현재 매핑 확인
            mapping = self.client.indices.get_mapping(index=self.index_name)
            current_dims = mapping[self.index_name]['mappings']['properties']['embedding'].get('dims', 64)
            
            if current_dims != embedding_dim:
                # 매핑 업데이트
                update_mapping = {
                    "properties": {
                        "embedding": {
                            "type": "dense_vector",
                            "dims": embedding_dim,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                }
                
                self.client.indices.put_mapping(
                    index=self.index_name,
                    body=update_mapping
                )
                
                logger.info(f"Updated embedding dimension to {embedding_dim}")
                
        except Exception as e:
            logger.warning(f"Failed to update embedding dimension", error=str(e))
    
    def close(self):
        """연결을 종료합니다."""
        if self.client:
            self.client.close()
            logger.info("Elasticsearch connection closed")


# 전역 클라이언트 인스턴스
elasticsearch_client = ElasticsearchClient()
