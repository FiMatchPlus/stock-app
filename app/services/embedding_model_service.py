"""시계열 임베딩 모델 서비스"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import uuid
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.models.schemas import (
    TimeSeriesData, FinancialFeatures, EmbeddingRequest, 
    EmbeddingResponse, ModelTrainingRequest, ModelTrainingResponse
)
from app.services.feature_extraction_service import FeatureExtractionService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TimeSeriesAutoEncoder(nn.Module):
    """시계열 AutoEncoder 모델"""
    
    def __init__(self, input_dim: int, embedding_dim: int, sequence_length: int, feature_dim: int = 0):
        super(TimeSeriesAutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(128 + feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        self.decoder_lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.output_layer = nn.Linear(128, input_dim)
        
    def forward(self, x, features=None):
        # Encoder
        lstm_out, (hidden, cell) = self.encoder_lstm(x)
        lstm_features = lstm_out[:, -1, :]  # 마지막 시점의 출력
        
        # 피처가 있는 경우 결합
        if features is not None and self.feature_dim > 0:
            combined = torch.cat([lstm_features, features], dim=1)
        else:
            combined = lstm_features
            
        embedding = self.encoder_fc(combined)
        
        # Decoder
        decoded = self.decoder_fc(embedding)
        decoded = decoded.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        lstm_out, _ = self.decoder_lstm(decoded)
        output = self.output_layer(lstm_out)
        
        return output, embedding


class LSTMAutoEncoder(nn.Module):
    """LSTM 기반 AutoEncoder 모델"""
    
    def __init__(self, input_dim: int, embedding_dim: int, sequence_length: int, feature_dim: int = 0):
        super(LSTMAutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=embedding_dim * 2,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(embedding_dim * 4 + feature_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim * 4)
        )
        
        self.decoder = nn.LSTM(
            input_size=embedding_dim * 4,
            hidden_size=embedding_dim * 2,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        self.output_layer = nn.Linear(embedding_dim * 4, input_dim)
        
    def forward(self, x, features=None):
        # Encoder
        lstm_out, (hidden, cell) = self.encoder(x)
        lstm_features = lstm_out[:, -1, :]  # 마지막 시점의 출력
        
        # 피처가 있는 경우 결합
        if features is not None and self.feature_dim > 0:
            combined = torch.cat([lstm_features, features], dim=1)
        else:
            combined = lstm_features
            
        embedding = self.encoder_fc(combined)
        
        # Decoder
        decoded = self.decoder_fc(embedding)
        decoded = decoded.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        lstm_out, _ = self.decoder(decoded)
        output = self.output_layer(lstm_out)
        
        return output, embedding


class EmbeddingModelService:
    """시계열 임베딩 모델 서비스"""
    
    def __init__(self):
        self.feature_service = FeatureExtractionService()
        self.models = {}  # 모델 저장소
        self.scalers = {}  # 스케일러 저장소
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    async def train_model(self, request: ModelTrainingRequest) -> ModelTrainingResponse:
        """모델을 훈련합니다."""
        try:
            model_id = str(uuid.uuid4())
            logger.info(f"Starting model training", model_id=model_id, symbols_count=len(request.symbols))
            
            # 시계열 데이터 수집
            time_series_data = []
            financial_features = []
            
            for symbol in request.symbols:
                try:
                    # 시계열 데이터 추출
                    ts_data = await self.feature_service.extract_time_series_data(
                        symbol=symbol,
                        sequence_length=request.sequence_length
                    )
                    time_series_data.append(ts_data)
                    
                    # 금융 피처 추출
                    features = await self.feature_service.extract_financial_features(symbol)
                    financial_features.append(features)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract data for {symbol}", error=str(e))
                    continue
            
            if len(time_series_data) < 5:
                raise ValueError("Insufficient data for training")
            
            # 데이터 전처리
            X, X_features = self._prepare_training_data(time_series_data, financial_features)
            
            # 모델 생성
            model = self._create_model(
                model_type="autoencoder",
                input_dim=X.shape[2],
                embedding_dim=request.embedding_dim,
                sequence_length=request.sequence_length,
                feature_dim=X_features.shape[1] if X_features is not None else 0
            )
            
            # 훈련 실행
            training_loss, validation_loss, final_metrics = await self._train_model_async(
                model, X, X_features, request
            )
            
            # 모델 저장
            self.models[model_id] = model
            self.scalers[model_id] = self._get_scalers()
            
            training_time = sum(final_metrics.get('training_time', [0]))
            
            return ModelTrainingResponse(
                model_id=model_id,
                training_loss=training_loss,
                validation_loss=validation_loss,
                final_metrics=final_metrics,
                training_time=training_time
            )
            
        except Exception as e:
            logger.error("Failed to train model", error=str(e))
            raise
    
    async def generate_embeddings(self, request: EmbeddingRequest) -> List[EmbeddingResponse]:
        """임베딩을 생성합니다."""
        try:
            # 모델이 없으면 기본 모델 사용
            if not self.models:
                await self._create_default_model()
            
            model_id = list(self.models.keys())[0]
            model = self.models[model_id]
            scalers = self.scalers[model_id]
            
            results = []
            
            for symbol in request.symbols:
                try:
                    # 시계열 데이터 추출
                    ts_data = await self.feature_service.extract_time_series_data(
                        symbol=symbol,
                        sequence_length=request.sequence_length
                    )
                    
                    # 금융 피처 추출 (옵션)
                    features = None
                    if request.include_features:
                        features = await self.feature_service.extract_financial_features(symbol)
                    
                    # 데이터 전처리
                    X, X_features = self._prepare_inference_data(ts_data, features, scalers)
                    
                    # 임베딩 생성
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X).to(self.device)
                        X_features_tensor = torch.FloatTensor(X_features).to(self.device) if X_features is not None else None
                        
                        _, embedding = model(X_tensor, X_features_tensor)
                        embedding_vector = embedding.cpu().numpy().flatten().tolist()
                    
                    # 응답 생성
                    response = EmbeddingResponse(
                        symbol=symbol,
                        embedding=embedding_vector,
                        features=features,
                        model_info={
                            "model_id": model_id,
                            "embedding_dim": len(embedding_vector),
                            "sequence_length": request.sequence_length,
                            "model_type": request.model_type
                        }
                    )
                    results.append(response)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for {symbol}", error=str(e))
                    continue
            
            return results
            
        except Exception as e:
            logger.error("Failed to generate embeddings", error=str(e))
            raise
    
    def _create_model(self, model_type: str, input_dim: int, embedding_dim: int, 
                     sequence_length: int, feature_dim: int = 0) -> nn.Module:
        """모델을 생성합니다."""
        if model_type == "autoencoder":
            return TimeSeriesAutoEncoder(input_dim, embedding_dim, sequence_length, feature_dim)
        elif model_type == "lstm":
            return LSTMAutoEncoder(input_dim, embedding_dim, sequence_length, feature_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _prepare_training_data(self, time_series_data: List[TimeSeriesData], 
                              financial_features: List[FinancialFeatures]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """훈련 데이터를 준비합니다."""
        # 시계열 데이터 준비
        sequences = []
        for ts_data in time_series_data:
            # 가격, 거래량, 수익률을 결합
            sequence = np.column_stack([
                ts_data.prices,
                ts_data.volumes,
                ts_data.returns
            ])
            sequences.append(sequence)
        
        X = np.array(sequences)
        
        # 금융 피처 준비
        X_features = None
        if financial_features:
            feature_vectors = []
            for features in financial_features:
                feature_vector = [
                    features.volatility,
                    features.beta,
                    features.pe_ratio or 0.0,
                    features.pb_ratio or 0.0,
                    features.dividend_yield or 0.0
                ]
                feature_vectors.append(feature_vector)
            X_features = np.array(feature_vectors)
        
        return X, X_features
    
    def _prepare_inference_data(self, ts_data: TimeSeriesData, features: Optional[FinancialFeatures], 
                               scalers: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """추론 데이터를 준비합니다."""
        # 시계열 데이터 준비
        sequence = np.column_stack([
            ts_data.prices,
            ts_data.volumes,
            ts_data.returns
        ])
        X = sequence.reshape(1, -1, sequence.shape[1])
        
        # 스케일링
        if 'sequence_scaler' in scalers:
            X = scalers['sequence_scaler'].transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # 금융 피처 준비
        X_features = None
        if features is not None:
            feature_vector = np.array([[
                features.volatility,
                features.beta,
                features.pe_ratio or 0.0,
                features.pb_ratio or 0.0,
                features.dividend_yield or 0.0
            ]])
            
            if 'feature_scaler' in scalers:
                X_features = scalers['feature_scaler'].transform(feature_vector)
            else:
                X_features = feature_vector
        
        return X, X_features
    
    def _get_scalers(self) -> Dict[str, Any]:
        """스케일러를 반환합니다."""
        return {
            'sequence_scaler': StandardScaler(),
            'feature_scaler': MinMaxScaler()
        }
    
    async def _train_model_async(self, model: nn.Module, X: np.ndarray, X_features: Optional[np.ndarray], 
                                request: ModelTrainingRequest) -> Tuple[List[float], List[float], Dict[str, float]]:
        """비동기 모델 훈련을 실행합니다."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._train_model_sync, 
            model, X, X_features, request
        )
    
    def _train_model_sync(self, model: nn.Module, X: np.ndarray, X_features: Optional[np.ndarray], 
                         request: ModelTrainingRequest) -> Tuple[List[float], List[float], Dict[str, float]]:
        """동기 모델 훈련을 실행합니다."""
        model = model.to(self.device)
        
        # 데이터 분할
        n_samples = len(X)
        n_val = int(n_samples * request.validation_split)
        n_train = n_samples - n_val
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_train = X[train_indices]
        X_val = X[val_indices]
        X_features_train = X_features[train_indices] if X_features is not None else None
        X_features_val = X_features[val_indices] if X_features is not None else None
        
        # 데이터로더 생성
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.FloatTensor(X_features_train).to(self.device) if X_features_train is not None else None
        )
        train_loader = DataLoader(train_dataset, batch_size=request.batch_size, shuffle=True)
        
        # 옵티마이저와 손실함수
        optimizer = optim.Adam(model.parameters(), lr=request.learning_rate)
        criterion = nn.MSELoss()
        
        # 훈련 루프
        training_losses = []
        validation_losses = []
        
        for epoch in range(request.epochs):
            # 훈련
            model.train()
            train_loss = 0.0
            for batch_X, batch_features in train_loader:
                optimizer.zero_grad()
                
                if batch_features is not None:
                    output, _ = model(batch_X, batch_features)
                else:
                    output, _ = model(batch_X)
                
                loss = criterion(output, batch_X)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            training_losses.append(avg_train_loss)
            
            # 검증
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                X_features_val_tensor = torch.FloatTensor(X_features_val).to(self.device) if X_features_val is not None else None
                
                if X_features_val_tensor is not None:
                    output, _ = model(X_val_tensor, X_features_val_tensor)
                else:
                    output, _ = model(X_val_tensor)
                
                val_loss = criterion(output, X_val_tensor).item()
                validation_losses.append(val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        final_metrics = {
            'final_train_loss': training_losses[-1],
            'final_val_loss': validation_losses[-1],
            'min_val_loss': min(validation_losses),
            'training_time': [0.1] * len(training_losses)  # 실제로는 시간 측정
        }
        
        return training_losses, validation_losses, final_metrics
    
    async def _create_default_model(self):
        """기본 모델을 생성합니다."""
        model_id = str(uuid.uuid4())
        model = self._create_model("autoencoder", 3, 64, 60, 5)
        self.models[model_id] = model
        self.scalers[model_id] = self._get_scalers()
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
