"""Repository 패턴 기본 클래스"""

from typing import Generic, TypeVar, Type, Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from sqlalchemy.orm import selectinload
from app.models.database import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Repository 기본 클래스"""
    
    def __init__(self, model: Type[ModelType], session: AsyncSession):
        self.model = model
        self.session = session
    
    async def create(self, obj_in: Dict[str, Any]) -> ModelType:
        """단일 객체 생성"""
        db_obj = self.model(**obj_in)
        self.session.add(db_obj)
        await self.session.commit()
        await self.session.refresh(db_obj)
        return db_obj
    
    async def create_many(self, objs_in: List[Dict[str, Any]]) -> List[ModelType]:
        """다중 객체 생성"""
        db_objs = [self.model(**obj_in) for obj_in in objs_in]
        self.session.add_all(db_objs)
        await self.session.commit()
        for db_obj in db_objs:
            await self.session.refresh(db_obj)
        return db_objs
    
    async def get_by_id(self, id: Any) -> Optional[ModelType]:
        """ID로 단일 객체 조회"""
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_field(self, field_name: str, value: Any) -> Optional[ModelType]:
        """특정 필드로 단일 객체 조회"""
        field = getattr(self.model, field_name)
        result = await self.session.execute(
            select(self.model).where(field == value)
        )
        return result.scalar_one_or_none()
    
    async def get_many(
        self, 
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None
    ) -> List[ModelType]:
        """다중 객체 조회"""
        query = select(self.model)
        
        # 필터 적용
        if filters:
            for field_name, value in filters.items():
                field = getattr(self.model, field_name)
                query = query.where(field == value)
        
        # 정렬 적용
        if order_by:
            field = getattr(self.model, order_by)
            query = query.order_by(field)
        
        # 페이징 적용
        query = query.offset(skip).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def update(self, id: Any, obj_in: Dict[str, Any]) -> Optional[ModelType]:
        """객체 업데이트"""
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        db_obj = result.scalar_one_or_none()
        
        if db_obj:
            for field, value in obj_in.items():
                setattr(db_obj, field, value)
            await self.session.commit()
            await self.session.refresh(db_obj)
        
        return db_obj
    
    async def delete(self, id: Any) -> bool:
        """객체 삭제"""
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        db_obj = result.scalar_one_or_none()
        
        if db_obj:
            await self.session.delete(db_obj)
            await self.session.commit()
            return True
        
        return False
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """객체 개수 조회"""
        query = select(func.count(self.model.id))
        
        if filters:
            for field_name, value in filters.items():
                field = getattr(self.model, field_name)
                query = query.where(field == value)
        
        result = await self.session.execute(query)
        return result.scalar()
    
    async def exists(self, filters: Dict[str, Any]) -> bool:
        """객체 존재 여부 확인"""
        query = select(self.model.id)
        
        for field_name, value in filters.items():
            field = getattr(self.model, field_name)
            query = query.where(field == value)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none() is not None

