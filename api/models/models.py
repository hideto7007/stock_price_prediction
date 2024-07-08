from sqlalchemy import Column, Integer, String, DateTime, Boolean # type: ignore
from datetime import datetime

from api.databases.databases import Base


class BrandModel(Base):
    __tablename__ = 'brand'

    brand_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    brand_name = Column(String, nullable=False, index=True)
    brand_code = Column(Integer, nullable=False)
    create_at = Column(DateTime, default=datetime.astimezone, nullable=False)
    create_by = Column(String, nullable=False)
    update_at = Column(DateTime, default=datetime.astimezone, onupdate=datetime.astimezone, nullable=False)
    update_by = Column(String, nullable=False)
    is_valid = Column(Boolean, default=True, nullable=False)


class BrandInfoModel(Base):
    __tablename__ = 'brand_info'

    brand_code_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    brand_name = Column(String, nullable=False)
    brand_code = Column(Integer, nullable=False)
    learned_model_name = Column(String, nullable=False)
    user_id = Column(Integer, nullable=False)
    create_at = Column(DateTime, default=datetime.astimezone, nullable=False)
    create_by = Column(String, nullable=False)
    update_at = Column(DateTime, default=datetime.astimezone, onupdate=datetime.astimezone, nullable=False)
    update_by = Column(String, nullable=False)
    is_valid = Column(Boolean, default=True, nullable=False)


class PredictionResultModel(Base):
    __tablename__ = 'prediction_result'

    prediction_result_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    future_predictions = Column(String, nullable=True, default='[]')
    days_list = Column(String, nullable=True, default='[]')
    brand_code = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)
    create_at = Column(DateTime, default=datetime.astimezone, nullable=False)
    create_by = Column(String, nullable=False)
    update_at = Column(DateTime, default=datetime.astimezone, onupdate=datetime.astimezone, nullable=False)
    update_by = Column(String, nullable=False)
    is_valid = Column(Boolean, default=True, nullable=False)


# # データベースとモデルを同期（既存のテーブルがある場合は実行しません）
# Base.metadata.create_all(bind=engine)
