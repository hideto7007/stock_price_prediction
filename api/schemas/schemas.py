from typing_extensions import Final
from pydantic import BaseModel, Field, field_validator  # type: ignore
from pydantic.generics import GenericModel
# from datetime import datetime
from typing import Generic, Optional, TypeVar, List

# ジェネリック型 T を定義
T = TypeVar("T")


class StockPriceResponse(BaseModel):
    feature_stock_price: List[float]
    days_list: List[str]


# Response クラスをジェネリック型で定義
class Response(GenericModel, Generic[T]):
    data: Optional[List[T] | T] = None
    headers: Optional[dict[str, str]] = None


class BrandInfoBase(BaseModel):
    brand_name: str = Field(..., max_length=128)
    brand_code: int = Field(...)
    user_id: int = Field(...)
    create_by: str = Field(..., max_length=128)
    update_by: str = Field(..., max_length=128)
    is_valid: bool

    @field_validator('brand_name', 'create_by', 'update_by')
    def validate_string_fields(cls, v, info):
        if not isinstance(v, str):
            raise ValueError(f"{info.name}は文字列のみです")
        if len(v) > 128:
            raise ValueError(f"{info.name}の文字数オーバーです")
        return v

    @field_validator('brand_code', 'user_id')
    def validate_integer_fields(cls, v, info):
        if not isinstance(v, int):
            raise ValueError(f"{info.name}は整数値のみです")
        return v


class PredictionResultBase(BaseModel):
    future_predictions: str
    days_list: str
    brand_code: int
    user_id: int
    create_by: str
    update_by: str
    is_valid: bool


class CreateBrandInfo(BrandInfoBase):
    pass


# この更新だけBaseを継承しない
class UpdateBrandInfo(BaseModel):
    brand_name: str = Field(..., max_length=128)
    brand_code: int = Field(...)
    update_by: str = Field(..., max_length=128)
    user_id: int = Field(...)

    @field_validator('brand_name', 'update_by')
    def validate_string_fields(cls, v, info):
        if not isinstance(v, str):
            raise ValueError(f"{info.name}は文字列のみです")
        if len(v) > 128:
            raise ValueError(f"{info.name}の文字数オーバーです")
        return v

    @field_validator('brand_code', 'user_id')
    def validate_integer_fields(cls, v, info):
        if not isinstance(v, int):
            raise ValueError(f"{info.name}は整数値のみです")
        return v


class DeleteBrandInfo(BaseModel):
    brand_code: int = Field(...)
    user_id: int = Field(...)


class CreatePredictionResult(PredictionResultBase):
    pass


class UpdatePredictionResult(PredictionResultBase):
    pass


class DeletePredictionResult(PredictionResultBase):
    pass


class BrandInfoList(BaseModel):
    brand_info_id: int
    brand_name: str
    brand_code: int
    learned_model_name: str
    user_id: int

    class Config:
        from_attributes = True


class Brand(BaseModel):
    brand_name: str
    brand_code: int

    class Config:
        from_attributes = True


class PredictionResult(BaseModel):
    future_predictions: str
    days_list: str
    brand_code: int
    user_id: int

    class Config:
        from_attributes = True
