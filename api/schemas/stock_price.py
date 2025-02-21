from pydantic import BaseModel
from typing import List

###########################
# ユーザーケースでも使うモデル #
###########################


class StockPriceResponse(BaseModel):
    feature_stock_price: List[float]
    days_list: List[str]


class BrandInfoBase(BaseModel):
    brand_name: str
    brand_code: int
    user_id: int
    create_by: str
    update_by: str
    is_valid: bool


class PredictionResultBase(BaseModel):
    future_predictions: str
    days_list: str
    brand_code: int
    user_id: int
    create_by: str
    update_by: str
    is_valid: bool


class BrandInfo(BaseModel):
    brand_name: str
    brand_code: int


##################
# リクエストモデル #
##################
class CreateBrandInfoRequest(BrandInfoBase):
    pass


# この更新だけBaseを継承しない
class UpdateBrandInfoRequest(BaseModel):
    brand_name: str
    brand_code: int
    update_by: str


##################
# レスポンスモデル #
##################

class BrandInfoListResponse(BaseModel):
    brand_info_id: int
    brand_name: str
    brand_code: int
    learned_model_name: str
    user_id: int


class BrandInfoResponse(BrandInfo):
    pass


class PredictionResultResponse(BaseModel):
    future_predictions: list[float]
    days_list: List[str]
    brand_code: int
    user_id: int


#####################
# バリデーションモデル #
#####################
class GetPredictionDataValidationModel(BaseModel):
    user_id: int
    brand_code: int


class BrandInfoListValidationModel(BaseModel):
    user_id: int


class UpdateBrandInfoValidationModel(UpdateBrandInfoRequest):
    user_id: int


class DeleteBrandInfoValidationModel(GetPredictionDataValidationModel):
    pass
