from fastapi import APIRouter, HTTPException # type: ignore
from pydantic import BaseModel # type: ignore
from typing import List, Optional

from const.const import HttpStatusCode, ErrorCode
from prediction.train.train import PredictionTrain
from prediction.test.test import PredictionTest


# Pydanticモデル
class StockPriceResponse(BaseModel):
    feature_stock_price: List[float]
    days_list: List[str]


class SuccessResponseModel(BaseModel):
    status: int
    result: Optional[StockPriceResponse] = None


class ErrorMsg(BaseModel):
    code: int
    message: str


router = APIRouter()


@router.get("/get_stock_price")
def get_stock_price(params: str, user_id: int):
    """予測データ取得API"""
    try:
        # インスタンス
        prediction_train = PredictionTrain(params, user_id)
        prediction_test = PredictionTest(params, user_id)
        is_exist = prediction_train.check_brand_info()

        # 学習済みモデル存在チェック
        if is_exist is False:
            prediction_train.main()

        future_predictions, days_list = prediction_test.main()
        return SuccessResponseModel(status=HttpStatusCode.SUCCESS.value, result=StockPriceResponse(feature_stock_price=future_predictions, days_list=days_list))
    except KeyError as e:
        raise HTTPException(status_code=HttpStatusCode.BADREQUEST.value, detail=[ErrorMsg(code=ErrorCode.CHECK_EXIST.value, message=str(e)).dict()])
    except Exception as e:
        raise HTTPException(status_code=HttpStatusCode.SERVER_ERROR.value, detail=[ErrorMsg(code=ErrorCode.SERVER_ERROR.value, message=str(e)).dict()])


@router.post("/create_brand")
async def create_brand():
    return {"message": "Hello World"}
