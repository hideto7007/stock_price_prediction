import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request  # type: ignore
from sqlalchemy.orm import Session  # type: ignore

from api.schemas.response import Content
from api.usercase.stock_price import StockPriceBase, StockPriceService
from api.models.models import BrandInfoModel, BrandModel, PredictionResultModel
from api.databases.databases import get_db
from api.schemas.stock_price import (
    CreateBrandInfoRequest,
    UpdateBrandInfoRequest,
    BrandInfoListResponse,
    BrandResponse,
)
from const.const import HttpStatusCode, PredictionResultConst


router = APIRouter()


@router.get(
    "/get_prediction_data",
    tags=["株価予測"]
)
def get_data(brand_code: int, user_id: int, db: Session = Depends(get_db)):
    """予測データ取得API"""
    res = db.query(PredictionResultModel).filter(
        PredictionResultModel.brand_code == brand_code,
        PredictionResultModel.user_id == user_id,
        PredictionResultModel.is_valid
    ).first()

    if res is None:
        raise HTTPException(
            status_code=HttpStatusCode.NOT_FOUND.value,
            detail=[Content(result="登録されてない予測データです").dict()]
        )

    # レスポンス形式に変換
    res_dict = {
        PredictionResultConst.FUTURE_PREDICTIONS.value:
        StockPriceBase.str_to_float_list(res.future_predictions),
        PredictionResultConst.DAYS_LIST.value:
        StockPriceBase.str_to_str_list(res.days_list),
        PredictionResultConst.BRAND_CODE.value: res.brand_code,
        PredictionResultConst.USER_ID.value: res.user_id
    }
    return res_dict


@router.get(
    "/brand_info_list{user_id}",
    tags=["株価予測"],
    response_model=list[BrandInfoListResponse]
)
async def brand_list(user_id: int, db: Session = Depends(get_db)):
    """対象ユーザーの学習ずみ銘柄情報取得API"""
    res = db.query(BrandInfoModel).filter(
        BrandInfoModel.user_id == user_id,
        BrandInfoModel.is_valid
    ).all()

    return res


@router.get(
    "/brand",
    tags=["株価予測"],
    response_model=list[BrandResponse]
)
async def brand(db: Session = Depends(get_db)):
    """全ての銘柄取得API"""
    res = db.query(BrandModel).all()
    return res


@router.post(
    "/create_stock_price",
    tags=["株価予測"]
)
async def create_stock_price(
    request: Request,
    create_data: CreateBrandInfoRequest,
    db: Session = Depends(get_db)
):
    """予測データ登録API"""
    service = StockPriceService(db)
    return service._create(request, create_data)


@router.put(
    "/upadte_stock_price/{user_id}",
    tags=["株価予測"]
)
async def upadte_stock_price(
    request: Request,
    user_id: int,
    update_data: UpdateBrandInfoRequest,
    db: Session = Depends(get_db)
):
    """予測データ更新API"""
    service = StockPriceService(db)
    return service._update(request, user_id, update_data)


@router.delete(
    "/delete_stock_price/{user_id}/{brand_code}",
    tags=["株価予測"]
)
async def delete_stock_price(
    user_id: int,
    brand_code: int,
    db: Session = Depends(get_db)
):
    """予測データ削除API"""
    service = StockPriceService(db)
    return service._brand_info_and_prediction_result_delete(
        user_id, brand_code
    )


@router.get(
    "/slow",
    tags=["テスト"]
)
async def slow_endpoint():
    """Timeout検証用のAPI(テストでしか使わない)"""
    await asyncio.sleep(5)
    return {"result": "This should timeout"}
