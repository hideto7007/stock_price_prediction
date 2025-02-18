import asyncio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from api.common.exceptions import HttpExceptionHandler, NotFoundException
from api.schemas.response import Content
from api.usercase.stock_price import StockPriceBase, StockPriceService
from api.models.models import BrandModel
from api.databases.databases import get_db
from api.schemas.stock_price import (
    CreateBrandInfoRequest,
    PredictionResultResponse,
    UpdateBrandInfoRequest,
    BrandInfoListResponse,
    BrandResponse,
)
from const.const import HttpStatusCode
from utils.utils import Swagger, Utils


router = APIRouter()


@router.get(
    "/get_prediction_data/{user_id}/{brand_code}",
    tags=["株価予測"],
    responses=Swagger.swagger_responses({
        200: {
            "future_predictions": ["3.14", "2.71", "-1.0"],
            "days_list": ["2024-01-01", "2023-12-25", "2022-12-25"],
            "brand_code": 1234,
            "user_id": 1
        },
        404: "登録されてない予測データです。",
        500: "予期せぬエラーが発生しました"
    })
)
async def get_prediction_data(
    request: Request,
    user_id: int,
    brand_code: int,
    db: Session = Depends(get_db)
):
    """
        予測データ取得API

        引数:
            request (Request): fastapiリクエスト
            user_id (int): ユーザーid
            brand_code (int): 銘柄コード
            db (Session): dbインスタンス

        戻り値:
            Response: レスポンス型
    """

    try:
        service = StockPriceService(db)
        res = service.get_prediction_info(user_id, brand_code)

        if res is None:
            raise NotFoundException("登録されてない予測データです。")

        # レスポンス形式に変換
        context = Content[PredictionResultResponse](
            result=PredictionResultResponse(
                future_predictions=StockPriceBase.str_to_float_list(
                    res.future_predictions
                ),
                days_list=StockPriceBase.str_to_str_list(res.days_list),
                brand_code=Utils.int(res.brand_code),
                user_id=Utils.int(res.user_id)
            )
        )
        return JSONResponse(
            status_code=HttpStatusCode.SUCCESS.value,
            content=context.model_dump()
        )
    except Exception as e:
        return await HttpExceptionHandler.main_handler(request, e)


@router.get(
    "/brand_info_list{user_id}",
    tags=["株価予測"],
    response_model=list[BrandInfoListResponse]
)
async def brand_list(user_id: int, db: Session = Depends(get_db)):
    """対象ユーザーの学習ずみ銘柄情報取得API"""

    service = StockPriceService(db)
    res = service.get_brand_list(user_id)

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
