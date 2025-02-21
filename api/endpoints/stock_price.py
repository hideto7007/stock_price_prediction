import asyncio
from typing import List
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from api.common.exceptions import HttpExceptionHandler, NotFoundException
from api.common.response import ValidationResponse
from api.schemas.response import Content
from api.schemas.validation import ValidatonModel
from api.usercase.stock_price import StockPriceBase, StockPriceService
from api.databases.databases import get_db
from api.schemas.stock_price import (
    BrandInfoListValidationModel,
    BrandInfoResponse,
    CreateBrandInfoRequest,
    DeleteBrandInfoValidationModel,
    GetPredictionDataValidationModel,
    PredictionResultResponse,
    UpdateBrandInfoRequest,
    BrandInfoListResponse,
    UpdateBrandInfoValidationModel
)
from api.validation.stock_price import (
    GetPredictionDataValidation,
    BrandInfoListValidation,
    CreateBrandInfoRequestValidation,
    UpdateBrandInfoRequestValidation,
    DeleteBrandInfoRequestValidation
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
        # バリデーションチェック
        valid = GetPredictionDataValidation(
            GetPredictionDataValidationModel(
                user_id=user_id,
                brand_code=brand_code
            )
        )

        if len(valid) > 0:
            return ValidationResponse(
                content=Content[list[ValidatonModel]](
                    result=valid
                ).model_dump()
            )

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
    "/brand_info_list/{user_id}",
    tags=["株価予測"],
    responses=Swagger.swagger_responses({
        200: [
            {
                "brand_code": 1111,
                "brand_info_id": 1,
                "user_id": 1,
                "create_by": "TEST",
                "update_by": "TEST",
                "is_valid": True,
                "brand_name": "テスト1",
                "learned_model_name": "/test/save/1/test.pth",
                "create_at": "2024-07-15T13:19:41",
                "update_at": "2024-07-15T13:19:41"
            },
            {
                "brand_code": 2222,
                "brand_info_id": 2,
                "user_id": 1,
                "create_by": "TEST",
                "update_by": "TEST",
                "is_valid": True,
                "brand_name": "テスト2",
                "learned_model_name": "/test/save/1/test.pth",
                "create_at": "2024-07-15T13:19:41",
                "update_at": "2024-07-15T13:19:41"
            },
        ],
        500: "予期せぬエラーが発生しました"
    })
)
async def brand_info_list(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db)
):
    """
        対象ユーザーの学習ずみ銘柄情報取得API

        引数:
            request (Request): fastapiリクエスト
            user_id (int): ユーザーid
            db (Session): dbインスタンス
        戻り値:
            Response: レスポンス型
    """
    try:
        # バリデーションチェック
        valid = BrandInfoListValidation(
            BrandInfoListValidationModel(
                user_id=user_id,
            )
        )

        if len(valid) > 0:
            return ValidationResponse(
                content=Content[list[ValidatonModel]](
                    result=valid
                ).model_dump()
            )

        service = StockPriceService(db)
        res_list = service.get_brand_list(user_id)

        result: List[BrandInfoListResponse] = []
        for res in res_list:
            result.append(
                BrandInfoListResponse(
                    brand_info_id=Utils.int(res.brand_info_id),
                    brand_name=Utils.str(res.brand_name),
                    brand_code=Utils.int(res.brand_code),
                    learned_model_name=Utils.str(res.learned_model_name),
                    user_id=Utils.int(res.user_id),
                )
            )

        # レスポンス形式に変換
        context = Content[List[BrandInfoListResponse]](
            result=result
        )
        return JSONResponse(
            status_code=HttpStatusCode.SUCCESS.value,
            content=context.model_dump()
        )
    except Exception as e:
        return await HttpExceptionHandler.main_handler(request, e)


@router.get(
    "/brand",
    tags=["株価予測"],
    responses=Swagger.swagger_responses({
        200: [
            {
                "brand_name": "日本水産",
                "brand_code": 1332
            },
            {
                "brand_name": "INPEX",
                "brand_code": 1605
            },
        ],
        500: "予期せぬエラーが発生しました"
    })
)
async def brand(
    request: Request,
    db: Session = Depends(get_db)
):
    """
        全ての銘柄取得API

        引数:
            request (Request): fastapiリクエスト
            db (Session): dbインスタンス
        戻り値:
            Response: レスポンス型
    """
    try:
        service = StockPriceService(db)
        res_list = service.get_brand_all_list()

        result: List[BrandInfoResponse] = []
        for res in res_list:
            result.append(
                BrandInfoResponse(
                    brand_name=Utils.str(res.brand_name),
                    brand_code=Utils.int(res.brand_code),
                )
            )

        # レスポンス形式に変換
        context = Content[List[BrandInfoResponse]](
            result=result
        )
        return JSONResponse(
            status_code=HttpStatusCode.SUCCESS.value,
            content=context.model_dump()
        )
    except Exception as e:
        return await HttpExceptionHandler.main_handler(request, e)


@router.post(
    "/create_stock_price",
    tags=["株価予測"],
    responses=Swagger.swagger_responses({
        200: "予測データの登録成功",
        409: "銘柄情報は既に登録済みです。又は、予測結果データは既に登録済みです。",
        500: "予期せぬエラーが発生しました"
    })
)
async def create_stock_price(
    request: Request,
    data: CreateBrandInfoRequest,
    db: Session = Depends(get_db)
):
    """
        予測データ登録API

        引数:
            request (Request): fastapiリクエスト
            data (CreateBrandInfoRequest): リクエストボディ
            db (Session): dbインスタンス
        戻り値:
            Response: レスポンス型
    """
    try:
        # バリデーションチェック
        valid = CreateBrandInfoRequestValidation(data)

        if len(valid) > 0:
            return ValidationResponse(
                content=Content[list[ValidatonModel]](
                    result=valid
                ).model_dump()
            )

        service = StockPriceService(db)
        service._create(request, data)

        # レスポンス形式に変換
        context = Content[str](
            result="予測データの登録成功"
        )
        return JSONResponse(
            status_code=HttpStatusCode.SUCCESS.value,
            content=context.model_dump()
        )
    except Exception as e:
        return await HttpExceptionHandler.main_handler(request, e)


@router.put(
    "/upadte_stock_price/{user_id}",
    tags=["株価予測"],
    responses=Swagger.swagger_responses({
        200: "予測データの更新成功",
        409: "銘柄情報は既に登録済みです。又は、予測結果データは既に登録済みです。",
        500: "予期せぬエラーが発生しました"
    })
)
async def upadte_stock_price(
    request: Request,
    user_id: int,
    data: UpdateBrandInfoRequest,
    db: Session = Depends(get_db)
):
    """
        予測データ更新API

        引数:
            request (Request): fastapiリクエスト
            user_id (int): ユーザーid
            data (UpdateBrandInfoRequest): リクエストボディ
            db (Session): dbインスタンス
        戻り値:
            Response: レスポンス型
    """
    try:
        # バリデーションチェック
        valid = UpdateBrandInfoRequestValidation(
            UpdateBrandInfoValidationModel(
                user_id=user_id,
                brand_name=data.brand_name,
                brand_code=data.brand_code,
                update_by=data.update_by,
            )
        )

        if len(valid) > 0:
            return ValidationResponse(
                content=Content[list[ValidatonModel]](
                    result=valid
                ).model_dump()
            )

        service = StockPriceService(db)
        service._update(request, user_id, data)

        # レスポンス形式に変換
        context = Content[str](
            result="予測データの更新成功"
        )
        return JSONResponse(
            status_code=HttpStatusCode.SUCCESS.value,
            content=context.model_dump()
        )
    except Exception as e:
        return await HttpExceptionHandler.main_handler(request, e)


@router.delete(
    "/delete_stock_price/{user_id}/{brand_code}",
    tags=["株価予測"]
)
async def delete_stock_price(
    request: Request,
    user_id: int,
    brand_code: int,
    db: Session = Depends(get_db)
):
    """
        予測データ削除API

        引数:
            request (Request): fastapiリクエスト
            user_id (int): ユーザーid
            brand_code (int): 銘柄コード
            db (Session): dbインスタンス
        戻り値:
            Response: レスポンス型
    """
    try:
        # バリデーションチェック
        valid = DeleteBrandInfoRequestValidation(
            DeleteBrandInfoValidationModel(
                user_id=user_id,
                brand_code=brand_code
            )
        )

        if len(valid) > 0:
            return ValidationResponse(
                content=Content[list[ValidatonModel]](
                    result=valid
                ).model_dump()
            )

        service = StockPriceService(db)
        service._brand_info_and_prediction_result_delete(
            user_id, brand_code
        )

        # レスポンス形式に変換
        context = Content[str](
            result="予測データの削除成功"
        )
        return JSONResponse(
            status_code=HttpStatusCode.SUCCESS.value,
            content=context.model_dump()
        )
    except Exception as e:
        return await HttpExceptionHandler.main_handler(request, e)


@router.get(
    "/slow",
    tags=["テスト"]
)
async def slow_endpoint():
    """Timeout検証用のAPI(テストでしか使わない)"""
    await asyncio.sleep(5)
    return {"result": "This should timeout"}
