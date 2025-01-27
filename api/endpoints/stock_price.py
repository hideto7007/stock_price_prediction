import asyncio
from fastapi import APIRouter, HTTPException, Depends  # type: ignore
from sqlalchemy.orm import Session  # type: ignore
import re
from datetime import datetime, timedelta

from const.const import HttpStatusCode, ErrorCode, PredictionResultConst
from prediction.train.train import PredictionTrain
from prediction.test.test import PredictionTest
from api.models.models import BrandInfoModel, BrandModel, PredictionResultModel
from api.databases.databases import get_db
from api.schemas.schemas import (
    ErrorMsg,
    CreateBrandInfo,
    UpdateBrandInfo,
    DeleteBrandInfo,
    BrandInfoList,
    Brand,
)


router = APIRouter()


class StockPriceBase:

    @classmethod
    def prediction(cls, brand_name, user_id):
        """モデル学習を行い株価予測結果を算出する"""
        try:
            prediction_train = PredictionTrain(brand_name, user_id)
            prediction_train.check_brand_info()

            save_path = prediction_train.main()

            prediction_test = PredictionTest(brand_name, user_id)
            future_predictions, days_list = prediction_test.main()

            return str(future_predictions), str(days_list), save_path
        except KeyError as e:
            raise HTTPException(status_code=HttpStatusCode.BADREQUEST.value, detail=[
                                ErrorMsg(code=ErrorCode.CHECK_EXIST.value, message=str(e)).dict()])

    @classmethod
    def str_to_float_list(cls, str_list):
        """文字列形式の数値リストを浮動小数点数のリストに変換"""
        pattern = r'[-+]?\d*\.\d+|\d+'
        data = re.findall(pattern, str_list)
        data = [float(i) for i in data]

        return data

    @classmethod
    def str_to_str_list(cls, str_list):
        """文字列形式の数値リストを文字列のリストに変換"""
        pattern = r'(\d{4}-\d{2}-\d{2})'
        result1 = re.findall(pattern, str_list)

        return result1

    @classmethod
    def get_jst_now(cls):
        """日本時間の現在日時を取得"""
        return datetime.now() + timedelta(hours=9)


class StockPriceService:
    def __init__(self, db: Session):
        self.db = db

    def _brand_info_validation(self, create_data, save_path):
        """銘柄情報のバリデーションチェック"""
        db_brand_info = BrandInfoModel(
            brand_name=create_data.brand_name,
            brand_code=create_data.brand_code,
            learned_model_name=save_path,
            user_id=create_data.user_id,
            create_by=create_data.create_by,
            update_by=create_data.update_by,
            create_at=StockPriceBase.get_jst_now(),
            update_at=StockPriceBase.get_jst_now(),
            is_valid=create_data.is_valid
        )
        return db_brand_info

    def _prediction_result_validation(self, future_predictions, days_list, create_data):
        """予測結果のバリデーションチェック"""
        db_prediction_result = PredictionResultModel(
            future_predictions=future_predictions,
            days_list=days_list,
            brand_code=create_data.brand_code,
            user_id=create_data.user_id,
            create_by=create_data.create_by,
            update_by=create_data.update_by,
            create_at=StockPriceBase.get_jst_now(),
            update_at=StockPriceBase.get_jst_now(),
            is_valid=create_data.is_valid
        )
        return db_prediction_result

    def _save(self, db_data, flag):
        """登録・更新"""
        if flag:
            self.db.add(db_data)
            self.db.commit()
            self.db.refresh(db_data)
        else:
            self.db.commit()
            self.db.refresh(db_data)

    def _delete(self, db_data):
        """削除"""
        self.db.delete(db_data)
        self.db.commit()

    def _exist_brand_info_check(self, data):
        """銘柄テーブルにデータが存在するかチェック"""
        exist_check = self.db.query(BrandInfoModel).filter(
            BrandInfoModel.brand_code == data.brand_code,
            BrandInfoModel.user_id == data.user_id,
            BrandInfoModel.is_valid
        ).first()
        return exist_check

    def _exist_prediction_result_check(self, data):
        """予測結果テーブルにデータが存在するかチェック"""
        exist_check = self.db.query(PredictionResultModel).filter(
            PredictionResultModel.brand_code == data.brand_code,
            PredictionResultModel.user_id == data.user_id,
            PredictionResultModel.is_valid
        ).first()
        return exist_check

    def delete_brand_info(self, data):
        """銘柄テーブルのデータ削除"""
        brand_info = self.db.query(BrandInfoModel).filter(
            BrandInfoModel.brand_code == data.brand_code,
            BrandInfoModel.user_id == data.user_id,
            BrandInfoModel.is_valid
        ).first()
        if brand_info is None:
            raise HTTPException(
                status_code=HttpStatusCode.NOT_FOUND.value,
                detail=[ErrorMsg(code=ErrorCode.NOT_DATA.value,
                                 message="削除対象の銘柄情報が見つかりません。").dict()]
            )
        self._delete(brand_info)

    def delete_prediction_result(self, data):
        """予測結果テーブルのデータ削除"""
        prediction_result = self.db.query(PredictionResultModel).filter(
            PredictionResultModel.brand_code == data.brand_code,
            PredictionResultModel.user_id == data.user_id,
            PredictionResultModel.is_valid
        ).first()
        if prediction_result is None:
            raise HTTPException(
                status_code=HttpStatusCode.NOT_FOUND.value,
                detail=[ErrorMsg(code=ErrorCode.NOT_DATA.value,
                                 message="削除対象の予測結果データが見つかりません。").dict()]
            )
        self._delete(prediction_result)

    def _create(self, create_data: CreateBrandInfo):
        """銘柄情報と予測結果の登録処理"""
        if self._exist_brand_info_check(create_data) is not None:
            raise HTTPException(
                status_code=HttpStatusCode.CONFLICT.value,
                detail=[ErrorMsg(code=ErrorCode.CHECK_EXIST.value,
                                 message="銘柄情報は既に登録済みです。").dict()]
            )

        future_predictions, days_list, save_path = StockPriceBase.prediction(
            create_data.brand_name, create_data.user_id)

        db_brand_info = self._brand_info_validation(create_data, save_path)

        # 銘柄情報登録
        self._save(db_brand_info, True)
        db_prediction_result = self._prediction_result_validation(
            future_predictions, days_list, create_data)

        if self._exist_prediction_result_check(create_data) is not None:
            raise HTTPException(
                status_code=HttpStatusCode.CONFLICT.value,
                detail=[ErrorMsg(code=ErrorCode.CHECK_EXIST.value,
                                 message="予測結果データは既に登録済みです。").dict()]
            )

        # 予測結果登録
        self._save(db_prediction_result, True)

        return {}

    def _update(self, update_data: UpdateBrandInfo):
        """銘柄情報と予測結果の更新処理"""
        db_brand_info = self._exist_brand_info_check(update_data)

        if db_brand_info is None:
            raise HTTPException(
                status_code=HttpStatusCode.NOT_FOUND.value,
                detail=[ErrorMsg(code=ErrorCode.NOT_DATA.value,
                                 message="更新対象の銘柄データが存在しません。").dict()]
            )

        future_predictions, days_list, save_path = StockPriceBase.prediction(
            update_data.brand_name, update_data.user_id)

        db_prediction_result = self._exist_prediction_result_check(update_data)

        db_brand_info.update_by = update_data.update_by
        db_brand_info.learned_model_name = save_path
        db_brand_info.update_at = StockPriceBase.get_jst_now()

        # 銘柄情報更新
        self._save(db_brand_info, False)

        if db_prediction_result is None:
            raise HTTPException(
                status_code=HttpStatusCode.NOT_FOUND.value,
                detail=[ErrorMsg(code=ErrorCode.NOT_DATA.value,
                                 message="更新対象の予測結果データが存在しません。").dict()]
            )

        db_prediction_result.future_predictions = future_predictions
        db_prediction_result.days_list = days_list
        db_prediction_result.update_by = update_data.update_by
        db_prediction_result.update_at = StockPriceBase.get_jst_now()

        # 予測結果更新
        self._save(db_brand_info, False)

        return {}

    def _brand_info_and_prediction_result_delete(self, delete_data: DeleteBrandInfo):
        """銘柄情報と予測結果の削除処理"""
        self.delete_brand_info(delete_data)
        self.delete_prediction_result(delete_data)

        return {}


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
            detail=[ErrorMsg(code=ErrorCode.CHECK_EXIST.value,
                             message="登録されてない予測データです").dict()]
        )

    # レスポンス形式に変換
    res_dict = {
        PredictionResultConst.FUTURE_PREDICTIONS.value: StockPriceBase.str_to_float_list(res.future_predictions),
        PredictionResultConst.DAYS_LIST.value: StockPriceBase.str_to_str_list(res.days_list),
        PredictionResultConst.BRAND_CODE.value: res.brand_code,
        PredictionResultConst.USER_ID.value: res.user_id
    }
    return res_dict


@router.get(
    "/brand_info_list",
    tags=["株価予測"],
    response_model=list[BrandInfoList]
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
    response_model=list[Brand]
)
async def brand(db: Session = Depends(get_db)):
    """全ての銘柄取得API"""
    res = db.query(BrandModel).all()
    return res


@router.post(
    "/create_stock_price",
    tags=["株価予測"]
)
async def create_stock_price(create_data: CreateBrandInfo, db: Session = Depends(get_db)):
    """予測データ登録API"""
    service = StockPriceService(db)
    return service._create(create_data)


@router.put(
    "/upadte_stock_price",
    tags=["株価予測"]
)
async def upadte_stock_price(update_data: UpdateBrandInfo, db: Session = Depends(get_db)):
    """予測データ更新API"""
    service = StockPriceService(db)
    return service._update(update_data)


@router.delete(
    "/delete_stock_price",
    tags=["株価予測"]
)
async def delete_stock_price(delete_data: DeleteBrandInfo, db: Session = Depends(get_db)):
    """予測データ削除API"""
    service = StockPriceService(db)
    return service._brand_info_and_prediction_result_delete(delete_data)


@router.get(
    "/slow",
    tags=["テスト"]
)
async def slow_endpoint():
    """Timeout検証用のAPI(テストでしか使わない)"""
    await asyncio.sleep(5)
    return {"message": "This should timeout"}
