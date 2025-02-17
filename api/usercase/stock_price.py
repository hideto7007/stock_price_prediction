from typing import Tuple
from fastapi import Request  # type: ignore
from sqlalchemy.orm import Session  # type: ignore
import re
from datetime import datetime, timedelta

from api.common.exceptions import ConflictException, NotFoundException
from prediction.train.train import PredictionTrain
from prediction.test.test import PredictionTest
from api.models.models import BrandInfoModel, PredictionResultModel
from api.schemas.schemas import (
    CreateBrandInfo,
    UpdateBrandInfo,
    DeleteBrandInfo
)
from utils.utils import Utils


class StockPriceBase:

    @staticmethod
    def prediction(
        req: Request,
        brand_name: str,
        user_id: int
    ) -> Tuple[str, str, str]:
        """
        モデル学習を行い株価予測結果を算出する

        引数:
            req (Request): リクエスト
            brand_name (str): 銘柄
            user_id (int): ユーザーid

        戻り値:
            Tuple[str, str, str]:
                - future_predictions (str): 予測結果
                - days_list (str): 日付リスト
                - save_path (str): 保存ファイルパス
        """
        try:
            prediction_train = PredictionTrain(req, brand_name, user_id)
            prediction_train.check_brand_info()

            save_path = prediction_train.main()

            prediction_test = PredictionTest(req, brand_name, user_id)
            future_predictions, days_list = prediction_test.main()

            return str(future_predictions), str(days_list), save_path
        except KeyError as e:
            raise e

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

    def _prediction_result_validation(
        self,
        future_predictions,
        days_list,
        create_data
    ):
        """予測結果のバリデーションチェック"""
        db_pred = PredictionResultModel(
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
        return db_pred

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
            raise NotFoundException("削除対象の銘柄情報が見つかりません。")
        self._delete(brand_info)

    def delete_prediction_result(self, data):
        """予測結果テーブルのデータ削除"""
        prediction_result = self.db.query(PredictionResultModel).filter(
            PredictionResultModel.brand_code == data.brand_code,
            PredictionResultModel.user_id == data.user_id,
            PredictionResultModel.is_valid
        ).first()
        if prediction_result is None:
            raise NotFoundException("削除対象の予測結果データが見つかりません。")
        self._delete(prediction_result)

    def _create(self, req: Request, create_data: CreateBrandInfo):
        """銘柄情報と予測結果の登録処理"""
        if self._exist_brand_info_check(create_data) is not None:
            raise ConflictException("銘柄情報は既に登録済みです。")

        future_predictions, days_list, save_path = StockPriceBase.prediction(
            req,
            create_data.brand_name,
            create_data.user_id
        )

        db_brand_info = self._brand_info_validation(create_data, save_path)

        # 銘柄情報登録
        self._save(db_brand_info, True)
        db_pred = self._prediction_result_validation(
            future_predictions, days_list, create_data)

        if self._exist_prediction_result_check(create_data) is not None:
            raise ConflictException("予測結果データは既に登録済みです。")

        # 予測結果登録
        self._save(db_pred, True)

        return {}

    def _update(self, req: Request, update_data: UpdateBrandInfo):
        """銘柄情報と予測結果の更新処理"""
        db_brand_info = self._exist_brand_info_check(update_data)
        today = Utils.today()

        if db_brand_info is None:
            raise KeyError("更新対象の銘柄データが存在しません。")

        future_predictions, days_list, save_path = StockPriceBase.prediction(
            req,
            update_data.brand_name,
            update_data.user_id
        )

        db_pred = self._exist_prediction_result_check(update_data)

        db_brand_info.update_by = Utils.column_str(update_data.update_by)
        db_brand_info.learned_model_name = Utils.column_str(save_path)
        db_brand_info.update_at = Utils.column_datetime(today)

        # 銘柄情報更新
        self._save(db_brand_info, False)

        if db_pred is None:
            raise KeyError("更新対象の予測結果データが存在しません。")

        db_pred.future_predictions = Utils.column_str(future_predictions)
        db_pred.days_list = Utils.column_str(days_list)
        db_pred.update_by = Utils.column_str(update_data.update_by)
        db_pred.update_at = Utils.column_datetime(today)

        # 予測結果更新
        self._save(db_brand_info, False)

        return {}

    def _brand_info_and_prediction_result_delete(
        self,
        delete_data: DeleteBrandInfo
    ):
        """銘柄情報と予測結果の削除処理"""
        self.delete_brand_info(delete_data)
        self.delete_prediction_result(delete_data)

        return {}
