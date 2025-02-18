from typing import Optional, Tuple
from fastapi import Request  # type: ignore
from sqlalchemy.orm import Session  # type: ignore
import re

from api.common.exceptions import ConflictException, NotFoundException
from prediction.train.train import PredictionTrain
from prediction.test.test import PredictionTest
from api.models.models import BrandInfoModel, PredictionResultModel
from api.schemas.stock_price import (
    CreateBrandInfoRequest,
    UpdateBrandInfoRequest
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
        """
        文字列形式の数値リストを浮動小数点数のリストに変換する

        引数:
            str_list (str): 数値が含まれた文字列 (例: "3.14, 2.71, -1.0")

        戻り値:
            List[float]: 文字列内の数値を抽出し、`float` 型に変換したリスト
        """
        pattern = r'[-+]?\d*\.\d+|\d+'
        data = re.findall(pattern, str_list)
        data = [float(i) for i in data]

        return data

    @classmethod
    def str_to_str_list(cls, str_list):
        """
        文字列内の日付 (YYYY-MM-DD) を抽出し、リストとして返す

        引数:
            str_list (str): 日付を含む文字列 (例: "2024-01-01, 2023-12-25")

        戻り値:
            List[str]: 抽出された日付文字列のリスト (例: ["2024-01-01", "2023-12-25"])
        """
        pattern = r'(\d{4}-\d{2}-\d{2})'
        result = re.findall(pattern, str_list)

        return result


class StockPriceService:
    def __init__(self, db: Session):
        self.db = db

    def _brand_info_model(
        self,
        create_data: CreateBrandInfoRequest,
        save_path: str
    ) -> BrandInfoModel:
        """
        銘柄情報モデル

        引数:
            create_data (CreateBrandInfoRequest): 登録用の銘柄情報
            save_path (str): モデル保存パス

        戻り値:
            BrandInfoModel: 銘柄情報dbモデル
        """
        db_brand_info = BrandInfoModel(
            brand_name=create_data.brand_name,
            brand_code=create_data.brand_code,
            learned_model_name=save_path,
            user_id=create_data.user_id,
            create_by=create_data.create_by,
            update_by=create_data.update_by,
            create_at=Utils.today(),
            update_at=Utils.today(),
            is_valid=create_data.is_valid
        )
        return db_brand_info

    def _prediction_result_model(
        self,
        future_predictions: str,
        days_list: str,
        create_data: CreateBrandInfoRequest
    ) -> PredictionResultModel:
        """
        予測結果モデル

        引数:
            future_predictions (str): 予測結果(リストを文字列に変換済み)
            days_list (str): 日付(リストを文字列に変換済み)
            create_data (CreateBrandInfoRequest): 登録用の銘柄情報

        戻り値:
            PredictionResultModel: 予測結果dbモデル
        """
        db_pred = PredictionResultModel(
            future_predictions=future_predictions,
            days_list=days_list,
            brand_code=create_data.brand_code,
            user_id=create_data.user_id,
            create_by=create_data.create_by,
            update_by=create_data.update_by,
            create_at=Utils.today(),
            update_at=Utils.today(),
            is_valid=create_data.is_valid
        )
        return db_pred

    def _save(
        self,
        db_data: PredictionResultModel | BrandInfoModel,
        flag: bool
    ) -> None:
        """
        登録・更新

        引数:
            db_data (PredictionResultModel | BrandInfoModel): 各モデル
            flag (bool): Trueなら新規、Falseなら更新

        戻り値:
            None
        """
        if flag:
            self.db.add(db_data)
            self.db.commit()
            self.db.refresh(db_data)
        else:
            self.db.commit()
            self.db.refresh(db_data)

    def _delete(
        self,
        db_data
    ) -> None:
        """
        削除

        引数:
            db_data (PredictionResultModel | BrandInfoModel): 各モデル

        戻り値:
            None
        """
        self.db.delete(db_data)
        self.db.commit()

    def _exist_brand_info_check(
        self,
        user_id: Optional[int],
        data: CreateBrandInfoRequest | UpdateBrandInfoRequest
    ) -> BrandInfoModel | None:
        """
        銘柄テーブルにデータが存在するかチェック

        引数:
            user_id (Optional[int]): ユーザーid (新規登録時はNoneが渡される)
            data (CreateBrandInfoRequest | UpdateBrandInfoRequest): リクエストモデル

        戻り値:
            PredictionResultModel: 予測結果モデル
        """
        if isinstance(data, CreateBrandInfoRequest) and user_id is None:
            user_id = data.user_id

        exist_check = self.db.query(BrandInfoModel).filter(
            BrandInfoModel.brand_code == data.brand_code,
            BrandInfoModel.user_id == user_id,
            BrandInfoModel.is_valid == 1
        ).first()
        return exist_check

    def _exist_prediction_result_check(
        self,
        user_id: Optional[int],
        data: CreateBrandInfoRequest | UpdateBrandInfoRequest
    ) -> PredictionResultModel | None:
        """
        予測結果テーブルにデータが存在するかチェック

        引数:
            user_id (Optional[int]): ユーザーid (新規登録時はNoneが渡される)
            data (CreateBrandInfoRequest | UpdateBrandInfoRequest): リクエストモデル

        戻り値:
            PredictionResultModel: 予測結果モデル
        """
        if isinstance(data, CreateBrandInfoRequest) and user_id is None:
            user_id = data.user_id

        exist_check = self.db.query(PredictionResultModel).filter(
            PredictionResultModel.brand_code == data.brand_code,
            PredictionResultModel.user_id == user_id,
            PredictionResultModel.is_valid
        ).first()
        return exist_check

    def delete_brand_info(
        self,
        user_id: int,
        brand_code: int
    ) -> None:
        """
        銘柄テーブルのデータ削除

        引数:
            user_id (int): ユーザーid
            brand_code (int): 銘柄コード

        戻り値:
            None
        """
        brand_info = self.db.query(BrandInfoModel).filter(
            BrandInfoModel.brand_code == brand_code,
            BrandInfoModel.user_id == user_id,
            BrandInfoModel.is_valid
        ).first()
        if brand_info is None:
            raise NotFoundException("削除対象の銘柄情報が見つかりません。")
        self._delete(brand_info)

    def delete_prediction_result(self, user_id, brand_code):
        """
        予測結果テーブルのデータ削除

        引数:
            user_id (int): ユーザーid
            brand_code (int): 銘柄コード

        戻り値:
            None
        """
        prediction_result = self.db.query(PredictionResultModel).filter(
            PredictionResultModel.brand_code == brand_code,
            PredictionResultModel.user_id == user_id,
            PredictionResultModel.is_valid
        ).first()
        if prediction_result is None:
            raise NotFoundException("削除対象の予測結果データが見つかりません。")
        self._delete(prediction_result)

    def _create(
        self,
        req: Request,
        create_data: CreateBrandInfoRequest
    ) -> None:
        """
        銘柄情報と予測結果の登録処理

        引数:
            req (Request): リクエスト
            create_data (CreateBrandInfoRequest): リクエストデータ

        戻り値:
            None
        """
        if self._exist_brand_info_check(None, create_data) is not None:
            raise ConflictException("銘柄情報は既に登録済みです。")

        future_predictions, days_list, save_path = StockPriceBase.prediction(
            req,
            create_data.brand_name,
            create_data.user_id
        )

        db_brand_info = self._brand_info_model(create_data, save_path)

        # 銘柄情報登録
        self._save(db_brand_info, True)
        db_pred = self._prediction_result_model(
            future_predictions, days_list, create_data)

        if self._exist_prediction_result_check(None, create_data) is not None:
            raise ConflictException("予測結果データは既に登録済みです。")

        # 予測結果登録
        self._save(db_pred, True)

        return None

    def _update(
        self,
        req: Request,
        user_id: int,
        update_data: UpdateBrandInfoRequest
    ) -> None:
        """
        銘柄情報と予測結果の更新処理

        引数:
            req (Request): リクエスト
            user_id (int): ユーザーid
            create_data (UpdateBrandInfoRequest): リクエストデータ

        戻り値:
            None
        """
        db_brand_info = self._exist_brand_info_check(user_id, update_data)
        today = Utils.today()

        if db_brand_info is None:
            raise KeyError("更新対象の銘柄データが存在しません。")

        future_predictions, days_list, save_path = StockPriceBase.prediction(
            req,
            update_data.brand_name,
            user_id
        )

        db_pred = self._exist_prediction_result_check(user_id, update_data)

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

        return None

    def _brand_info_and_prediction_result_delete(
        self,
        user_id: int,
        brand_code: int,
    ) -> None:
        """
        銘柄情報と予測結果の削除処理

        引数:
            user_id (int): ユーザーid
            brand_code (int): 銘柄コード

        戻り値:
            None
        """
        self.delete_brand_info(user_id, brand_code)
        self.delete_prediction_result(user_id, brand_code)

        return None
