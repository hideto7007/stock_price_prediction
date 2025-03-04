
from unittest.mock import MagicMock, patch

from api.common.exceptions import (
    ConflictException, NotFoundException, SqlException
)
from api.models.models import BrandInfoModel, PredictionResultModel
from api.endpoints.stock_price import StockPriceBase
from api.schemas.stock_price import (
    CreateBrandInfoRequest, UpdateBrandInfoRequest
)
from api.schemas.test import RequestId, TestRequest
from api.usercase.stock_price import StockPriceService
from tests.test_case import TestBase, TestBaseAPI

# モック
REQUEST = 'fastapi.Request'
TRAIN_MAIN = 'prediction.train.train.PredictionTrain.main'
TEST_MAIN = 'prediction.test.test.PredictionTest.main'
STOCK_PRICE_BASE_PREDICTION = (
    'api.endpoints.stock_price.StockPriceBase.prediction'
)


class TestStockPriceBase(TestBase):

    @patch(TEST_MAIN)
    @patch(TRAIN_MAIN)
    @patch(REQUEST)
    def test_prediction_success_01(self, _request, _train_main, _test_main):
        """
        正常系： 予測データが正しく取得できること
        """
        # テストデータ
        data = [100.0, 200.0, 300.0]
        days = ["2023-07-01", "2023-07-02", "2023-07-03"]
        brand_name = "住友電気工業"
        brand_code = 5802
        _train_main.return_value = "test.pth"
        _test_main.return_value = (data, days)
        _request.return_value = TestRequest(state=RequestId(), method="GET")
        brand_info = {
            "住友電気工業": "5802",
            "INPEX": "1605",
            "コムシスホールディングス": "1721"
        }

        future_predictions, days_list, save_path = StockPriceBase.prediction(
            _request, brand_name, brand_info, brand_code
        )

        self.assertEqual(future_predictions, str(data))
        self.assertEqual(days_list, str(days))
        self.assertEqual(save_path, "test.pth")

    @patch(REQUEST)
    @patch(TEST_MAIN)
    @patch(TRAIN_MAIN)
    def test_prediction_key_error_01(self, _request, _train_main, _test_main):
        """
        異常系： 予測データが正しく取得できなく例外が発生すること
        """
        # テストデータ
        data = [100.0, 200.0, 300.0]
        days = ["2023-07-01", "2023-07-02", "2023-07-03"]
        brand_name = "住友電気工業"
        brand_code = 5802
        _train_main.return_value = "test.pth"
        _test_main.return_value = (data, days)
        _request.return_value = TestRequest(state=RequestId(), method="GET")
        brand_info = {
            "INPEX": "1605",
            "コムシスホールディングス": "1721"
        }

        # KeyError が発生することを確認
        with self.assertRaisesRegex(
            KeyError,
            "住友電気工業"
        ):
            StockPriceBase.prediction(
                _request, brand_name, brand_info, brand_code
            )

    def test_str_to_float_list(self):
        """
        正常系： 文字列形式の数値リストを浮動小数点数のリストに変換する
        """
        # テストデータ
        data = "3.14, 2.71, -1.0"
        expected_result = [3.14, 2.71, -1.0]
        result = StockPriceBase.str_to_float_list(data)

        self.assertEqual(result, expected_result)

    def test_str_to_str_list(self):
        """
        正常系： 文字列内の日付 (YYYY-MM-DD) を抽出し、リストとして返す
        """
        # テストデータ
        data = "2024-01-01, 2023-12-25"
        expected_result = ["2024-01-01", "2023-12-25"]
        result = StockPriceBase.str_to_str_list(data)

        self.assertEqual(result, expected_result)


class TestStockPriceService(TestBaseAPI):

    def setUp(self):
        """テストごとにデータベースを初期化"""
        super().setUp()
        self.mock_db = MagicMock()
        self.db.begin()  # トランザクション開始

    def tearDown(self):
        """テストごとにデータベースをリセット"""
        self.db.rollback()  # 変更をロールバック
        self.db.close()
        super().tearDown()

    def expected_get_brand_info(
        self,
        user_id: int,
        brand_code: int
    ) -> BrandInfoModel | None:
        """テストの検証時に使用する銘柄情報取得"""
        return self.db.query(BrandInfoModel).filter(
            BrandInfoModel.user_id == user_id,
            BrandInfoModel.brand_code == brand_code,
            BrandInfoModel.is_valid == 1
        ).first()

    def test_get_prediction_info_success_01(self):
        """
        正常系： 予測情報取得できること
        """
        service = StockPriceService(self.db)
        result = service.get_prediction_info(1, 7203)

        self.assertEqual(result.brand_code, 7203)
        self.assertEqual(result.user_id, 1)

    def test_get_prediction_info_success_02(self):
        """
        正常系： 予測情報取得できないこと
        - brand_codeが存在しない
        """
        service = StockPriceService(self.db)
        result = service.get_prediction_info(1, 7024)

        self.assertEqual(result, None)

    def test_get_prediction_info_success_03(self):
        """
        正常系： 予測情報取得できないこと
        - is_validがFalse
        """
        service = StockPriceService(self.db)
        result = service.get_prediction_info(1, 7202)

        self.assertEqual(result, None)

    def test_get_brand_list_success_01(self):
        """
        正常系： ユーザーの学習ずみ銘柄情報取得できること
        """
        service = StockPriceService(self.db)
        result = service.get_brand_list(1)

        # ✅ 期待されるデータ
        expected_data = [
            {"brand_code": 7203, "brand_name": "トヨタ自動車"},
            {"brand_code": 7205, "brand_name": "日野自動車"},
        ]

        self.assertEqual(len(result), len(expected_data))
        for i, r in enumerate(result):
            self.assertEqual(r.brand_code, expected_data[i]["brand_code"])
            self.assertEqual(r.brand_name, expected_data[i]["brand_name"])

    def test_get_brand_list_success_02(self):
        """
        正常系： ユーザーの学習ずみ銘柄情報取得できないこと
        - ユーザーidが異なり、0件
        """
        service = StockPriceService(self.db)
        result = service.get_brand_list(2)

        self.assertEqual(len(result), 0)

    def test_get_brand_all_list_success_01(self):
        """
        正常系： 全ての銘柄取得できること
        """
        service = StockPriceService(self.db)
        result = service.get_brand_all_list()

        self.assertEqual(len(result), 10)

    @patch(REQUEST)
    def test_create_error_01(self, _request):
        """
        異常系： 登録する銘柄情報が既に存在していること
        """
        data = CreateBrandInfoRequest(
            brand_name="トヨタ自動車",
            brand_code=7203,
            user_id=1,
            create_by="test",
            update_by="test",
            is_valid=True
        )
        _request.return_value = TestRequest(state=RequestId(), method="POST")
        service = StockPriceService(self.db)

        with self.assertRaisesRegex(
            ConflictException,
            "銘柄情報は既に登録済みです。"
        ):
            service._create(_request, data)

    @patch(REQUEST)
    def test_create_error_02(self, _request):
        """
        異常系： 登録する予測結果が既に存在していること
        """
        data = CreateBrandInfoRequest(
            brand_name="三菱自動車工業",
            brand_code=7211,
            user_id=1,
            create_by="test",
            update_by="test",
            is_valid=True
        )
        _request.return_value = TestRequest(state=RequestId(), method="POST")
        service = StockPriceService(self.db)

        with self.assertRaisesRegex(
            ConflictException,
            "予測結果データは既に登録済みです。"
        ):
            service._create(_request, data)

    @patch(STOCK_PRICE_BASE_PREDICTION)
    @patch(REQUEST)
    def test_create_success_01(self, _request, _stock_price_base_prediction):
        """
        異常系： 銘柄情報と予測結果が新規登録できること
        """
        data = CreateBrandInfoRequest(
            brand_name="マツダ",
            brand_code=7261,
            user_id=1,
            create_by="test",
            update_by="test",
            is_valid=True
        )
        _request.return_value = TestRequest(state=RequestId(), method="POST")
        _stock_price_base_prediction.return_value = (
            "[1, 2, 3]",
            "[2025-02-11, 2025-02-12, 2025-02-13]",
            "test.pth"
        )
        service = StockPriceService(self.db)
        service._create(_request, data)

        # 登録されているか確認
        brand_info_check = service._exist_brand_info_check(1, data)
        prediction_result_check = service._exist_prediction_result_check(
            1, data
        )

        # 登録されていればNoneではない
        self.assertIsNotNone(brand_info_check)
        self.assertIsNotNone(prediction_result_check)

        # テスト終了後に登録データを削除
        service._brand_info_and_prediction_result_delete(
            1, 7261
        )

    @patch(REQUEST)
    def test_update_error_01(self, _request):
        """
        異常系： 更新対象の銘柄情報が存在していないこと
        """
        data = UpdateBrandInfoRequest(
            brand_name="テスト",
            brand_code=1111,
            update_by="test_update",
        )
        _request.return_value = TestRequest(state=RequestId(), method="PUT")
        service = StockPriceService(self.db)

        with self.assertRaisesRegex(
            KeyError,
            "更新対象の銘柄データが存在しません。"
        ):
            service._update(_request, 1, data)

    @patch(REQUEST)
    def test_update_error_02(self, _request):
        """
        異常系： 更新対象の予測結果が存在していないこと
        """
        data = UpdateBrandInfoRequest(
            brand_name="日野自動車",
            brand_code=7205,
            update_by="test_update",
        )
        _request.return_value = TestRequest(state=RequestId(), method="PUT")
        service = StockPriceService(self.db)

        with self.assertRaisesRegex(
            KeyError,
            "更新対象の予測結果データが存在しません。"
        ):
            service._update(_request, 1, data)

    @patch(STOCK_PRICE_BASE_PREDICTION)
    @patch(REQUEST)
    def test_update_success_01(self, _request, _stock_price_base_prediction):
        """
        異常系： 銘柄情報と予測結果が更新ができること
        """
        user_id = 1
        brand_code = 7203
        data = UpdateBrandInfoRequest(
            brand_name="トヨタ自動車",
            brand_code=brand_code,
            update_by="test_update",
        )
        _request.return_value = TestRequest(state=RequestId(), method="POST")
        _stock_price_base_prediction.return_value = (
            "[1, 2, 3]",
            "[2025-02-11, 2025-02-12, 2025-02-13]",
            "test.pth"
        )
        service = StockPriceService(self.db)
        service._update(_request, user_id, data)

        # 更新データ確認
        get_pred = service.get_prediction_info(user_id, brand_code)
        self.assertEqual(get_pred.future_predictions, "[1, 2, 3]")
        self.assertEqual(
            get_pred.days_list,
            "[2025-02-11, 2025-02-12, 2025-02-13]"
        )
        self.assertFalse(
            get_pred.create_at == get_pred.update_at
        )
        get_brand = self.expected_get_brand_info(user_id, brand_code)
        self.assertEqual(get_brand.learned_model_name, "test.pth")
        self.assertFalse(
            get_brand.create_at == get_brand.update_at
        )

    def test_save_error_01(self):
        """
        異常系： 登録・更新時に予期せぬエラーが発生すること
        """
        self.mock_db.commit.side_effect = SqlException("予期せぬエラー")
        with self.assertRaisesRegex(
            SqlException,
            "予期せぬエラー"
        ):
            StockPriceService(self.mock_db)._save(
                PredictionResultModel(),
                False
            )
        self.mock_db.rollback.assert_called_once()

    def test_delete_error_01(self):
        """
        異常系： 削除時に予期せぬエラーが発生すること
        """
        self.mock_db.commit.side_effect = SqlException("予期せぬエラー")
        with self.assertRaisesRegex(
            SqlException,
            "予期せぬエラー"
        ):
            StockPriceService(self.mock_db)._delete(
                PredictionResultModel()
            )
        self.mock_db.rollback.assert_called_once()

    def test_brand_info_and_prediction_result_delete_error_01(self):
        """
        異常系： 削除対象の銘柄情報が存在していないこと
        """
        service = StockPriceService(self.db)
        with self.assertRaisesRegex(
            NotFoundException,
            "削除対象の銘柄情報が見つかりません。"
        ):
            service._brand_info_and_prediction_result_delete(
                1, 1111
            )

    def test_brand_info_and_prediction_result_delete_error_02(self):
        """
        異常系： 削除対象の予測結果が存在していないこと
        """
        service = StockPriceService(self.db)
        with self.assertRaisesRegex(
            NotFoundException,
            "削除対象の予測結果データが見つかりません。"
        ):
            service._brand_info_and_prediction_result_delete(
                1, 7205
            )

    @patch(STOCK_PRICE_BASE_PREDICTION)
    @patch(REQUEST)
    def test_brand_info_and_prediction_result_delete_success_01(
        self, _request, _stock_price_base_prediction
    ):
        """
        正常系： 銘柄情報と予測結果の削除できること
        """
        # 事前にデータ登録
        user_id = 1
        brand_code = 1111
        data = CreateBrandInfoRequest(
            brand_name="テスト",
            brand_code=1111,
            user_id=1,
            create_by="test",
            update_by="test",
            is_valid=True
        )
        _request.return_value = TestRequest(state=RequestId(), method="POST")
        _stock_price_base_prediction.return_value = (
            "[1, 2, 3]",
            "[2025-02-11, 2025-02-12, 2025-02-13]",
            "test.pth"
        )
        service = StockPriceService(self.db)
        service._create(_request, data)

        # 登録されているか確認
        brand_info_check = service._exist_brand_info_check(user_id, data)
        prediction_result_check = service._exist_prediction_result_check(
            user_id, data
        )
        # 登録されていればNoneではない
        self.assertIsNotNone(brand_info_check)
        self.assertIsNotNone(prediction_result_check)

        # 対象のデータ削除
        service = StockPriceService(self.db)
        service._brand_info_and_prediction_result_delete(
            user_id, brand_code
        )

        expected_get_pred = service.get_prediction_info(user_id, brand_code)
        expected_get_brand = self.expected_get_brand_info(user_id, brand_code)
        # 削除されていれば存在してないのでNone
        self.assertIsNone(expected_get_pred)
        self.assertIsNone(expected_get_brand)
