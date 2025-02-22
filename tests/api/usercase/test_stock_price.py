
from unittest.mock import patch

from api.models.models import BrandModel, BrandInfoModel, PredictionResultModel
from api.endpoints.stock_price import StockPriceBase, brand
from api.schemas.test import RequestId, TestRequest
from api.usercase.stock_price import StockPriceService
from const.const import (
    HttpStatusCode, ErrorCode,
    PredictionResultConst, BrandInfoModelConst
)
from tests.api.endpoints.test_case import TestBase, TestBaseAPI
from utils.utils import Utils

# モック
REQUEST = 'fastapi.Request'
CHECK_BRAND_INFO = 'prediction.train.train.PredictionTrain.check_brand_info'
TRAIN_MAIN = 'prediction.train.train.PredictionTrain.main'
TEST_MAIN = 'prediction.test.test.PredictionTest.main'
STOCK_PRICE_BASE_PREDICTION = (
    'api.endpoints.stock_price.StockPriceBase.prediction'
)
GET_TEST_DB = 'tests.api.database.test_database.get_test_db'
GET_DB = 'api.databases.databases.get_db'

GET_STOCK_PRICE_PATH = "/get_stock_price"
GET_BRAND_INFO_LIST_PATH = "/brand_info_list"
GET_BRAND_PATH = "/brand"
CREATE_STOCK_PRICE_PATH = "/create_stock_price"
UPDATE_STOCK_PRICE_PATH = "/upadte_stock_price"
DELETE_BRAND_INFO_PATH = "/delete_stock_price"


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
        with self.assertRaises(KeyError):
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

    def test_get_prediction_info(self):
        """
        正常系： 予測情報取得できること
        """
        service = StockPriceService(self.db)
        result = service.get_prediction_info(1, 7203)

        self.assertEqual(result.brand_code, 7203)
