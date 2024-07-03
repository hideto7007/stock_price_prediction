import unittest
from fastapi.testclient import TestClient # type: ignore
from unittest.mock import patch

from api.main import app  # FastAPIアプリケーションが定義されているモジュールをインポート
from const.const import HttpStatusCode, ErrorCode

# モック
CHECK_BRAND_INFO = 'prediction.train.train.PredictionTrain.check_brand_info'
TRAIN_MAIN = 'prediction.train.train.PredictionTrain.main'
TEST_MAIN = 'prediction.test.test.PredictionTest.main'

# テストデータ
data = [100.0, 200.0, 300.0]
days = ["2023-07-01", "2023-07-02", "2023-07-03"]

client = TestClient(app)


class TestGetStockPrice(unittest.TestCase):

    def get_path(self, params="トヨタ自動車", user_id=1):
        return f"/get_stock_price?params={params}&user_id={user_id}"

    @patch(CHECK_BRAND_INFO)
    @patch(TRAIN_MAIN)
    @patch(TEST_MAIN)
    def test_get_stock_price_success_01(self, mock_main_test, mock_check_brand_info):
        """正常系: 取得データを返す trainは実行しない"""
        mock_check_brand_info.return_value = True
        mock_main_test.return_value = (data, days)

        response = client.get(self.get_path())

        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), {
            "status": HttpStatusCode.SUCCESS.value,
            "result": {
                "feature_stock_price": data,
                "days_list": days
            }
        })

    @patch(CHECK_BRAND_INFO)
    @patch(TRAIN_MAIN)
    @patch(TEST_MAIN)
    def test_get_stock_price_success_01(self, mock_main_test, mock_main_train, mock_check_brand_info):
        """正常系: 取得データを返す trainは実行する"""
        mock_check_brand_info.return_value = False
        mock_main_train.return_value = None
        mock_main_test.return_value = (data, days)

        response = client.get(self.get_path())

        self.assertEqual(mock_main_train.return_value, None)
        self.assertEqual(response.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(response.json(), {
            "status": HttpStatusCode.SUCCESS.value,
            "result": {
                "feature_stock_price": data,
                "days_list": days
            }
        })

    def test_get_stock_price_key_error(self):
        """異常系: 400 エラーチェック 存在しない銘柄"""

        response = client.get(self.get_path("ddd", 1))

        self.assertEqual(response.status_code, HttpStatusCode.BADREQUEST.value)
        self.assertEqual(response.json(), {
            "detail": [
                {
                    "code": ErrorCode.CHECK_EXIST.value,
                    "message": "'対象の銘柄は存在しません'"
                }
            ]
        })

    @patch(CHECK_BRAND_INFO)
    def test_get_stock_price_server_error(self, mock_check_brand_info):
        """異常系: 500 エラーチェック"""
        mock_check_brand_info.side_effect = Exception("Server error")

        response = client.get(self.get_path())

        self.assertEqual(response.status_code, HttpStatusCode.SERVER_ERROR.value)
        self.assertEqual(response.json(), {
            "detail": [
                {
                    "code": ErrorCode.SERVER_ERROR.value,
                    "message": "Server error"
                }
            ]
        })
