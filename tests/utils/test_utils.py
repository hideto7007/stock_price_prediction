import datetime
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import URL
from unittest.mock import AsyncMock, MagicMock, patch
import pytz
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String
from const.const import HttpStatusCode, LoggerConst
from tests.test_case import TestBase, TestBaseAPI
from utils.utils import Swagger, Utils


class TestUtils(TestBase):

    @patch("datetime.datetime")
    @patch("datetime.date")
    def test_datetime_success_01(
        self,
        mock_date,
        mock_datetime
    ):
        """
        正常系: システム時間取得
        - 現在時刻
        - 現在日時
        - 現在時間
        """
        expected_today = datetime.datetime(
            2025, 1, 1, 0, 0, 0,
            tzinfo=pytz.timezone('Asia/Tokyo')
        )
        expected_date = datetime.date(2025, 1, 1)
        expected_time = datetime.time(0, 0, 0)

        # モックの返り値を設定
        mock_datetime.now.return_value = expected_today
        mock_date.today.return_value = expected_date
        expected_today.time.return_value = expected_time

        # 検証
        self.assertEqual(Utils.today(), expected_today)
        self.assertEqual(Utils.date(), expected_date)
        self.assertEqual(Utils.time(), expected_time)

    def test_column_type_success_01(self):
        """
        正常系: 標準型を`sqlalchemy`の型に変換できること
        """
        # 文字列
        self.assertEqual(
            Utils.column_str("hoge"),
            Column(String, default="hoge").default.arg
        )

        # 整数値
        self.assertEqual(
            Utils.column_int(10),
            Column(Integer, default=10).default.arg
        )

        # 小数点
        self.assertEqual(
            Utils.column_float(10.1),
            Column(Float, default=10.1).default.arg
        )

        # 真偽値
        self.assertEqual(
            Utils.column_bool(True),
            Column(Boolean, default=True).default.arg
        )

        # 現在時刻
        d = datetime.datetime(2025, 3, 3, 0, 0, 0)
        self.assertEqual(
            Utils.column_datetime(d),
            Column(DateTime, default=d).default.arg
        )

    def test_column_type_success_02(self):
        """
        正常系: `sqlalchemy`型から標準型に変換できること
        """
        column_value1 = Column(String, default="test_value")
        self.assertEqual(Utils.str(column_value1.default.arg), "test_value")

        column_value2 = Column(Integer, default=10)
        self.assertEqual(Utils.int(column_value2.default.arg), 10)

        column_value3 = Column(Float, default=10.5)
        self.assertEqual(Utils.float(column_value3.default.arg), 10.5)

        column_value4 = Column(Boolean, default=True)
        self.assertEqual(Utils.bool(column_value4.default.arg), True)

        test_datetime5 = datetime.datetime(2024, 1, 1, 12, 30, 45)
        column_value = Column(DateTime, default=test_datetime5)
        self.assertEqual(Utils.datetime(
            column_value.default.arg), test_datetime5)

    def test_get_logger_file_name_success_01(self):
        """正常系: `logger.log`のファイル名が取得できる"""
        # URLをモック化
        mock_url = MagicMock(spec=URL)
        mock_url.__str__.return_value = (
            "http://localhost:8000/api/logs/access.log"
        )

        self.assertEqual(
            Utils.get_logger_file_name(mock_url),
            LoggerConst.MAIN_FILE_NAME.value
        )

    def test_get_logger_file_name_success_02(self):
        """正常系: `stock_price_logger.log`のファイル名が取得できる"""
        # URLをモック化
        mock_url = MagicMock(spec=URL)
        mock_url.__str__.return_value = (
            "http://localhost:8000/api/stock_price/access.log"
        )

        self.assertEqual(
            Utils.get_logger_file_name(mock_url),
            LoggerConst.STOCK_PRICE_FILE_NAME.value
        )

    def test_get_logger_file_name_success_03(self):
        """正常系: `login_logger.log`のファイル名が取得できる"""
        # URLをモック化
        mock_url = MagicMock(spec=URL)
        mock_url.__str__.return_value = (
            "http://localhost:8000/api/login/access.log"
        )

        self.assertEqual(
            Utils.get_logger_file_name(mock_url),
            LoggerConst.LOGIN_FILE_NAME.value
        )

    def test_get_logger_file_name_success_04(self):
        """正常系: `test_logger.log`のファイル名が取得できる"""
        # URLをモック化
        mock_url = MagicMock(spec=URL)
        mock_url.__str__.return_value = (
            "http://testserver/api/login/access.log"
        )

        self.assertEqual(
            Utils.get_logger_file_name(mock_url),
            LoggerConst.TEST_FILE_NAME.value
        )

    def test_get_logger_file_name_success_05(self):
        """正常系: `logger.log`のファイル名が取得できる"""
        # URLをモック化
        mock_url = MagicMock(spec=URL)
        mock_url.__str__.return_value = (
            "http://localhost"
        )

        self.assertEqual(
            Utils.get_logger_file_name(mock_url),
            LoggerConst.MAIN_FILE_NAME.value
        )


class TestSwagger(TestBaseAPI):

    # モック
    __GET_OAUTH2_SCHEME = 'api.middleware.OAuth2Middleware.get_oauth2_scheme'
    __IS_VALID_TOKEN = 'api.middleware.OAuth2Middleware.is_valid_token'

    def setUp(self):
        """テスト前のセットアップ"""
        super().setUp()
        self.get_oauth2_scheme_patch = patch(
            TestSwagger.__GET_OAUTH2_SCHEME,
            new_callable=AsyncMock
        )
        self.is_valid_token_patch = patch(TestSwagger.__IS_VALID_TOKEN)

        self.mock_get_oauth2_scheme = self.get_oauth2_scheme_patch.start()
        self.mock_is_valid_token = self.is_valid_token_patch.start()

        self.mock_get_oauth2_scheme.return_value = "token"
        self.mock_is_valid_token.return_value = {"auth": "token"}
        self.app = FastAPI()
        self.router = APIRouter()

    def test_generate_content_success_01(self):
        """正常系: `content生成`で`503`のレスポンス"""

        self.assertEqual(
            Swagger.swagger_responses({503: "test read time out."}),
            {
                503: {
                    "description": "タイムアウトエラー",
                    "content": {
                        "application/json": {
                            "example": {
                                "result": "test read time out."
                            }
                        }
                    }
                }
            }
        )

    def test_custom_openapi_initialization(self):
        # テスト用のAPI
        @self.router.get("/test")
        async def test_endpoint(test: str):
            return {"message": test}

        self.app.include_router(self.router)
        client = TestClient(self.app)
        result = Swagger.custom_openapi(self.app)

        # カバレッジ100%にするためのテスト
        res = client.get("/test", params={"test": "hoge"})
        self.assertEqual(res.status_code, HttpStatusCode.SUCCESS.value)
        self.assertEqual(res.json(), {"message": "hoge"})

        # OpenAPI スキーマが適用されることを確認
        self.assertIsNotNone(result)
        self.assertIn("components", result)
        self.assertIn("securitySchemes", result["components"])
        self.assertIn("BearerAuth", result["components"]["securitySchemes"])

        # BearerAuth の設定が正しくされているか
        bearer_auth = result["components"]["securitySchemes"]["BearerAuth"]
        self.assertEqual(bearer_auth["type"], "http")
        self.assertEqual(bearer_auth["scheme"], "Bearer")
        self.assertEqual(bearer_auth["description"],
                         "Enter your BearerAuth token to authenticate")

    def test_custom_openapi_already_set(self):
        """正常系: すでに OpenAPI スキーマが設定されている場合、そのまま返す"""
        mock_schema = {"openapi": "3.0.0"}
        self.app.openapi_schema = mock_schema

        result = Swagger.custom_openapi(self.app)

        # 変更されずに元のスキーマがそのまま返される
        self.assertEqual(result, mock_schema)
