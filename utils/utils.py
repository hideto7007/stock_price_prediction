import datetime
from typing import Any, cast
from starlette.datastructures import URL

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import pytz
from sqlalchemy import Column

from const.const import LoggerConst


class Utils:
    """アプリケーションに関するデータ取得"""

    @staticmethod
    def today() -> datetime.datetime:
        """日本時間の現在時刻取得"""
        return datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

    @staticmethod
    def date() -> datetime.date:
        """現在の日付を取得"""
        return datetime.date.today()

    @staticmethod
    def time() -> datetime.time:
        """現在の時間を取得"""
        return datetime.datetime.now().time()

    ###################
    # sqlalchemy型へ変換 #
    ###################

    @staticmethod
    def column_str(val: str) -> Column[str]:
        return cast(Column[str], val)

    @staticmethod
    def column_int(val: int) -> Column[int]:
        return cast(Column[int], val)

    @staticmethod
    def column_float(val: float) -> Column[float]:
        return cast(Column[float], val)

    @staticmethod
    def column_bool(val: bool) -> Column[bool]:
        return cast(Column[bool], val)

    @staticmethod
    def column_datetime(val: datetime.datetime) -> Column[datetime.datetime]:
        return cast(Column[datetime.datetime], val)

    ###################
    # pythonh標準型変換 #
    ###################

    @staticmethod
    def str(val: Column[str]) -> str:
        return cast(str, val)

    @staticmethod
    def int(val: Column[int]) -> int:
        return cast(int, val)

    @staticmethod
    def float(val: Column[float]) -> float:
        return cast(float, val)

    @staticmethod
    def bool(val: Column[bool]) -> bool:
        return cast(bool, val)

    @staticmethod
    def datetime(val: Column[datetime.datetime]) -> datetime.datetime:
        return cast(datetime.datetime, val)

    @staticmethod
    def get_logger_file_name(url: URL) -> Any:
        str_url = str(url).split("/")
        if len(str_url) < 5:
            return LoggerConst.MAIN_FILE_NAME.value
        domain = str_url[2]
        service_name = str_url[4]

        # 単体テスト時はテスト用のlogファイルに記載
        if domain == "testserver":
            logger_file_name = LoggerConst.TEST_FILE_NAME.value
        else:
            if service_name == "login":
                logger_file_name = LoggerConst.LOGIN_FILE_NAME.value
            elif service_name == "stock_price":
                logger_file_name = LoggerConst.STOCK_PRICE_FILE_NAME.value
            else:
                logger_file_name = LoggerConst.MAIN_FILE_NAME.value

        return logger_file_name


class Swagger:
    """swagger出力で使うクラス"""

    @staticmethod
    def generate_content(
        val: Any
    ) -> dict:
        """
            content生成

            引数:
                val (Any): レスポンスに返すデータ

            戻り値:
                dict: content定義
        """

        return {
            "application/json": {
                "example": {
                    "result": val
                }
            }
        }

    @staticmethod
    def swagger_responses(
        res_dict: dict
    ) -> dict[int | str, dict[str, Any]] | None:
        """
            共通レスポンス定義

            引数:
                user (UserCreate): リクエストボディ
                db (Session): dbインスタンス

            戻り値:
                dict[int | str, dict[str, Any]] | None: レスポンス定義
        """
        response_definitions = {}
        for status_code, v in res_dict.items():
            if status_code == 200:
                response_definitions[200] = {
                    "description": "成功時のレスポンス",
                    "content": Swagger.generate_content(
                        v
                    )
                }
            elif status_code == 400:
                response_definitions[400] = {
                    "description": "リクエスト又はバリデーションエラー",
                    "content": Swagger.generate_content(
                        v
                    )
                }
            elif status_code == 401:
                response_definitions[401] = {
                    "description": "認証エラー",
                    "content": Swagger.generate_content(
                        v
                    )
                }
            elif status_code == 409:
                response_definitions[409] = {
                    "description": "重複エラー",
                    "content": Swagger.generate_content(
                        v
                    )
                }
            elif status_code == 500:
                response_definitions[500] = {
                    "description": "サーバーエラー",
                    "content": Swagger.generate_content(
                        v
                    )
                }
            elif status_code == 503:
                response_definitions[503] = {
                    "description": "タイムアウトエラー",
                    "content": Swagger.generate_content(
                        v
                    )
                }
        return response_definitions

    @staticmethod
    def custom_openapi(app: FastAPI):
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title="Custom FastAPI",
            version="1.0.0",
            description=(
                "This is a custom OpenAPI schema with authentication applied"
            ),
            routes=app.routes,
        )

        if "securitySchemes" not in openapi_schema["components"]:
            openapi_schema["components"]["securitySchemes"] = {}

        openapi_schema["components"]["securitySchemes"]["BearerAuth"] = {
            "type": "http",
            "scheme": "Bearer",
            "bearerFormat": "",
            "description": "Enter your BearerAuth token to authenticate"
        }

        for _, methods in openapi_schema["paths"].items():
            for method in methods:
                methods[method]["security"] = [{"BearerAuth": []}]

        app.openapi_schema = openapi_schema
        return app.openapi_schema
