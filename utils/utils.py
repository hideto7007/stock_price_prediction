import datetime
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import pytz


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
