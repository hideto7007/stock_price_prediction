from typing import Any
from const.const import HttpStatusCode


class Swagger:
    """swagger出力で使うクラス"""

    @classmethod
    def generate_content_succss(
        cls,
        status: HttpStatusCode,
        val: Any
    ) -> dict:
        """
            content成功部分生成

            引数:
                status (HttpStatusCode): ステータスコード
                val (Any): レスポンスに返すデータ

            戻り値:
                dict: content定義
        """

        return {
            "application/json": {
                "example": {
                    "status_code": status,
                    "data": val
                }
            }
        }

    @classmethod
    def generate_content_error(
        cls,
        status: HttpStatusCode,
        val: Any
    ) -> dict:
        """
            contentエラー部分生成

            引数:
                status (HttpStatusCode): ステータスコード
                val (Any): レスポンスに返すデータ

            戻り値:
                dict: content定義
        """

        return {
            "application/json": {
                "example": {
                    "status_code": status,
                    "detail": val
                }
            }
        }

    @classmethod
    def swagger_responses(
        cls,
        res_dict: dict
    ) -> dict[int, dict[str, Any]]:
        """
            共通レスポンス定義

            引数:
                user (UserCreate): リクエストボディ
                db (Session): dbインスタンス

            戻り値:
                dict[int, dict[str, Any]]: レスポンス定義
        """
        response_definitions = {}
        for status_code, v in res_dict.items():
            if status_code == 200:
                response_definitions[200] = {
                    "description": "成功時のレスポンス",
                    "content": Swagger.generate_content_succss(
                        HttpStatusCode.SUCCESS,
                        v
                    )
                }
            elif status_code == 400:
                response_definitions[400] = {
                    "description": "リクエスト又はバリデーションエラー",
                    "content": Swagger.generate_content_error(
                        HttpStatusCode.BADREQUEST,
                        v
                    )
                }
            elif status_code == 401:
                response_definitions[401] = {
                    "description": "認証エラー",
                    "content": Swagger.generate_content_error(
                        HttpStatusCode.UNAUTHORIZED,
                        v
                    )
                }
            elif status_code == 409:
                response_definitions[409] = {
                    "description": "重複エラー",
                    "content": Swagger.generate_content_error(
                        HttpStatusCode.CONFLICT,
                        v
                    )
                }
            elif status_code == 500:
                response_definitions[500] = {
                    "description": "サーバーエラー",
                    "content": Swagger.generate_content_error(
                        HttpStatusCode.SERVER_ERROR,
                        v
                    )
                }
            elif status_code == 503:
                response_definitions[503] = {
                    "description": "タイムアウトエラー",
                    "content": Swagger.generate_content_error(
                        HttpStatusCode.TIMEOUT,
                        v
                    )
                }
        return response_definitions
