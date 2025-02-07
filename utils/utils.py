from typing import Any


class Swagger:
    """swagger出力で使うクラス"""

    @classmethod
    def generate_content(
        cls,
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

    @classmethod
    def swagger_responses(
        cls,
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
