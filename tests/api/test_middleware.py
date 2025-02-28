import asyncio
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI, Request, HTTPException
from starlette.datastructures import Headers
from api.middleware import OAuth2Middleware
from api.schemas.response import Content
from api.usercase.login import LoginService
from const.const import HttpStatusCode
from tests.test_case import TestBaseAPI


# モック
GET_OAUTH2_SCHEME = 'api.middleware.OAuth2Middleware.get_oauth2_scheme'
IS_VALID_TOKEN = 'api.middleware.OAuth2Middleware.is_valid_token'


class TestOAuth2Middleware1(TestBaseAPI):

    def _app(self):
        """FastAPIのインスタンス"""
        return FastAPI()

    def _middleware(self):
        """Middlewareのインスタンス"""
        return OAuth2Middleware(app=self._app())

    def get_path(self, token: str) -> str:
        """
        ユーザー情報取得APIのパス

        引数:
            token (str): トークン
        戻り値:
            str: urlパス
        """
        return self.get_login_path(
            f'read_users_me/{token}'
        )

    @patch.object(
        OAuth2Middleware,
        "get_oauth2_scheme",
        new_callable=AsyncMock
    )
    def test_dispatch_error_01(
        self,
        _get_oauth2_scheme
    ):
        """
        異常系: 認証トークンが必要
        """
        # エラーモック
        _get_oauth2_scheme.return_value = None
        # API実行
        response = self.get(self.get_path("mock_token"))
        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        expected_response = Content[str](
            result="認証トークンが必要です。"
        )
        self.assertEqual(
            response.json(),
            expected_response.model_dump()
        )

    @patch.object(
        OAuth2Middleware,
        "get_oauth2_scheme",
        new_callable=AsyncMock
    )
    def test_dispatch_error_02(
        self,
        _get_oauth2_scheme
    ):
        """
        異常系: 予期せぬエラー
        """
        # エラーモック
        _get_oauth2_scheme.side_effect = HTTPException(
            HttpStatusCode.SERVER_ERROR.value,
            "予期せぬエラー"
        )
        # API実行
        response = self.get(self.get_path("mock_token"))
        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        expected_response = Content[str](
            result="500: 予期せぬエラー"
        )
        self.assertEqual(
            response.json(),
            expected_response.model_dump()
        )

    @patch.object(
        OAuth2Middleware,
        "get_oauth2_scheme",
        new_callable=AsyncMock
    )
    @patch.object(
        OAuth2Middleware,
        "is_valid_token"
    )
    def test_dispatch_error_03(
        self,
        _is_valid_token,
        _get_oauth2_scheme
    ):
        """
        異常系: トークンが無効
        """
        # エラーモック
        _get_oauth2_scheme.return_value = "mock_token"
        _is_valid_token.return_value = None
        # API実行
        response = self.get(self.get_path("mock_token"))
        self.assertEqual(
            response.status_code,
            HttpStatusCode.UNAUTHORIZED.value
        )
        expected_response = Content[str](
            result="トークンが無効です。"
        )
        self.assertEqual(
            response.json(),
            expected_response.model_dump()
        )

    @patch.object(LoginService, "get_payload")
    def test_is_valid_token_success(self, _get_payload):
        """
        正常系: トークン取得できること
        """
        jwt = {
            "sub": "test_user",
            "exp": 1740408437
        }
        _get_payload.return_value = jwt
        self.assertEqual(
            OAuth2Middleware(self._app()).is_valid_token("mock_token"),
            jwt
        )

    def test_get_oauth2_scheme_valid_token(self):
        """
        正しいリクエストヘッダーがある場合、トークンを取得できる
        """
        request = AsyncMock(Request)
        request.headers = Headers({"Authorization": "Bearer valid_token"})

        token = asyncio.run(self._middleware().get_oauth2_scheme(request))
        self.assertEqual(token, "valid_token")

    def test_get_oauth2_scheme_no_token(self):
        """
        Authorization ヘッダーがない場合、None を返す
        """
        request = AsyncMock(Request)
        request.headers = Headers({})

        with self.assertRaises(HTTPException):
            asyncio.run(self._middleware().get_oauth2_scheme(request))

    def test_get_oauth2_scheme_invalid_token(self):
        """
        Authorization ヘッダーが不正な場合、HTTPException が発生する
        """
        request = AsyncMock(Request)
        request.headers = Headers({"Authorization": "InvalidToken"})

        with self.assertRaises(HTTPException):
            asyncio.run(self._middleware().get_oauth2_scheme(request))
