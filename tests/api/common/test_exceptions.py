from unittest.mock import patch

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError

from api.common.exceptions import (
    ConflictException, CustomBaseException,
    ExpiredSignatureException, NotFoundException,
    SqlException
)
from api.schemas.login import CreateUserRequest
from api.schemas.response import Content
from api.usercase.login import LoginService
from const.const import HttpStatusCode

from tests.test_case import TestBaseAPI


class TestHttpExceptionHandler(TestBaseAPI):

    def get_path(self) -> str:
        """
        ログイン情報登録APIのパスで検証

        引数:
            なし
        戻り値:
            str: urlパス
        """
        return self.get_login_path(
            'register_user'
        )

    @patch.object(LoginService, "get_user_info")
    def test_exception_error_01(
        self,
        _get_user_info
    ):
        """
        異常系: `HTTPException`の例外
        """
        # データセット
        data = CreateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_password="Test12345!"
        ).model_dump()
        # エラーモック
        _get_user_info.side_effect = HTTPException(
            HttpStatusCode.SERVER_ERROR.value,
            "予期せぬエラー"
        )
        response = self.post(self.get_path(), data)
        expected_response = Content[str](
            result="500: 予期せぬエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_user_info")
    def test_exception_error_02(
        self,
        _get_user_info
    ):
        """
        異常系: `RequestValidationError`の例外
        """
        # データセット
        data = CreateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_password="Test12345!"
        ).model_dump()
        # エラーモック
        _get_user_info.side_effect = RequestValidationError(
            "バリデーションエラー"
        )
        response = self.post(self.get_path(), data)
        expected_response = Content[str](
            result="バリデーションエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.VALIDATION.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_user_info")
    def test_exception_error_03(
        self,
        _get_user_info
    ):
        """
        異常系: `TypeError`の例外
        """
        # データセット
        data = CreateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_password="Test12345!"
        ).model_dump()
        # エラーモック
        _get_user_info.side_effect = TypeError(
            "タイプエラー"
        )
        response = self.post(self.get_path(), data)
        expected_response = Content[str](
            result="タイプエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_user_info")
    def test_exception_error_04(
        self,
        _get_user_info
    ):
        """
        異常系: `KeyError`の例外
        """
        # データセット
        data = CreateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_password="Test12345!"
        ).model_dump()
        # エラーモック
        _get_user_info.side_effect = KeyError(
            "キーエラー"
        )
        response = self.post(self.get_path(), data)
        expected_response = Content[str](
            result="'キーエラー'"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_user_info")
    def test_exception_error_05(
        self,
        _get_user_info
    ):
        """
        異常系: `SqlException`の例外
        """
        # データセット
        data = CreateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_password="Test12345!"
        ).model_dump()
        # エラーモック
        _get_user_info.side_effect = SqlException(
            "sqlエラー"
        )
        response = self.post(self.get_path(), data)
        expected_response = Content[str](
            result="sqlエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_user_info")
    def test_exception_error_06(
        self,
        _get_user_info
    ):
        """
        異常系: `ConflictException`の例外
        """
        # データセット
        data = CreateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_password="Test12345!"
        ).model_dump()
        # エラーモック
        _get_user_info.side_effect = ConflictException(
            "重複エラー"
        )
        response = self.post(self.get_path(), data)
        expected_response = Content[str](
            result="重複エラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.CONFLICT.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_user_info")
    def test_exception_error_07(
        self,
        _get_user_info
    ):
        """
        異常系: `NotFoundException`の例外
        """
        # データセット
        data = CreateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_password="Test12345!"
        ).model_dump()
        # エラーモック
        _get_user_info.side_effect = NotFoundException(
            "存在しないエラー"
        )
        response = self.post(self.get_path(), data)
        expected_response = Content[str](
            result="存在しないエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.NOT_FOUND.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_user_info")
    def test_exception_error_08(
        self,
        _get_user_info
    ):
        """
        異常系: `AttributeError`の例外
        """
        # データセット
        data = CreateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_password="Test12345!"
        ).model_dump()
        # エラーモック
        _get_user_info.side_effect = AttributeError(
            "属性エラー"
        )
        response = self.post(self.get_path(), data)
        expected_response = Content[str](
            result="属性エラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_user_info")
    def test_exception_error_09(
        self,
        _get_user_info
    ):
        """
        異常系: `ExpiredSignatureException`の例外
        """
        # データセット
        data = CreateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_password="Test12345!"
        ).model_dump()
        # エラーモック
        _get_user_info.side_effect = ExpiredSignatureException(
            "有効期限切れエラー"
        )
        response = self.post(self.get_path(), data)
        expected_response = Content[str](
            result="有効期限切れエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.UNAUTHORIZED.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())

    @patch.object(LoginService, "get_user_info")
    def test_exception_error_10(
        self,
        _get_user_info
    ):
        """
        異常系: `CustomBaseException`の例外
        """
        # データセット
        data = CreateUserRequest(
            user_name="test",
            user_email="test@example.com",
            user_password="Test12345!"
        ).model_dump()
        # エラーモック
        _get_user_info.side_effect = CustomBaseException(
            "予期せぬエラー"
        )
        response = self.post(self.get_path(), data)
        expected_response = Content[str](
            result="予期せぬエラー"
        )

        self.assertEqual(
            response.status_code,
            HttpStatusCode.SERVER_ERROR.value
        )
        self.assertEqual(response.json(), expected_response.model_dump())
